"""LLM wrapper — agentic loop calls via litellm.

Handles the interactive agentic loop (tool-calling, multi-turn). Provider
selection for the Distiller's constrained decoding path is in agent/providers.py
(openai SDK directly, per ADR-008 §M2). This module handles everything else.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable

# Quiet down Bedrock/SageMaker missing botocore warnings during import
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

import litellm

logger = logging.getLogger(__name__)

from silica.config import CONFIG

# Suppress litellm's verbose logging by default
litellm.suppress_debug_info = True
litellm.drop_params = True


def retry_transient(fn, exceptions: tuple, attempts: int = 3, base_delay: float = 1.0, jitter: float = 0.0):
    """Call fn(), retrying on transient exceptions with exponential backoff.

    Sleeps base_delay * 2**attempt (+ uniform jitter) between attempts and
    re-raises the last exception once attempts are exhausted. The single
    retry policy for every LLM call site (litellm and openai SDK alike).
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except exceptions as e:
            if attempt == attempts:
                logger.error("Transient error, %d attempts exhausted: %s", attempts, e)
                raise
            delay = base_delay * (2 ** attempt) + (random.uniform(0, jitter) if jitter else 0.0)
            logger.warning(
                "Transient error (attempt %d/%d): %s. Retrying in %.1fs...",
                attempt, attempts, e, delay,
            )
            time.sleep(delay)


def openrouter_routing() -> dict | None:
    """OpenRouter `extra_body` provider-routing block from CONFIG, or None.

    CONFIG.openrouter_provider is a comma-separated list of provider names
    pinned as the routing `order`. `allow_fallbacks` is False: an explicit pin
    means "these providers or fail" — silently bouncing to an unpinned (maybe
    rate-limited) provider is exactly the surprise this knob exists to prevent.
    Shared by both LLM paths — litellm (call_llm) and the openai SDK
    (agent/providers.py) — so the pin applies everywhere openrouter is used.
    """
    order = [p.strip() for p in CONFIG.openrouter_provider.split(",") if p.strip()]
    return {"provider": {"order": order, "allow_fallbacks": False}} if order else None


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    args: dict


@dataclass
class LLMResponse:
    """Structured response from the LLM."""

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    assistant_message: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    reasoning: str | None = None
    finish_reason: str | None = None



def call_llm(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int | None = None,
    response_format=None,
    on_delta: Callable[[str, str], None] | None = None,
) -> LLMResponse:
    """Call the LLM with function-calling support.

    Args:
        model: litellm model string (e.g. "openrouter/anthropic/claude-sonnet-4-20250514")
        messages: conversation history in OpenAI format
        tools: list of tool JSON schemas (OpenAI function format)
        max_tokens: optional maximum tokens to generate
        on_delta: optional (chunk_type, content) sink; when given the call streams,
            emitting "reasoning"/"text" deltas as they arrive (plus a "reset" at the
            start of each attempt, so a mid-stream retry can clear any preview).
            The final LLMResponse is identical to the non-streaming path.

    Returns:
        LLMResponse with either text or tool_calls populated
    """
    if CONFIG.verbose:
        tool_count = len(tools) if tools else 0
        logger.info("LLM call: model=%s | msg=%d | tools=%d", model, len(messages), tool_count)

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens if max_tokens is not None else int(os.getenv("MAX_TOKENS", "256000")),
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if response_format is not None:
        kwargs["response_format"] = response_format
    if model.startswith("openrouter/") and (CONFIG.show_thinking or CONFIG.verbose):
        kwargs["include_reasoning"] = True
    if model.startswith("openrouter/") and (rt := openrouter_routing()):
        kwargs["extra_body"] = rt

    kwargs["timeout"] = 120.0

    _TRANSIENT = (
        litellm.Timeout,
        litellm.APIConnectionError,
        litellm.RateLimitError,
        litellm.ServiceUnavailableError,
        litellm.BadGatewayError,
    )
    if on_delta is None:
        response = retry_transient(lambda: litellm.completion(**kwargs), _TRANSIENT)
    else:
        def _stream_once():
            on_delta("reset", "")
            chunks = []
            for chunk in litellm.completion(**kwargs, stream=True):
                chunks.append(chunk)
                try:
                    delta = chunk.choices[0].delta
                except (IndexError, AttributeError):
                    continue  # usage-only / malformed trailing chunk
                r = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if isinstance(r, str) and r:
                    on_delta("reasoning", r)
                c = getattr(delta, "content", None)
                if isinstance(c, str) and c:
                    on_delta("text", c)
            # Reassemble the canonical response (content, tool_calls, usage) so
            # everything below is identical to the non-streaming path.
            return litellm.stream_chunk_builder(chunks, messages=messages)

        response = retry_transient(_stream_once, _TRANSIENT)
        if response is None:
            raise RuntimeError(f"LLM stream from {model} produced no chunks")

    choice = response.choices[0]
    message = choice.message
    finish_reason = getattr(choice, "finish_reason", None)

    # Extract reasoning
    reasoning = getattr(message, "reasoning_content", None)
    if not isinstance(reasoning, str):
        reasoning = getattr(message, "reasoning", None)
    if not isinstance(reasoning, str) and isinstance(message, dict):
        reasoning = message.get("reasoning_content") or message.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = None

    blocks = getattr(message, "thinking_blocks", None)
    if not reasoning and isinstance(blocks, list):
        reasoning = "\n".join(b.get("thinking", "") for b in blocks if isinstance(b, dict))

    # Build the assistant message dict for conversation history
    assistant_msg: dict = {"role": "assistant"}
    if message.content:
        assistant_msg["content"] = message.content
    if reasoning:
        assistant_msg["reasoning_content"] = reasoning
    if isinstance(blocks, list):
        assistant_msg["thinking_blocks"] = blocks

    # Parse tool calls and build sanitized history
    parsed_calls: list[ToolCall] = []
    if message.tool_calls:
        assistant_msg_tool_calls = []
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
                valid_args_str = tc.function.arguments
            except json.JSONDecodeError:
                args = {}
                valid_args_str = "{}"  # Sanitize to prevent API rejection
                logger.warning(
                    "Failed to parse tool args for %s: %s",
                    tc.function.name,
                    tc.function.arguments,
                )
            
            parsed_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, args=args)
            )
            assistant_msg_tool_calls.append({
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": valid_args_str},
            })
            
        assistant_msg["tool_calls"] = assistant_msg_tool_calls

    if CONFIG.verbose:
        text_preview = (message.content or "")[:80].replace("\n", " ")
        logger.info(
            "LLM resp: finish=%s | tool_calls=%d | text=%r",
            finish_reason,
            len(parsed_calls),
            text_preview + ("…" if len(message.content or "") > 80 else ""),
        )

    return LLMResponse(
        text=message.content,
        tool_calls=parsed_calls,
        assistant_message=assistant_msg,
        usage=dict(response.usage) if response.usage else {},
        reasoning=reasoning,
        finish_reason=finish_reason,
    )
