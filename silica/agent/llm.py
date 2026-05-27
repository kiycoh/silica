"""LLM wrapper — agentic loop calls via litellm.

Handles the interactive agentic loop (tool-calling, multi-turn). Provider
selection for the Distiller's constrained decoding path is in agent/providers.py
(openai SDK directly, per ADR-008 §M2). This module handles everything else.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

# Quiet down Bedrock/SageMaker missing botocore warnings during import
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

import litellm

logger = logging.getLogger(__name__)

from silica.config import CONFIG

# Suppress litellm's verbose logging by default
litellm.suppress_debug_info = True


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
) -> LLMResponse:
    """Call the LLM with function-calling support.

    Args:
        model: litellm model string (e.g. "openrouter/anthropic/claude-sonnet-4-20250514")
        messages: conversation history in OpenAI format
        tools: list of tool JSON schemas (OpenAI function format)
        max_tokens: optional maximum tokens to generate

    Returns:
        LLMResponse with either text or tool_calls populated
    """
    if CONFIG.verbose:
        litellm.suppress_debug_info = False
        logger.info(
            "[DEBUG LLM Request]: Model: %s | Messages count: %d | Enabled Tools: %s",
            model,
            len(messages),
            [t["function"]["name"] for t in tools] if tools else [],
        )

    kwargs: dict = {
        "model": model,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    try:
        response = litellm.completion(**kwargs)
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        raise

    choice = response.choices[0]
    message = choice.message
    finish_reason = getattr(choice, "finish_reason", None)

    # Build the assistant message dict for conversation history
    assistant_msg: dict = {"role": "assistant"}
    if message.content:
        assistant_msg["content"] = message.content

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

    reasoning = getattr(message, "reasoning_content", None)
    if not reasoning:
        blocks = getattr(message, "thinking_blocks", None)
        if blocks:
            reasoning = "\n".join(b.get("thinking", "") for b in blocks if isinstance(b, dict))

    if CONFIG.verbose:
        logger.info(
            "[DEBUG LLM Response]: Text: %s | Tool Calls: %s | Usage: %s | Reasoning: %s",
            message.content,
            [(tc.name, tc.args) for tc in parsed_calls],
            dict(response.usage) if response.usage else {},
            reasoning,
        )

    return LLMResponse(
        text=message.content,
        tool_calls=parsed_calls,
        assistant_message=assistant_msg,
        usage=dict(response.usage) if response.usage else {},
        reasoning=reasoning,
        finish_reason=finish_reason,
    )
