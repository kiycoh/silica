# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

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
import threading
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


# Run-wide adaptive pacing. A 429 anywhere lifts a floor delay that is slept
# before the *first* attempt of every later call this process makes, so we back
# off an upstream rate limit instead of hammering it. Per-process = per-run for
# the CLI; the TUI and the GUI server keep it for their (long) lifetime, so the
# floor also *halves* on every clean first-try success — one bad 429 episode
# must not slow every later message of a day-long GUI session by 20s.
_run_cooldown = 0.0
_COOLDOWN_STEP = 2.0   # seconds added to the floor per 429
_COOLDOWN_CAP = 20.0   # ceiling on the floor delay
_RATE_LIMIT_ATTEMPTS = 6  # 429s get more tries than other transients (backoff to ~1min)


def retry_transient(fn, exceptions: tuple, attempts: int = 3, base_delay: float = 1.0, jitter: float = 0.0, cancel: threading.Event | None = None):
    """Call fn(), retrying on transient exceptions with exponential backoff.

    Sleeps base_delay * 2**attempt (+ uniform jitter) between attempts and
    re-raises the last exception once attempts are exhausted. The single
    retry policy for every LLM call site (litellm and openai SDK alike).

    Rate limits (HTTP 429) are treated specially: they get _RATE_LIMIT_ATTEMPTS
    tries (an upstream limit clears on the order of seconds), and each one lifts
    a run-wide cooldown paced before the next call so the whole run slows down
    rather than repeatedly re-hitting the limit.

    `cancel` marks the call abandoned (e.g. Ctrl+C orphaned the worker running
    it): once set, the in-flight attempt still finishes but no further retry is
    scheduled, and a backoff sleep wakes early. Without it an orphaned worker
    keeps hammering the API for minutes and can outlive the interpreter.
    """
    global _run_cooldown
    ceiling = attempts
    for attempt in range(1, max(attempts, _RATE_LIMIT_ATTEMPTS) + 1):
        if attempt == 1 and _run_cooldown:
            time.sleep(_run_cooldown)  # pace the start of every call once a 429 was seen
        try:
            result = fn()
            if attempt == 1 and _run_cooldown:  # clean first try → upstream healthy, decay the floor
                _run_cooldown = _run_cooldown / 2 if _run_cooldown > 0.5 else 0.0
            return result
        except exceptions as e:
            if getattr(e, "status_code", None) == 429:
                _run_cooldown = min(_run_cooldown + _COOLDOWN_STEP, _COOLDOWN_CAP)
                ceiling = _RATE_LIMIT_ATTEMPTS
            if cancel is not None and cancel.is_set():
                logger.info("Call abandoned; dropping retries: %s", e)
                raise
            if attempt >= ceiling:
                logger.error("Transient error, %d attempts exhausted: %s", attempt, e)
                raise
            delay = base_delay * (2 ** attempt) + (random.uniform(0, jitter) if jitter else 0.0)
            logger.warning(
                "Transient error (attempt %d/%d): %s. Retrying in %.1fs...",
                attempt, ceiling, e, delay,
            )
            if cancel is not None:
                if cancel.wait(delay):  # backoff sleep that wakes on abandonment
                    logger.info("Call abandoned during backoff; dropping retries: %s", e)
                    raise
            else:
                time.sleep(delay)


_LOCAL_LLM_TIMEOUT = 130.0  # wall-clock backstop we enforce ourselves (> the litellm
# timeout below, so litellm's own timeout wins if it ever fires). litellm's `timeout`
# kwarg does NOT fire on a provider that accepts the request then never sends a body
# — observed: OpenRouter holding an ESTAB socket idle ~58min, zero retries, the whole
# process wedged on one call. Kept 10s above the 120s litellm timeout so litellm fires
# first on a normal timeout and this only catches the silent-hang case.


def run_with_deadline(fn, timeout: float, on_timeout, *, catch: type = Exception):
    """Run fn() on a daemon thread, joining up to `timeout` seconds.

    Past the deadline, raise `on_timeout()`; if fn raised (of type `catch`),
    re-raise it on the caller thread; otherwise return fn()'s value. The only
    wall-clock bound we control — a transport read-timeout can silently not fire
    when the provider trickles keep-alive bytes.

    ponytail: on timeout the worker thread is abandoned (daemon) — a blocked
    C-level socket read can't be force-cancelled. Bounded by the caller's retry
    cap / single-turn use; swap for a cancellable HTTP client if abandoned
    threads ever pile up.
    """
    box: dict = {}

    def _work():
        try:
            box["r"] = fn()
        except catch as e:  # noqa: BLE001 - carried to the calling thread
            box["e"] = e

    th = threading.Thread(target=_work, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise on_timeout()
    if "e" in box:
        raise box["e"]
    return box["r"]


def _bounded(fn, timeout: float, model: str):
    """Run fn() but raise litellm.Timeout if it exceeds `timeout` seconds, routing
    the silent-hang case into the normal transient-retry path (see _LOCAL_LLM_TIMEOUT)."""
    return run_with_deadline(
        fn, timeout,
        lambda: litellm.Timeout(
            message=f"local wall-clock timeout after {timeout:.0f}s (provider sent no response)",
            model=model, llm_provider=model.split("/", 1)[0]),
        catch=BaseException,
    )


def _bounded_stream(make_iter, per_chunk_timeout: float, model: str):
    """Yield chunks from make_iter()'s stream, raising litellm.Timeout if any
    single gap (including connecting + the first chunk) exceeds per_chunk_timeout.

    The streaming twin of _bounded: the non-stream call's silent-hang mode
    (provider accepts the request then never sends a body) shows up on the stream
    path as a blocking next() that never returns. A per-chunk deadline catches
    that without capping a healthy long stream — the clock resets on every chunk.

    ponytail: pump thread is daemon and abandoned on timeout, same trade as
    _bounded; swap for a cancellable HTTP client if abandoned threads pile up.
    """
    import queue

    q: "queue.Queue" = queue.Queue()
    _DONE = object()

    def _pump():
        try:
            for c in make_iter():
                q.put(("chunk", c))
        except BaseException as e:  # noqa: BLE001 - carried to the consumer
            q.put(("err", e))
        finally:
            q.put(("done", _DONE))

    threading.Thread(target=_pump, daemon=True, name="llm-stream").start()
    while True:
        try:
            kind, payload = q.get(timeout=per_chunk_timeout)
        except queue.Empty:
            raise litellm.Timeout(
                message=f"local wall-clock timeout after {per_chunk_timeout:.0f}s (stream stalled)",
                model=model, llm_provider=model.split("/", 1)[0])
        if kind == "err":
            raise payload
        if kind == "done":
            return
        yield payload


def openrouter_routing(provider_list: str | None = None) -> dict | None:
    """OpenRouter `extra_body` provider-routing block, or None.

    `provider_list` is a comma-separated list of provider names pinned as the
    routing `order`; defaults to CONFIG.openrouter_provider. The distiller path
    passes CONFIG.openrouter_provider_distiller for its own pin. `allow_fallbacks`
    is False: an explicit pin means "these providers or fail" — silently bouncing
    to an unpinned (maybe rate-limited) provider is exactly the surprise this knob
    exists to prevent. Shared by both LLM paths — litellm (call_llm) and the
    openai SDK (agent/providers.py) — so the pin applies everywhere openrouter is used.
    """
    raw = CONFIG.openrouter_provider if provider_list is None else provider_list
    order = [p.strip() for p in raw.split(",") if p.strip()]
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


def _split_arg_objects(cid: str, name: str, arg_str: str) -> list[tuple[str, dict, str]]:
    """Yield (id, args, wire_str) for one raw tool-call arguments string.

    Normally one object in, one out. But some OpenAI-compatible backends can't
    emit parallel tool_calls, so a model wanting N calls concatenates N JSON
    objects into one blob (e.g. '{"name":"a"}{"name":"b"}'). Fan those back out
    with distinct ids. Unsalvageable args degrade to {} as before.
    """
    s = (arg_str or "").strip()
    if not s:
        return [(cid, {}, "{}")]
    try:
        return [(cid, json.loads(s), s)]
    except json.JSONDecodeError:
        pass
    dec, objs, i = json.JSONDecoder(), [], 0
    while i < len(s):
        try:
            obj, end = dec.raw_decode(s, i)
        except json.JSONDecodeError:
            break
        objs.append(obj)
        i = end
        while i < len(s) and s[i].isspace():
            i += 1
    if not objs:
        logger.warning("Failed to parse tool args for %s: %s", name, arg_str)
        return [(cid, {}, "{}")]
    if len(objs) == 1:
        return [(cid, objs[0], json.dumps(objs[0]))]
    return [(f"{cid}_{k}", o, json.dumps(o)) for k, o in enumerate(objs)]


def expand_tool_calls(
    raw: list[tuple[str, str, str]],
) -> tuple[list[ToolCall], list[dict]]:
    """Parse (id, name, arguments) triples into ToolCalls + sanitized wire dicts.

    Fans out concatenated-JSON blobs into separate calls (see _split_arg_objects)
    so the agent loop dispatches each. Returned wire dicts always carry valid
    JSON arguments, keeping the assistant/tool message pairing API-safe.
    """
    parsed: list[ToolCall] = []
    wire: list[dict] = []
    for cid, name, arg_str in raw:
        for sub_id, obj, obj_str in _split_arg_objects(cid, name, arg_str):
            parsed.append(ToolCall(id=sub_id, name=name, args=obj))
            wire.append(
                {"id": sub_id, "type": "function",
                 "function": {"name": name, "arguments": obj_str}}
            )
    return parsed, wire


def build_assistant_message(
    content: str | None, tool_calls_raw: list[tuple[str, str, str]] | None,
) -> tuple[dict, list[ToolCall]]:
    """Assemble the assistant history dict + parsed ToolCalls both provider paths
    build identically: {"role": "assistant"} (+content if any, +tool_calls if any).

    Callers add path-specific keys (reasoning_content, thinking_blocks) afterwards.
    """
    msg: dict = {"role": "assistant"}
    if content:
        msg["content"] = content
    parsed: list[ToolCall] = []
    if tool_calls_raw:
        parsed, msg["tool_calls"] = expand_tool_calls(tool_calls_raw)
    return msg, parsed


def call_llm(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int | None = None,
    response_format=None,
    temperature: float | None = None,
    on_delta: Callable[[str, str], None] | None = None,
    openrouter_provider: str | None = None,
    cancel: threading.Event | None = None,
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
        cancel: optional abandonment flag, forwarded to retry_transient — set it
            when nobody is waiting on this call anymore so retries stop.

    Returns:
        LLMResponse with either text or tool_calls populated
    """
    if CONFIG.verbose:
        tool_count = len(tools) if tools else 0
        logger.info("LLM call: model=%s | msg=%d | tools=%d", model, len(messages), tool_count)

    from silica.agent.providers import clamp_max_tokens  # lazy: providers.py imports this module

    input_chars = len(str(messages)) + (len(str(tools)) if tools else 0)
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": clamp_max_tokens(model.split("/", 1)[0], model, max_tokens, input_chars),
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if response_format is not None:
        kwargs["response_format"] = response_format
    if temperature is not None:
        kwargs["temperature"] = temperature
    if model.startswith("openrouter/") and (CONFIG.show_thinking or CONFIG.verbose):
        kwargs["include_reasoning"] = True
    if model.startswith("openrouter/") and (rt := openrouter_routing(openrouter_provider)):
        kwargs["extra_body"] = rt

    kwargs["timeout"] = 120.0  # litellm's own (fires first if it works); _bounded is the backstop

    _TRANSIENT = (
        litellm.Timeout,
        litellm.APIConnectionError,
        litellm.RateLimitError,
        litellm.ServiceUnavailableError,
        litellm.BadGatewayError,
    )
    if on_delta is None:
        response = retry_transient(
            lambda: _bounded(lambda: litellm.completion(**kwargs), _LOCAL_LLM_TIMEOUT, model),
            _TRANSIENT, cancel=cancel)
    else:
        def _stream_once():
            on_delta("reset", "")
            chunks = []
            _stream = _bounded_stream(
                lambda: litellm.completion(**kwargs, stream=True),
                _LOCAL_LLM_TIMEOUT, model)
            for chunk in _stream:
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

        response = retry_transient(_stream_once, _TRANSIENT, cancel=cancel)
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
    raw = ([(tc.id, tc.function.name, tc.function.arguments) for tc in message.tool_calls]
           if message.tool_calls else None)
    assistant_msg, parsed_calls = build_assistant_message(message.content, raw)
    if reasoning:
        assistant_msg["reasoning_content"] = reasoning
    if isinstance(blocks, list):
        assistant_msg["thinking_blocks"] = blocks

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
