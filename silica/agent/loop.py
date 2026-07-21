# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""The agentic loop — the core of Silica.

This is the 'while True' from SILICA.md §8.1:
  loop:
    response = LLM(system_prompt, message_history, tool_schemas)
    if response has tool_calls:
        for each tool_call:
            result = execute_tool(name, args)
            append tool_result to history
        continue  (re-call LLM with results)
    else:
        return response text to user

Everything else (streaming, TUI, context compression) is ergonomics
around this nucleus. Build this first, then ergonomics.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Any
import concurrent.futures as _cf
import threading
import time
import logging
import json

if TYPE_CHECKING:
    from silica.kernel.progress import ProgressLedger

import silica.agent.bus as _bus_mod
from silica.agent.events import (
    ToolStartEvent,
    ToolCompleteEvent,
    ToolErrorEvent,
    ReasoningEvent,
    RenderEvent,
    ThinkingStartEvent,
    ThinkingEndEvent,
    LLMStreamEvent,
)
from silica.agent.llm import call_llm
from silica.agent.compaction import eager_stub
from silica.agent.concurrency import worker_slot
from silica.agent.constraints import AgentConstraints
from silica.tools import TOOLS, Tool
from contextlib import nullcontext

logger = logging.getLogger(__name__)


def _topic_for(event: RenderEvent) -> str | None:
    if isinstance(event, ToolStartEvent):
        return "agent/tool_start"
    if isinstance(event, ToolCompleteEvent):
        return "agent/tool_complete"
    if isinstance(event, ToolErrorEvent):
        return "agent/tool_error"
    if isinstance(event, (ThinkingStartEvent, ThinkingEndEvent)):
        return "agent/thinking"
    if isinstance(event, ReasoningEvent):
        return "agent/reasoning"
    if isinstance(event, LLMStreamEvent):
        return "agent/stream"
    return None


def _is_tool_failure(result: Any) -> bool:
    """Helper to detect if a tool result indicates a failure."""
    if not result:
        return False
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict) and "error" in parsed:
                return True
        except Exception:
            pass
        # A non-JSON / non-error-keyed string result is a successful tool
        # output, not a failure. Substring-sniffing for "error"/"failed" here
        # misclassifies legitimate content (grep hits, "0 errors" reports).
    elif isinstance(result, dict) and "error" in result:
        return True
    return False


ToolProgressCallback = Callable[[RenderEvent], None] | None


def run_agent(
    messages: list[dict],
    model: str,
    tool_progress_callback: ToolProgressCallback = None,
    progress: "ProgressLedger | None" = None,
    cancel_token: "threading.Event | None" = None,
    constraints: "AgentConstraints | None" = None,
) -> str:
    """Execute the agentic loop until the model produces a text response.

    The loop calls the LLM, dispatches any tool calls, appends results,
    and re-calls until the model responds with text (no tool calls).

    Args:
        messages: mutable conversation history (modified in-place)
        model: litellm model string
        tool_progress_callback: callback for tool progress events

    Returns:
        The model's final text response
    """
    # Effective tool registry: full global, or the constrained subset.
    if constraints is not None:
        allowed: dict[str, "Tool"] = {
            name: TOOLS[name] for name in constraints.tools if name in TOOLS
        }
    else:
        # Non-ambient authority: the main agent's default toolset excludes
        # sensitive tools (ADR-0009 / ADR-0015) and pipeline internals the FSM
        # drives programmatically. Both are reachable only when a caller names
        # them in AgentConstraints.tools.
        allowed = {n: t for n, t in TOOLS.items() if not t.sensitive and not t.internal}

    # Collect tool schemas for the LLM
    schemas = [t.json_schema() for t in allowed.values()] if allowed else None

    effective_model = (
        constraints.model if (constraints is not None and constraints.model) else model
    )

    iteration = 0
    max_iterations = (
        constraints.max_iterations
        if (constraints is not None and constraints.max_iterations is not None)
        else 20
    )  # Hard safety cap lowered from 50

    # Track consecutive failures for the same (tool_name, args) pair
    # Key: (tool_name, args_json_string)
    # Value: consecutive failure count
    consecutive_failures: dict[tuple[str, str], int] = {}

    def _emit(event: RenderEvent) -> None:
        """Best-effort event emission to callback and bus."""
        if tool_progress_callback is not None:
            try:
                tool_progress_callback(event)
            except Exception as exc:
                logger.debug("tool_progress_callback error (swallowed): %s", exc)
        topic = _topic_for(event)
        if topic is not None:
            _bus_mod.BUS.publish(topic, event)

    # Set once the main thread stops waiting on the LLM (Ctrl+C or normal return).
    # The LLM runs on a detached daemon thread that keeps streaming deltas after an
    # interrupt; this gate stops those late deltas from re-opening the live region
    # and printing thinking below "(interrupted)". Also passed as `cancel` to abort
    # retries. Cleared at the top of each iteration.
    _abandon = threading.Event()

    def _stream_delta(chunk_type: str, content: str) -> None:
        # Called from the LLM worker thread; `iteration` reads the current loop pass.
        if _abandon.is_set():
            return  # interrupted/abandoned — stop feeding the renderer
        _emit(LLMStreamEvent(chunk_type=chunk_type, content=content, iteration=iteration))

    # Streaming is a TUI ergonomic: only the interactive main loop gets it —
    # constrained (worker/batch) runs stay on the plain non-streaming call.
    # The kwarg is only passed when active, so call_llm test doubles only need
    # the bare signature plus cancel=None.
    _llm_kwargs: dict = {"tools": None}
    if tool_progress_callback is not None and constraints is None:
        _llm_kwargs["on_delta"] = _stream_delta

    while iteration < max_iterations:
        if cancel_token is not None and cancel_token.is_set():
            logger.info("Agent loop cancelled at iteration %d", iteration)
            return "(silica: cancelled)"
        iteration += 1
        logger.debug("Agent loop iteration %d", iteration)

        _emit(ThinkingStartEvent(iteration=iteration))
        try:
            # Run the (synchronous, potentially slow) LLM call on a *daemon*
            # thread so a Ctrl+C on the main thread raises KeyboardInterrupt out
            # of _future.result() instead of being trapped in a C-level network
            # recv(). Daemon matters: a non-daemon orphan (the old throwaway
            # ThreadPoolExecutor worker) gets joined at interpreter shutdown,
            # hanging exit for minutes while its retries die against executors
            # already flagged shut ("cannot schedule new futures after shutdown").
            # The finally sets `_abandon` so retry_transient stops rescheduling
            # once nobody is waiting (harmless on success: the call already
            # returned). ponytail: sync litellm can't abort the in-flight HTTP
            # request — best we can do is stop waiting and stop retrying.
            slot = worker_slot() if constraints is not None else nullcontext()
            with slot:
                _abandon.clear()
                _llm_kwargs["tools"] = schemas
                _llm_kwargs["cancel"] = _abandon
                _future: _cf.Future = _cf.Future()

                def _llm_worker(kwargs=dict(_llm_kwargs)):
                    try:
                        _future.set_result(call_llm(effective_model, messages, **kwargs))
                    except BaseException as e:
                        _future.set_exception(e)

                threading.Thread(target=_llm_worker, daemon=True, name="llm-call").start()
                try:
                    resp = _future.result()
                finally:
                    _abandon.set()
        finally:
            _emit(ThinkingEndEvent(iteration=iteration))
        messages.append(resp.assistant_message)

        if resp.reasoning:
            _emit(ReasoningEvent(text=resp.reasoning, iteration=iteration))

        # No tool calls → model produced a final text response
        if not resp.tool_calls:
            return resp.text or ""

        # Dispatch each tool call
        for tc in resp.tool_calls:
            logger.info("Tool call: %s(%s)", tc.name, tc.args)

            # Key representing the specific tool call + args
            args_str = json.dumps(tc.args, sort_keys=True)
            tool_key = (tc.name, args_str)

            failed = False
            if tc.name not in allowed:
                failed = True
                result = f'{{"error": "Unknown or forbidden tool: {tc.name}"}}'
                _emit(
                    ToolErrorEvent(
                        name=tc.name,
                        call_id=tc.id,
                        error=f"Unknown or forbidden tool: {tc.name}",
                        iteration=iteration,
                    )
                )
            else:
                _emit(
                    ToolStartEvent(
                        name=tc.name,
                        args=tc.args,
                        call_id=tc.id,
                        iteration=iteration,
                    )
                )
                start_time = time.perf_counter()
                try:
                    result = allowed[tc.name].run(_cancel_token=cancel_token, **tc.args)
                    duration = time.perf_counter() - start_time
                    _emit(
                        ToolCompleteEvent(
                            name=tc.name,
                            args=tc.args,
                            call_id=tc.id,
                            result=result,
                            duration_s=duration,
                            iteration=iteration,
                        )
                    )
                    if _is_tool_failure(result):
                        failed = True
                except Exception as e:
                    duration = time.perf_counter() - start_time
                    _emit(
                        ToolErrorEvent(
                            name=tc.name,
                            call_id=tc.id,
                            error=str(e),
                            iteration=iteration,
                        )
                    )
                    failed = True
                    result = f'{{"error": "{type(e).__name__}: {str(e)}"}}'

            # Eager projection: a write/gate tool's fat JSON never enters the
            # history — the TUI already got the full result via the event above.
            # Errors stay verbatim so the model can react to them.
            if not failed and tc.name in allowed and allowed[tc.name].collapse == "eager":
                result = eager_stub(allowed[tc.name], result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

            # Update convergence guard
            if failed:
                consecutive_failures[tool_key] = consecutive_failures.get(tool_key, 0) + 1
                failures_count = consecutive_failures[tool_key]
                if failures_count >= 3:
                    logger.error("Convergence guard: tool '%s' with args %s failed %d times consecutively. Aborting agent run.", tc.name, tc.args, failures_count)
                    if progress is not None and progress.cursor:
                        try:
                            progress.set_status(
                                progress.cursor,
                                "blocked",
                                error=f"Convergence guard: '{tc.name}' failed 3× consecutively",
                            )
                            progress.save()
                        except Exception:
                            pass
                    raise RuntimeError(
                        f"Tool '{tc.name}' failed 3 consecutive times with the same arguments: {tc.args}"
                    )
                elif failures_count == 2:
                    logger.warning("Convergence guard: tool '%s' with args %s failed consecutively. Injecting warning message.", tc.name, tc.args)
                    messages.append(
                        {
                            "role": "system",
                            "content": f"IMPORTANT: Tool '{tc.name}' failed consecutively with these parameters. DO NOT call this tool again with the exact same arguments."
                        }
                    )
            else:
                consecutive_failures[tool_key] = 0

        # Loop continues: re-call LLM with tool results

    logger.warning("Agent loop hit max iterations (%d)", max_iterations)
    return "(silica: maximum iterations reached)"
