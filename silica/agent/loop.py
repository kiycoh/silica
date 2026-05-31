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
import time
import logging
import json

if TYPE_CHECKING:
    from silica.planner.progress import ProgressLedger

from silica.agent.events import (
    ToolProgressEvent,
    ToolStartEvent,
    ToolCompleteEvent,
    ToolErrorEvent,
    ReasoningEvent,
    RenderEvent,
    ThinkingStartEvent,
    ThinkingEndEvent,
)
from silica.agent.llm import call_llm
from silica.config import CONFIG
from silica.tools import TOOLS

logger = logging.getLogger(__name__)


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
        # Fallback keyword checks
        lower_res = result.lower()
        if "error" in lower_res or "exception" in lower_res or "failed" in lower_res:
            return True
    elif isinstance(result, dict) and "error" in result:
        return True
    return False


ToolProgressCallback = Callable[[RenderEvent], None] | None


def run_agent(
    messages: list[dict],
    model: str,
    tool_progress_callback: ToolProgressCallback = None,
    progress: "ProgressLedger | None" = None,
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
    # Collect tool schemas for the LLM
    schemas = [t.json_schema() for t in TOOLS.values()] if TOOLS else None

    iteration = 0
    max_iterations = 20  # Hard safety cap lowered from 50

    # Track consecutive failures for the same (tool_name, args) pair
    # Key: (tool_name, args_json_string)
    # Value: consecutive failure count
    consecutive_failures: dict[tuple[str, str], int] = {}

    def _emit(event: RenderEvent) -> None:
        """Best-effort event emission — swallows all consumer exceptions."""
        if tool_progress_callback is None:
            return
        try:
            tool_progress_callback(event)
        except Exception as exc:
            logger.debug("tool_progress_callback error (swallowed): %s", exc)

    while iteration < max_iterations:
        iteration += 1
        logger.debug("Agent loop iteration %d", iteration)

        _emit(ThinkingStartEvent(iteration=iteration))
        try:
            # Run the (synchronous, potentially slow) LLM call on a daemon thread
            # so KeyboardInterrupt on the main thread propagates on the first Ctrl+C
            # instead of being trapped inside a C-level network recv().
            with _cf.ThreadPoolExecutor(max_workers=1) as _llm_pool:
                _future = _llm_pool.submit(call_llm, model, messages, tools=schemas)
                try:
                    resp = _future.result()
                except KeyboardInterrupt:
                    _future.cancel()
                    raise
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
            if tc.name not in TOOLS:
                failed = True
                result = f'{{"error": "Unknown tool: {tc.name}"}}'
                _emit(
                    ToolErrorEvent(
                        name=tc.name,
                        call_id=tc.id,
                        error=f"Unknown tool: {tc.name}",
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
                    result = TOOLS[tc.name].run(**tc.args)
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
