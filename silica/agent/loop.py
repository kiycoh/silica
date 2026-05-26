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
from typing import Callable
import time
import logging

from silica.agent.events import (
    ToolProgressEvent,
    ToolStartEvent,
    ToolCompleteEvent,
    ToolErrorEvent,
    ReasoningEvent,
    RenderEvent,
)
from silica.agent.llm import call_llm
from silica.config import CONFIG
from silica.tools import TOOLS

logger = logging.getLogger(__name__)


ToolProgressCallback = Callable[[RenderEvent], None] | None


def run_agent(
    messages: list[dict],
    model: str,
    tool_progress_callback: ToolProgressCallback = None,
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
    max_iterations = 50  # Hard safety cap

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

        resp = call_llm(model, messages, tools=schemas)
        messages.append(resp.assistant_message)

        if resp.reasoning:
            _emit(ReasoningEvent(text=resp.reasoning, iteration=iteration))

        # No tool calls → model produced a final text response
        if not resp.tool_calls:
            return resp.text or ""

        # Dispatch each tool call
        for tc in resp.tool_calls:
            logger.info("Tool call: %s(%s)", tc.name, tc.args)

            if tc.name not in TOOLS:
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
                    raise

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

        # Loop continues: re-call LLM with tool results

    logger.warning("Agent loop hit max iterations (%d)", max_iterations)
    return "(silica: raggiunto il limite massimo di iterazioni)"
