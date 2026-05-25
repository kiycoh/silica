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

import logging

import time

from silica.agent.llm import call_llm
from silica.config import CONFIG
from silica.tools import TOOLS

logger = logging.getLogger(__name__)


def run_agent(messages: list[dict], model: str) -> str:
    """Execute the agentic loop until the model produces a text response.

    The loop calls the LLM, dispatches any tool calls, appends results,
    and re-calls until the model responds with text (no tool calls).

    Args:
        messages: mutable conversation history (modified in-place)
        model: litellm model string

    Returns:
        The model's final text response
    """
    # Collect tool schemas for the LLM
    schemas = [t.json_schema() for t in TOOLS.values()] if TOOLS else None

    iteration = 0
    max_iterations = 50  # Hard safety cap

    while iteration < max_iterations:
        iteration += 1
        logger.debug("Agent loop iteration %d", iteration)

        resp = call_llm(model, messages, tools=schemas)
        messages.append(resp.assistant_message)

        # No tool calls → model produced a final text response
        if not resp.tool_calls:
            return resp.text or ""

        # Dispatch each tool call
        for tc in resp.tool_calls:
            logger.info("Tool call: %s(%s)", tc.name, tc.args)

            start_time = time.perf_counter()
            if tc.name not in TOOLS:
                result = f'{{"error": "Unknown tool: {tc.name}"}}'
            else:
                if CONFIG.verbose:
                    logger.info("[DEBUG Tool Input]: Running tool '%s' with args: %s", tc.name, tc.args)
                result = TOOLS[tc.name].run(**tc.args)

            duration = time.perf_counter() - start_time
            if CONFIG.verbose:
                res_str = str(result)
                truncated_result = res_str
                if len(res_str) > 1000:
                    truncated_result = res_str[:1000] + "... (truncated to 1000 chars)"
                logger.info(
                    "[DEBUG Tool Output]: Tool '%s' executed in %.4fs | Result: %s",
                    tc.name,
                    duration,
                    truncated_result,
                )

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
