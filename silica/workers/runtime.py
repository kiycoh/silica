"""run_worker — run_agent constrained by a WorkerProfile.

A worker is a leaf: it runs a bounded tool-use loop on the worker model, under a
tool subset and iteration cap, gated by the global worker semaphore (inside
run_agent, because constraints is not None). It returns a structured WorkerResult
produced by the profile's result_parser from (final_text, tool_trace).
"""
from __future__ import annotations

import logging
from typing import Any

from silica.agent.loop import run_agent
from silica.agent.constraints import AgentConstraints
from silica.agent.events import ToolCompleteEvent
from silica.workers.profile import WorkerProfile, WorkerTask, WorkerResult, PROFILES

logger = logging.getLogger(__name__)


def _render_goal(task: WorkerTask) -> str:
    """Render the task into the user turn. Inputs are appended as JSON context."""
    import orjson

    parts = [task.goal]
    if task.inputs:
        parts.append("\nInputs:\n" + orjson.dumps(task.inputs).decode())
    return "\n".join(parts)


def run_worker(
    task: WorkerTask,
    *,
    config: Any,
    cancel_token: Any = None,
    profiles: dict[str, WorkerProfile] | None = None,
) -> WorkerResult:
    registry = profiles if profiles is not None else PROFILES
    profile = registry.get(task.profile)
    if profile is None:
        return WorkerResult(status="error", detail=f"no profile '{task.profile}'")

    worker_model = getattr(config, "worker_model", None) or getattr(config, "model", None)
    if not worker_model:
        return WorkerResult(status="error", detail="no worker_model configured")

    trace: list[dict] = []

    def _collect(event: Any) -> None:
        if isinstance(event, ToolCompleteEvent):
            trace.append({"name": event.name, "args": event.args, "result": event.result})

    messages = [
        {"role": "system", "content": profile.system_prompt},
        {"role": "user", "content": _render_goal(task)},
    ]

    try:
        final = run_agent(
            messages,
            model=worker_model,
            tool_progress_callback=_collect,
            cancel_token=cancel_token,
            constraints=AgentConstraints(
                tools=profile.tools,
                model=worker_model,
                max_iterations=profile.max_iterations,
            ),
        )
    except Exception as e:  # a worker error must never crash the pool
        logger.warning("run_worker '%s' failed: %s", task.profile, e)
        return WorkerResult(status="error", detail=str(e))

    try:
        return profile.result_parser(final or "", trace)
    except Exception as e:
        logger.warning("result_parser '%s' failed: %s", task.profile, e)
        return WorkerResult(status="error", detail=f"parser error: {e}")
