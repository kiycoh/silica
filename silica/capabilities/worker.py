"""Worker-profile capability — runs a WorkerProfile task as a WorkItem.

Bridges the two execution shapes so PROFILES stops being a parallel dispatch
table: a WorkItem whose ``kind`` names a WorkerProfile is dispatched to
``run_worker`` like any other capability, making the profile registry an
implementation detail behind the CAPABILITIES seam.

WorkItem contract for this capability:
    kind        — the WorkerProfile name ("reader", "router", ...)
    context     — {"goal": str, "inputs": dict}
    target_path — unused (worker profiles are read-only in Phase A)
"""
from __future__ import annotations

from typing import Any

from silica.planner.workqueue import WorkItem
from silica.workers.profile import WorkerTask
from silica.workers.runtime import run_worker
import silica.workers.profiles_builtin  # noqa: F401  (registers built-in profiles)


def run_worker_item(item: WorkItem, config: Any) -> dict[str, Any]:
    task = WorkerTask(
        profile=item.kind,
        goal=str(item.context.get("goal", "")),
        inputs=item.context.get("inputs", {}) or {},
    )
    result = run_worker(task, config=config, cancel_token=item.cancel_token)
    return {"status": result.status, "output": result.output, "detail": result.detail}
