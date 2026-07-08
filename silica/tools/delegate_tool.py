# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""silica_delegate — fan a list of worker tasks out through the capability seam.

Each task becomes a WorkItem (kind = profile name) dispatched by
``run_subagent_batch``: the same registry, dispatch, and consumer loop used for
every other background behaviour. Concurrency is bounded by the pool size (cap
10, the historical fan-out limit) plus the global worker semaphore inside
run_worker. Returns aggregated results in submission order plus a status
summary.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from silica.tools import tool
from silica.kernel.workqueue import WorkItem
from silica.agent.subagent import run_subagent_batch
import silica.capabilities  # noqa: F401  (registers capabilities incl. worker profiles)


_MAX_WORKERS = 10  # historical delegate() fan-out cap


class DelegateArgs(BaseModel):
    profile: str = Field(description="WorkerProfile name, e.g. 'reader' or 'router'")
    tasks: list[dict] = Field(
        description="List of {goal: str, inputs: dict} task specs for the workers"
    )
    max_workers: int = Field(default=7, description="Parallel workers (cap 10)")


@tool(DelegateArgs, cls="composed")
def silica_delegate(profile: str, tasks: list[dict], max_workers: int = 7) -> dict:
    """Fan a list of worker tasks out to parallel workers; return aggregated results.

    Each task is {goal, inputs}. An unknown profile yields status='skipped' per
    task. Returns {"results": [...], "summary": {status: count}}.
    """
    if not tasks:
        return {"results": [], "summary": {}}

    items = [
        WorkItem(
            kind=profile,
            target_path="",
            context={
                "goal": spec.get("goal", ""),
                "inputs": spec.get("inputs", {}) or {},
            },
            reason="silica_delegate",
        )
        for spec in tasks
    ]
    batch = run_subagent_batch(items, max_workers=min(max_workers, _MAX_WORKERS))

    # Keep the historical return shape: per-task dicts without the (empty)
    # target field, in submission order.
    results = [{k: v for k, v in r.items() if k != "target"} for r in batch["results"]]
    return {"results": results, "summary": batch["summary"]}
