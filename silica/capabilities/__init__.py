"""Capability registry — THE dispatch seam for background work.

A *capability* is a self-contained background behaviour: a plain
``run(item, config) -> dict`` function that claims a WorkItem of one ``kind`` and
executes it under that behaviour's bounds. One ``kind`` is owned by exactly one
capability, so dispatch is a keyed lookup — the same shape as the ``TOOLS``
table — not a scan. Adding a behaviour is: drop one module here and add one
line to ``CAPABILITIES``.

Everything that runs in the background flows through this registry:

  * in-run WorkItems produced by the Injector/Coordinator (dedup, orphan),
  * ad-hoc batches from /dedup, /refine, /enrich (via ``run_subagent_batch``),
  * worker-profile tasks (reader, router) — each WorkerProfile is registered
    here under its own name; the registry is the ONLY dispatch table.

This package is also the home of the worker engine itself (``profile``,
``profiles_builtin``, ``runtime``, ``prompts/``): the profiles and the seam
that dispatches them are two halves of one concept — procedural memory.
The execution engine is always ``BoundedSubAgent`` + the shared consumer loop
in ``silica/agent/subagent.py``; FSM pipelines (injector/refiner/organizer)
are deterministic foreground flows and intentionally stay outside this seam.
"""
from __future__ import annotations

from typing import Any, Callable

from silica.kernel.workqueue import WorkItem
from silica.capabilities.dedup import run_dedup
from silica.capabilities.expand import run_expand
from silica.capabilities.refine import run_refine
from silica.capabilities.enrich import run_enrich
from silica.capabilities.orphan import run_orphan
from silica.capabilities.profile import WorkerProfile
from silica.capabilities.profiles_builtin import READER, ROUTER
from silica.capabilities.runtime import run_worker

# A capability runs one WorkItem under its leash and returns a status dict.
Capability = Callable[[WorkItem, Any], dict]


def _run_worker_item(profile: WorkerProfile, item: WorkItem, config: Any) -> dict[str, Any]:
    """Run a WorkerProfile as a capability.

    WorkItem contract:
        context     — {"goal": str, "inputs": dict}
        target_path — unused (worker profiles are read-only in Phase A)
    """
    result = run_worker(
        profile,
        goal=str(item.context.get("goal", "")),
        inputs=item.context.get("inputs", {}) or {},
        config=config,
        cancel_token=item.cancel_token,
    )
    return {"status": result.status, "output": result.output, "detail": result.detail}


def run_reader(item: WorkItem, config: Any) -> dict[str, Any]:
    return _run_worker_item(READER, item, config)


def run_router(item: WorkItem, config: Any) -> dict[str, Any]:
    return _run_worker_item(ROUTER, item, config)


CAPABILITIES: dict[str, Capability] = {
    "dedup": run_dedup,
    "expand": run_expand,
    "refine": run_refine,
    "enrich": run_enrich,
    "orphan": run_orphan,
    "reader": run_reader,
    "router": run_router,
}
