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
  * worker-profile tasks (reader, router, ...) — every WorkerProfile is
    registered here under its own name via ``run_worker_item`` below, so
    PROFILES is an implementation detail, not a second dispatch table.

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
from silica.capabilities.refine import run_refine
from silica.capabilities.enrich import run_enrich
from silica.capabilities.orphan import run_orphan
from silica.capabilities.profile import WorkerTask, PROFILES
from silica.capabilities.runtime import run_worker
import silica.capabilities.profiles_builtin  # noqa: F401  (registers built-in profiles)

# A capability runs one WorkItem under its leash and returns a status dict.
Capability = Callable[[WorkItem, Any], dict]


def run_worker_item(item: WorkItem, config: Any) -> dict[str, Any]:
    """Run a WorkerProfile task as a WorkItem.

    Bridges the two execution shapes so PROFILES stops being a parallel dispatch
    table: a WorkItem whose ``kind`` names a WorkerProfile is dispatched to
    ``run_worker`` like any other capability.

    WorkItem contract for this capability:
        kind        — the WorkerProfile name ("reader", "router", ...)
        context     — {"goal": str, "inputs": dict}
        target_path — unused (worker profiles are read-only in Phase A)
    """
    task = WorkerTask(
        profile=item.kind,
        goal=str(item.context.get("goal", "")),
        inputs=item.context.get("inputs", {}) or {},
    )
    result = run_worker(task, config=config, cancel_token=item.cancel_token)
    return {"status": result.status, "output": result.output, "detail": result.detail}


CAPABILITIES: dict[str, Capability] = {
    "dedup": run_dedup,
    "refine": run_refine,
    "enrich": run_enrich,
    "orphan": run_orphan,
}

# Every worker profile is dispatchable through the same seam: kind == profile
# name. Importing profiles_builtin above registered the built-in profiles.
for _profile_name in PROFILES:
    CAPABILITIES[_profile_name] = run_worker_item
