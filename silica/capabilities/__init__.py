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
    registered here under its own name via the ``worker`` adapter, so PROFILES
    is an implementation detail, not a second dispatch table.

The execution engine is always ``BoundedSubAgent`` + the shared consumer loop
in ``silica/agent/subagent.py``; FSM pipelines (injector/refiner/organizer)
are deterministic foreground flows and intentionally stay outside this seam.
"""
from __future__ import annotations

from typing import Any, Callable

from silica.planner.workqueue import WorkItem
from silica.capabilities.dedup import run_dedup
from silica.capabilities.refine import run_refine
from silica.capabilities.enrich import run_enrich
from silica.capabilities.orphan import run_orphan
from silica.capabilities.worker import run_worker_item
from silica.workers.profile import PROFILES

# A capability runs one WorkItem under its leash and returns a status dict.
Capability = Callable[[WorkItem, Any], dict]

CAPABILITIES: dict[str, Capability] = {
    "dedup": run_dedup,
    "refine": run_refine,
    "enrich": run_enrich,
    "orphan": run_orphan,
}

# Every worker profile is dispatchable through the same seam: kind == profile
# name. Importing silica.capabilities.worker registered the built-in profiles.
for _profile_name in PROFILES:
    CAPABILITIES[_profile_name] = run_worker_item
