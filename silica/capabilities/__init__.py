"""Capability registry.

A *capability* is a self-contained background behaviour: a plain
``run(item, config) -> dict`` function that claims a WorkItem of one ``kind`` and
executes it under that behaviour's leash. Today one ``kind`` is owned by exactly
one capability, so dispatch is a keyed lookup — the same shape as Claude Code's
``TOOLS`` table — not a scan. Adding a behaviour is: drop one module here and add
one line to ``CAPABILITIES``.
"""
from __future__ import annotations

from typing import Any, Callable

from silica.planner.workqueue import WorkItem
from silica.capabilities.dedup import run_dedup
from silica.capabilities.refine import run_refine
from silica.capabilities.enrich import run_enrich
from silica.capabilities.orphan import run_orphan

# A capability runs one WorkItem under its leash and returns a status dict.
Capability = Callable[[WorkItem, Any], dict]

CAPABILITIES: dict[str, Capability] = {
    "dedup": run_dedup,
    "refine": run_refine,
    "enrich": run_enrich,
    "orphan": run_orphan,
}
