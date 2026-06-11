"""Architectural invariants for the capability seam.

CAPABILITIES (silica/capabilities/__init__.py) is THE dispatch table for
background work: every WorkItem kind produced anywhere in silica/ must have a
registered capability, and every WorkerProfile must be dispatchable through
the same table (kind == profile name). A new producer or profile that bypasses
the seam fails here and forces an explicit decision.
"""
from __future__ import annotations

import re
from pathlib import Path

SILICA_ROOT = Path(__file__).resolve().parent.parent / "silica"

# kind="..." within the argument list of a WorkItem(...) construction.
_KIND_RE = re.compile(r"WorkItem\([^)]*?kind=\"([a-z_]+)\"", re.DOTALL)


def _produced_kinds() -> set[str]:
    kinds: set[str] = set()
    for path in SILICA_ROOT.rglob("*.py"):
        kinds.update(_KIND_RE.findall(path.read_text(encoding="utf-8")))
    return kinds


def test_every_produced_workitem_kind_has_a_capability():
    from silica.capabilities import CAPABILITIES

    produced = _produced_kinds()
    assert produced, "scan found no WorkItem producers — regex drifted from the code"
    missing = produced - set(CAPABILITIES)
    assert not missing, (
        f"WorkItem kind(s) produced without a registered capability: {missing}. "
        "Register them in silica/capabilities/__init__.py or stop producing them."
    )


def test_every_worker_profile_is_dispatchable_through_the_seam():
    from silica.capabilities import CAPABILITIES
    from silica.capabilities.worker import run_worker_item
    from silica.workers.profile import PROFILES

    assert PROFILES, "no worker profiles registered"
    for name in PROFILES:
        assert CAPABILITIES.get(name) is run_worker_item, (
            f"profile '{name}' is not dispatchable via CAPABILITIES — "
            "the worker adapter must cover every registered profile"
        )


def test_worker_item_round_trip_through_dispatch(monkeypatch):
    """A WorkItem with kind=<profile> flows BoundedSubAgent → adapter → run_worker."""
    from silica.agent.subagent import BoundedSubAgent
    from silica.config import SilicaConfig
    from silica.planner.workqueue import WorkItem
    from silica.workers.profile import WorkerResult

    seen: dict = {}

    def fake_run_worker(task, *, config, cancel_token=None, profiles=None):
        seen["profile"] = task.profile
        seen["goal"] = task.goal
        return WorkerResult(status="ok", output="digest")

    monkeypatch.setattr("silica.capabilities.worker.run_worker", fake_run_worker)
    agent = BoundedSubAgent(SilicaConfig())
    res = agent.handle(WorkItem(kind="reader", target_path="", context={"goal": "g"}))

    assert res == {"status": "ok", "output": "digest", "detail": ""}
    assert seen == {"profile": "reader", "goal": "g"}
