"""Tests for the Async Review Queue surface (Tier 1 Item 3 — ADR-0007).

Covers:
- DeferredStore.queue_depth()
- queue_depth emitted in ledger digest
- /review command registered in COMMANDS
"""
from __future__ import annotations

import pytest

from silica.kernel.deferred import DeferredStore


# ---------------------------------------------------------------------------
# DeferredStore.queue_depth
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return DeferredStore(path=tmp_path / "deferred")


def test_queue_depth_empty(store):
    assert store.queue_depth() == 0


def test_queue_depth_one_bundle(store):
    store.put("h1", "inbox/a.md", "Dir", None, [{"op": "write"}])
    assert store.queue_depth() == 1


def test_queue_depth_multiple_bundles(store):
    store.put("h1", "inbox/a.md", "Dir", None, [{"op": "write"}])
    store.put("h2", "inbox/b.md", "Dir", None, [{"op": "patch"}, {"op": "write"}])
    assert store.queue_depth() == 2


def test_queue_depth_decreases_after_remove(store):
    store.put("h1", "inbox/a.md", "Dir", None, [])
    store.put("h2", "inbox/b.md", "Dir", None, [])
    store.remove("h1")
    assert store.queue_depth() == 1


# ---------------------------------------------------------------------------
# queue_depth in ledger digest
# ---------------------------------------------------------------------------

def test_digest_includes_queue_depth_when_nonzero(tmp_path, monkeypatch):
    import silica.kernel.progress as prog_mod
    import silica.kernel.deferred as deferred_mod

    prog_mod._RUNS_DIR = tmp_path
    store = DeferredStore(path=tmp_path / "deferred")
    store.put("h1", "inbox/a.md", "Dir", None, [{"op": "write"}])
    monkeypatch.setattr(deferred_mod, "_store", store)

    from silica.kernel.progress import ProgressLedger
    p = ProgressLedger.new(mode="inject", inputs={})
    digest = p.digest()
    assert "review" in digest.lower() or "deferred" in digest.lower() or "queue" in digest.lower()


def test_digest_omits_queue_line_when_empty(tmp_path, monkeypatch):
    import silica.kernel.progress as prog_mod
    import silica.kernel.deferred as deferred_mod

    prog_mod._RUNS_DIR = tmp_path
    store = DeferredStore(path=tmp_path / "deferred")
    monkeypatch.setattr(deferred_mod, "_store", store)

    from silica.kernel.progress import ProgressLedger
    p = ProgressLedger.new(mode="inject", inputs={})
    digest = p.digest()
    assert "REVIEW QUEUE" not in digest


# ---------------------------------------------------------------------------
# /review command registered in COMMANDS list
# ---------------------------------------------------------------------------

def test_review_command_in_commands_list():
    from silica.ui.commands import COMMANDS
    names = {c.name for c in COMMANDS}
    assert "/review" in names


def test_review_command_is_direct_group():
    from silica.ui.commands import COMMANDS
    cmd = next(c for c in COMMANDS if c.name == "/review")
    assert cmd.group == "direct"
