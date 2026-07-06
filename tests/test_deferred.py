"""Tests for the DeferredStore and the VALIDATE partial-write gate."""
import pytest
import tempfile
from pathlib import Path

from silica.kernel.deferred import DeferredStore


# ---------------------------------------------------------------------------
# DeferredStore unit tests
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return DeferredStore(path=tmp_path / "deferred")


def test_deferred_put_and_get(store):
    store.put(
        content_hash="abc123",
        source_path="inbox/lezione_15.md",
        target_dir="Agenti Autonomi",
        hub="Agenti Autonomi",
        rejected_ops=[{"op": "write", "path": "Agenti Autonomi/MCU.md"}],
        rejection_reasons={"Agenti Autonomi/MCU.md": "too generic"},
    )
    bundle = store.get("abc123")
    assert bundle is not None
    assert bundle["source_path"] == "inbox/lezione_15.md"
    assert bundle["target_dir"] == "Agenti Autonomi"
    assert len(bundle["rejected_ops"]) == 1
    assert bundle["rejection_reasons"]["Agenti Autonomi/MCU.md"] == "too generic"


def test_deferred_get_missing(store):
    assert store.get("nonexistent") is None


def test_deferred_put_overwrites(store):
    store.put("abc123", "inbox/a.md", "Dir", None, [{"op": "write", "path": "Dir/A.md"}])
    store.put("abc123", "inbox/a.md", "Dir", None, [{"op": "write", "path": "Dir/B.md"}])
    bundle = store.get("abc123")
    assert bundle["rejected_ops"][0]["path"] == "Dir/B.md"


def test_deferred_put_dedups_same_op(store):
    """Merging existing + new re-rejections of the same (path, heading) must
    collapse to one, keeping the latest content — not grow every run."""
    store.put("h", "inbox/a.md", "Dir", None,
              [{"op": "write", "path": "Dir/A.md", "heading": "A", "reason": "v1"}])
    existing = store.get("h")["rejected_ops"]
    store.put("h", "inbox/a.md", "Dir", None,
              existing + [{"op": "write", "path": "Dir/A.md", "heading": "A", "reason": "v2"}])
    ops = store.get("h")["rejected_ops"]
    assert len(ops) == 1
    assert ops[0]["reason"] == "v2"


def test_deferred_list_all(store):
    store.put("hash1", "inbox/a.md", "Dir", None, [{"op": "write", "path": "Dir/A.md"}])
    store.put("hash2", "inbox/b.md", "Dir2", "Hub2", [{"op": "write"}, {"op": "patch"}])
    items = store.list_all()
    assert len(items) == 2
    hashes = {i["content_hash"] for i in items}
    assert "hash1" in hashes
    assert "hash2" in hashes
    by_hash = {i["content_hash"]: i for i in items}
    assert by_hash["hash2"]["rejected_count"] == 2


def test_deferred_remove(store):
    store.put("abc123", "inbox/a.md", "Dir", None, [])
    assert store.remove("abc123") is True
    assert store.get("abc123") is None


def test_deferred_remove_missing(store):
    assert store.remove("nonexistent") is False


def test_deferred_list_empty(store):
    assert store.list_all() == []


# ---------------------------------------------------------------------------
# VALIDATE gate: partial-write behaviour
# ---------------------------------------------------------------------------

def _make_ops_file(tmp_path, ops: list) -> str:
    import orjson
    p = tmp_path / "ops.json"
    p.write_bytes(orjson.dumps(ops))
    return str(p)


def test_validate_returns_validated_and_rejected_lists(tmp_path):
    """validate_operations always returns (validated, rejected) lists — never raises."""
    from silica.kernel.ops import Op, OpType
    from silica.kernel.validate import validate_operations

    op_a = Op(op=OpType.write, path="Dir/GPU.md", heading="GPU", source_basename="lezione.md")
    op_b = Op(op=OpType.write, path="Dir/MCU.md", heading="MCU", source_basename="lezione.md")

    # No payloads → heading check is skipped; path check will fail because
    # Dir/GPU.md and Dir/MCU.md don't exist in the real vault. Both will
    # be validated (write to non-existent path is valid) or rejected by path
    # rules depending on target_dir. Either way it must return lists.
    validated, rejected = validate_operations([op_a, op_b], [], target_dir=str(tmp_path))
    assert isinstance(validated, list)
    assert isinstance(rejected, list)
    assert len(validated) + len(rejected) == 2


def test_deferred_store_populated_on_partial_rejection(tmp_path):
    """When some ops are rejected, deferred store must receive the rejected ops."""
    from silica.kernel.deferred import DeferredStore

    store = DeferredStore(path=tmp_path / "deferred")

    rejected_ops_raw = [
        {"op": {"op": "write", "path": "Dir/MCU.md", "heading": "MCU"}, "reason": "too generic"}
    ]

    deferred_ops = [
        r.get("op", r) if isinstance(r, dict) and "op" in r else r
        for r in rejected_ops_raw
    ]
    rejection_reasons = {
        (r.get("op", {}).get("path") or r.get("op", {}).get("heading") or "?"): r.get("reason", "")
        for r in rejected_ops_raw if isinstance(r, dict)
    }

    store.put(
        content_hash="testhash",
        source_path="inbox/lezione_15.md",
        target_dir="Dir",
        hub="Dir",
        rejected_ops=deferred_ops,
        rejection_reasons=rejection_reasons,
    )

    bundle = store.get("testhash")
    assert bundle is not None
    assert bundle["rejected_ops"][0]["path"] == "Dir/MCU.md"
    assert bundle["rejection_reasons"]["Dir/MCU.md"] == "too generic"


def test_defer_ops_accumulates_across_phases(tmp_path, monkeypatch):
    """_defer_ops must MERGE into the bundle, not overwrite it: COLLISION,
    VALIDATE and WRITE all key on the same source content_hash, so a later
    phase (or chunk) deferring ops must not clobber an earlier phase's ops."""
    import silica.kernel.deferred as deferred_mod
    from silica.router.orchestrator import InjectorFSM

    # conftest's _isolate_deferred_store already points the default store at tmp.
    fsm = InjectorFSM("Inbox/lez.md", "TargetDir", hub="Hub")
    fsm.context["source_content_hash"] = "shared-hash"

    # COLLISION defers one op...
    assert fsm._defer_ops(
        [{"op": "skip", "heading": "A", "source_basename": "lez.md", "path": None}],
        {"A": "borderline"},
        phase="COLLISION",
    )
    # ...then VALIDATE defers another for the SAME content hash.
    assert fsm._defer_ops(
        [{"op": "write", "path": "TargetDir/B.md", "heading": "B", "source_basename": "lez.md"}],
        {"TargetDir/B.md": "too generic"},
        phase="VALIDATE",
    )

    bundle = deferred_mod.get_deferred_store().get("shared-hash")
    headings = {o.get("heading") for o in bundle["rejected_ops"]}
    assert headings == {"A", "B"}, f"VALIDATE clobbered COLLISION's deferred op: {headings}"
    assert bundle["rejection_reasons"] == {"A": "borderline", "TargetDir/B.md": "too generic"}


def test_defer_ops_skips_without_content_hash(tmp_path, monkeypatch):
    """No content_hash → nothing persisted, returns False (no crash)."""
    import silica.kernel.deferred as deferred_mod
    from silica.router.orchestrator import InjectorFSM

    fsm = InjectorFSM("Inbox/lez.md", "TargetDir")
    # No source_content_hash set and no per-file hashes → empty hash.
    assert fsm._defer_ops([{"op": "skip", "heading": "X"}], {}, phase="VALIDATE") is False
    assert deferred_mod.get_deferred_store().list_all() == []
