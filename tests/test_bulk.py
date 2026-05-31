"""Tests for the refactored write path: execute_one, execute_operations,
and the silica_patch_note interactive tool.

Each test runs against an isolated fs-backed temp vault so it never mutates the
session-scoped synthetic vault, and restores the backend on teardown.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import silica.kernel.checkpoints as checkpoints
from silica.driver import DRIVER
from silica.kernel.bulk import execute_one, execute_operations
from silica.kernel.ops import Op, OpType


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """Isolated fs vault with seed notes; patches DRIVER + checkpoint store."""
    vault_dir = tmp_path / "vault"
    (vault_dir / "Concepts").mkdir(parents=True)
    (vault_dir / "Concepts" / "Existing.md").write_text(
        "---\ntags:\n  - seed\n---\n\n# Existing\n\nBody.\n", encoding="utf-8"
    )
    (vault_dir / "Hubs").mkdir(parents=True)
    (vault_dir / "Hubs" / "AI.md").write_text("# AI\n", encoding="utf-8")

    # Point the global DRIVER at this fs vault (same pattern as test_fsm.fs_vault).
    monkeypatch.setenv("SILICA_BACKEND", "fs")
    monkeypatch.setattr("silica.config.CONFIG.backend", "fs")
    monkeypatch.setattr("silica.config.CONFIG.vault_path", str(vault_dir))
    monkeypatch.setattr("silica.driver._driver", None)

    # Isolated checkpoint store for the duration of the test.
    monkeypatch.setattr("silica.kernel.checkpoints._store", None)
    checkpoints.get_checkpoint_store(tmp_path / "checkpoints.db")

    yield vault_dir

    monkeypatch.setattr("silica.driver._driver", None)
    monkeypatch.setattr("silica.kernel.checkpoints._store", None)


# ---------------------------------------------------------------------------
# execute_one — per op type
# ---------------------------------------------------------------------------

def test_execute_one_write(vault):
    op = Op(
        op=OpType.write,
        heading="New Concept",
        source_basename="src.md",
        path="Concepts/New Concept.md",
        snippet="A distilled idea.",
        hub="AI",
    )
    res = execute_one(op)
    assert res["success"] is True
    assert res["op"] == "write"
    content = DRIVER.read_note("Concepts/New Concept.md").content
    assert "# New Concept" in content
    assert "A distilled idea." in content
    assert "[[AI]]" in content


def test_execute_one_patch_appends_section(vault):
    op = Op(
        op=OpType.patch,
        heading="Extra",
        source_basename="src.md",
        path="Concepts/Existing.md",
        snippet="Appended insight.",
    )
    res = execute_one(op)
    assert res["success"] is True
    content = DRIVER.read_note("Concepts/Existing.md").content
    assert "Body." in content              # original preserved
    assert "Appended insight." in content  # new section added


def test_execute_one_overwrite(vault):
    op = Op(
        op=OpType.overwrite,
        heading="Existing",
        source_basename="src.md",
        path="Concepts/Existing.md",
        content="# Existing\n\nFully rewritten.\n",
    )
    res = execute_one(op)
    assert res["success"] is True
    content = DRIVER.read_note("Concepts/Existing.md").content
    assert content.strip() == "# Existing\n\nFully rewritten."
    assert "Body." not in content


def test_execute_one_delete(vault):
    op = Op(
        op=OpType.delete,
        heading="Existing",
        source_basename="src.md",
        path="Concepts/Existing.md",
    )
    res = execute_one(op)
    assert res["success"] is True
    with pytest.raises(Exception):
        DRIVER.read_note("Concepts/Existing.md")


def test_execute_one_skip_is_noop_success(vault):
    op = Op(op=OpType.skip, heading="x", source_basename="src.md")
    res = execute_one(op)
    assert res == {"op": "skip", "success": True}


# ---------------------------------------------------------------------------
# execute_one — missing required params raise
# ---------------------------------------------------------------------------

def test_execute_one_write_missing_hub_raises(vault):
    op = Op(
        op=OpType.write,
        heading="No Hub",
        source_basename="src.md",
        path="Concepts/No Hub.md",
        snippet="x",
    )
    with pytest.raises(ValueError, match="hub"):
        execute_one(op)


def test_execute_one_overwrite_missing_content_raises(vault):
    op = Op(
        op=OpType.overwrite,
        heading="Existing",
        source_basename="src.md",
        path="Concepts/Existing.md",
    )
    with pytest.raises(ValueError, match="content"):
        execute_one(op)


def test_execute_one_patch_missing_note_raises_cannot_patch(vault):
    op = Op(
        op=OpType.patch,
        heading="Ghost",
        source_basename="src.md",
        path="Concepts/DoesNotExist.md",
        snippet="x",
    )
    with pytest.raises(ValueError, match="Cannot patch"):
        execute_one(op)


# ---------------------------------------------------------------------------
# execute_operations — batch aggregation
# ---------------------------------------------------------------------------

def test_execute_operations_mixed_success_and_failure(vault):
    ops = [
        Op(op=OpType.write, heading="A", source_basename="s.md",
           path="Concepts/A.md", snippet="a", hub="AI"),
        # invalid: write with no hub -> failure
        Op(op=OpType.write, heading="B", source_basename="s.md",
           path="Concepts/B.md", snippet="b"),
        Op(op=OpType.skip, heading="C", source_basename="s.md"),
    ]
    res = execute_operations(ops)
    assert res.total == 3
    assert res.successful == 2          # A + skip
    assert res.ok is False
    assert len(res.failed) == 1
    assert res.failed[0].index == 1
    assert res.failed[0].path == "Concepts/B.md"
    # order preserved
    assert [r["index"] for r in res.results] == [0, 1, 2]


def test_execute_operations_all_success_ok_true(vault):
    ops = [
        Op(op=OpType.write, heading="A", source_basename="s.md",
           path="Concepts/A.md", snippet="a", hub="AI"),
        Op(op=OpType.skip, heading="C", source_basename="s.md"),
    ]
    res = execute_operations(ops)
    assert res.ok is True
    assert res.successful == 2
    assert res.failed == []


# ---------------------------------------------------------------------------
# silica_patch_note — interactive tool
# ---------------------------------------------------------------------------

def test_silica_patch_note_happy_path(vault):
    from silica.tools.composed import silica_patch_note

    out = silica_patch_note(
        name="Concepts/Existing.md",
        heading="Insight",
        snippet="Interactive patch body.",
        source_basename="chat.md",
    )
    assert out["success"] is True
    assert out["path"] == "Concepts/Existing.md"
    assert out["checkpoint_depth"] == 2   # floor (original) + this patch
    content = DRIVER.read_note("Concepts/Existing.md").content
    assert "Interactive patch body." in content
    assert "Body." in content


def test_silica_patch_note_note_not_found(vault):
    from silica.tools.composed import silica_patch_note

    out = silica_patch_note(
        name="Concepts/Nope.md",
        heading="x",
        snippet="y",
        source_basename="chat.md",
    )
    assert "error" in out
    assert "Nope" in out["error"]


def test_silica_patch_note_undo_restores_original(vault):
    from silica.tools.composed import silica_patch_note

    original = DRIVER.read_note("Concepts/Existing.md").content

    silica_patch_note(
        name="Concepts/Existing.md", heading="One",
        snippet="first patch", source_basename="chat.md",
    )
    silica_patch_note(
        name="Concepts/Existing.md", heading="Two",
        snippet="second patch", source_basename="chat.md",
    )

    store = checkpoints.get_checkpoint_store()
    # undo the second patch -> back to state after first patch
    after_first = store.undo("Concepts/Existing.md")
    assert "first patch" in after_first
    assert "second patch" not in after_first
    # undo the first patch -> back to original
    restored = store.undo("Concepts/Existing.md")
    assert restored == original
