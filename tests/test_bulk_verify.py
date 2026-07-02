"""Tests for the post-write verify gate (falsifiable gate) in kernel/bulk.py.

Spec: docs/spec-formalizzazione-orchestrazione.md §1 / .superpowers/sdd/task-1-brief.md.
Today `success: True` in kernel/bulk.py means "the driver did not raise", not
"the disk holds what I intended to write" (the LaTeX saga: a write channel
silently doubled backslashes for weeks with the suite green on the fs backend).
This gate re-reads from the DRIVER after every successful write/overwrite/
patch/delete dispatch and falsifies the op result on any mismatch, so it
flows into the EXISTING failure paths (FSM deferred, commit_ops rollback) —
no new downstream machinery.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from silica.driver.base import NoteContent, NoteRef
from silica.kernel.bulk import execute_operations
from silica.kernel.ops import Op, OpType


@pytest.fixture
def fs_vault(tmp_path, monkeypatch):
    """Isolated fs-backed vault (same pattern as test_bulk.py's `vault` fixture)."""
    vault_dir = tmp_path / "vault"
    (vault_dir / "Concepts").mkdir(parents=True)
    (vault_dir / "Concepts" / "Existing.md").write_text(
        "---\ntags:\n  - seed\n---\n\n# Existing\n\nBody.\n", encoding="utf-8"
    )
    (vault_dir / "Concepts" / "ToDelete.md").write_text(
        "# ToDelete\n\nGone soon.\n", encoding="utf-8"
    )
    (vault_dir / "Hubs").mkdir(parents=True)
    (vault_dir / "Hubs" / "AI.md").write_text("# AI\n", encoding="utf-8")

    monkeypatch.setenv("SILICA_BACKEND", "fs")
    monkeypatch.setattr("silica.config.CONFIG.backend", "fs")
    monkeypatch.setattr("silica.config.CONFIG.vault_path", str(vault_dir))
    monkeypatch.setattr("silica.driver._driver", None)

    yield vault_dir

    monkeypatch.setattr("silica.driver._driver", None)


# ---------------------------------------------------------------------------
# Fake drivers — genuinely corrupt the payload in flight (not mock assertions)
# ---------------------------------------------------------------------------

class _CorruptingDriver:
    """Reproduces the historical LaTeX bug: doubles every backslash on write.

    A write/overwrite whose content contains a backslash will land on "disk"
    altered — the read-back must catch that, mirroring the cli-backend bug
    that corrupted notes for weeks while the fs-backend suite stayed green.
    """

    def __init__(self, seed: dict[str, str] | None = None):
        self._notes: dict[str, str] = dict(seed or {})

    @staticmethod
    def _corrupt(content: str) -> str:
        return content.replace("\\", "\\\\")

    def create(self, path: str, content: str) -> NoteRef:
        self._notes[path] = self._corrupt(content)
        return NoteRef(name=path, path=path)

    def overwrite(self, path: str, content: str) -> NoteRef:
        self._notes[path] = self._corrupt(content)
        return NoteRef(name=path, path=path)

    def delete(self, ref) -> None:
        path = ref if isinstance(ref, str) else ref.path
        self._notes.pop(path, None)

    def read_note(self, ref) -> NoteContent:
        path = ref if isinstance(ref, str) else ref.path
        if path not in self._notes:
            raise RuntimeError(f"not found: {path}")
        return NoteContent(ref=NoteRef(name=path, path=path), content=self._notes[path])


class _SnippetDroppingDriver:
    """Overwrite lands, but the merged/appended snippet never makes it to disk
    (e.g. a backend that silently truncates on merge) — the patch write
    "succeeds" yet the substring the gate must find is absent."""

    def __init__(self, seed: dict[str, str] | None = None):
        self._notes: dict[str, str] = dict(seed or {})

    def create(self, path: str, content: str) -> NoteRef:
        self._notes[path] = content
        return NoteRef(name=path, path=path)

    def overwrite(self, path: str, content: str) -> NoteRef:
        # Drops whatever was actually merged in and writes an unrelated stub —
        # a genuine corruption of the payload, not a passthrough.
        self._notes[path] = "# Stub\n\nThe merge silently lost the snippet.\n"
        return NoteRef(name=path, path=path)

    def read_note(self, ref) -> NoteContent:
        path = ref if isinstance(ref, str) else ref.path
        if path not in self._notes:
            raise RuntimeError(f"not found: {path}")
        return NoteContent(ref=NoteRef(name=path, path=path), content=self._notes[path])


class _UndeadDriver:
    """Delete 'succeeds' but the note is still readable afterwards (e.g. a
    backend whose delete only touches an index, not the underlying store)."""

    def __init__(self, seed: dict[str, str] | None = None):
        self._notes: dict[str, str] = dict(seed or {})
        self.delete_called = False

    def delete(self, ref) -> None:
        self.delete_called = True
        # Intentionally does NOT remove the note — reproduces a driver whose
        # delete call returns without actually taking effect.

    def read_note(self, ref) -> NoteContent:
        path = ref if isinstance(ref, str) else ref.path
        if path not in self._notes:
            raise RuntimeError(f"not found: {path}")
        return NoteContent(ref=NoteRef(name=path, path=path), content=self._notes[path])


# ---------------------------------------------------------------------------
# 1. Red: corrupting driver on write -> op fails with post-write verify error
# ---------------------------------------------------------------------------

def test_write_corrupted_by_driver_fails_verify(monkeypatch):
    import silica.kernel.bulk as bulk_mod

    monkeypatch.setattr(bulk_mod, "DRIVER", _CorruptingDriver())

    op = Op(
        op=OpType.write,
        heading="Vectors",
        source_basename="src.md",
        path="Concepts/Vectors.md",
        snippet=r"The norm is $\|\boldsymbol{v}\|_2$.",
        hub="AI",
    )
    res = execute_operations([op])
    assert res.ok is False
    assert res.successful == 0
    assert len(res.failed) == 1
    assert "post-write verify" in res.failed[0].error


# ---------------------------------------------------------------------------
# 2. Clean write on fs backend -> green, no false positive
# ---------------------------------------------------------------------------

def test_write_clean_on_fs_backend_passes_verify(fs_vault):
    op = Op(
        op=OpType.write,
        heading="Clean Concept",
        source_basename="src.md",
        path="Concepts/Clean Concept.md",
        snippet="A distilled idea.",
        hub="AI",
    )
    res = execute_operations([op])
    assert res.ok is True
    assert res.successful == 1
    assert res.failed == []


# ---------------------------------------------------------------------------
# 3. Patch whose snippet never lands -> failed
# ---------------------------------------------------------------------------

def test_patch_snippet_missing_from_readback_fails_verify(monkeypatch):
    import silica.kernel.bulk as bulk_mod

    driver = _SnippetDroppingDriver(
        seed={"Concepts/Existing.md": "# Existing\n\nOriginal body.\n"}
    )
    monkeypatch.setattr(bulk_mod, "DRIVER", driver)

    op = Op(
        op=OpType.patch,
        heading="Extra",
        source_basename="src.md",
        path="Concepts/Existing.md",
        snippet="Appended insight that must survive the merge.",
    )
    res = execute_operations([op])
    assert res.ok is False
    assert res.successful == 0
    assert len(res.failed) == 1
    assert "post-write verify" in res.failed[0].error


# ---------------------------------------------------------------------------
# 4. Delete with the note still present afterwards -> failed
# ---------------------------------------------------------------------------

def test_delete_note_still_present_fails_verify(monkeypatch):
    import silica.kernel.bulk as bulk_mod

    driver = _UndeadDriver(seed={"Concepts/Ghost.md": "# Ghost\n\nStill here.\n"})
    monkeypatch.setattr(bulk_mod, "DRIVER", driver)

    op = Op(
        op=OpType.delete,
        heading="Ghost",
        source_basename="src.md",
        path="Concepts/Ghost.md",
    )
    res = execute_operations([op])
    assert res.ok is False
    assert res.successful == 0
    assert len(res.failed) == 1
    assert "post-write verify" in res.failed[0].error
    assert driver.delete_called is True


# ---------------------------------------------------------------------------
# 5. commit_ops with a corrupting driver -> rolled_back (existing machinery,
#    not a new path)
# ---------------------------------------------------------------------------

def test_commit_ops_rolls_back_on_verify_mismatch(monkeypatch):
    import silica.kernel.bulk as bulk_mod
    from silica.agent.commit import commit_ops

    monkeypatch.setattr(bulk_mod, "DRIVER", _CorruptingDriver())

    op = Op(
        op=OpType.write,
        heading="Vectors",
        source_basename="src.md",
        path="Concepts/Vectors.md",
        snippet=r"The norm is $\|\boldsymbol{v}\|_2$.",
        hub="AI",
    )

    with patch(
        "silica.tools.composed.silica_validate_ops",
        return_value={"validated_count": 1, "success": True},
    ), patch(
        "silica.tools.wrapped.silica_snapshot",
        return_value={"txn_id": "t1", "inverses": []},
    ), patch(
        "silica.tools.composed.silica_lint", return_value={"success": True}
    ), patch(
        "silica.tools.wrapped.silica_restore", return_value={"success": True}
    ) as restore:
        res = commit_ops([op], target_dir="Concepts")

    # silica_lint is stubbed to always pass, so a rollback here can only be
    # triggered by the bulk-write result — i.e. by the verify gate, not by an
    # unrelated lint failure reading from a different (real) backend.
    assert res["status"] == "rolled_back"
    assert res.get("error") == "all write ops failed"
    restore.assert_called_once()


# ---------------------------------------------------------------------------
# 6. End-to-end regression: minimal ingest on fs backend -> all ops committed,
#    the gate does not alter the happy path
# ---------------------------------------------------------------------------

def test_minimal_ingest_all_op_types_committed_on_fs_backend(fs_vault):
    ops = [
        Op(op=OpType.write, heading="New Concept", source_basename="s.md",
           path="Concepts/New Concept.md", snippet="A distilled idea.", hub="AI"),
        Op(op=OpType.patch, heading="Extra", source_basename="s.md",
           path="Concepts/Existing.md", snippet="Appended insight."),
        Op(op=OpType.overwrite, heading="Existing", source_basename="s.md",
           path="Concepts/Existing.md", content="# Existing\n\nFully rewritten.\n"),
        Op(op=OpType.delete, heading="ToDelete", source_basename="s.md",
           path="Concepts/ToDelete.md"),
        Op(op=OpType.skip, heading="Noop", source_basename="s.md"),
    ]
    res = execute_operations(ops)

    assert res.ok is True
    assert res.total == 5
    assert res.successful == 5
    assert res.failed == []
