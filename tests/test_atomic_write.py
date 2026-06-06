# tests/test_atomic_write.py
import pytest
from silica.kernel.ops import Op, OpType
from silica.kernel.atomic_write import commit_note_atomic, NoteCommitResult, bulk_write_atomic, AtomicBulkResult


def _patch_op(path: str, snippet: str = "body") -> Op:
    return Op(op=OpType.patch, heading="H", source_basename="src.md",
              path=path, snippet=snippet, hub="Hub")


def test_clean_patch_commits_and_returns_inverse(tmp_vault, monkeypatch):
    target = tmp_vault.note("People/Ada.md", "---\n---\nseed\n")
    res = commit_note_atomic(_patch_op(target), lint=False)
    assert isinstance(res, NoteCommitResult)
    assert res.ok is True
    assert res.path == target
    assert res.reverted is False
    assert res.post_hash is not None
    assert len(res.inverses) >= 1
    assert "body" in tmp_vault.read(target)


def test_lint_failure_reverts_only_this_note(tmp_vault, monkeypatch):
    target = tmp_vault.note("Areas/Roadmap.md", "---\n---\nseed\n")
    original = tmp_vault.read(target)
    monkeypatch.setattr("silica.tools.composed.silica_lint",
                        lambda *a, **k: {"success": False, "errors": ["bad link"]})
    res = commit_note_atomic(_patch_op(target), lint=True)
    assert res.ok is False
    assert res.reverted is True
    assert "lint failed" in res.error
    assert tmp_vault.read(target) == original


def test_failing_sibling_does_not_roll_back_others(tmp_vault, monkeypatch):
    a = tmp_vault.note("People/Ada.md", "---\n---\nseed\n")
    b = tmp_vault.note("Areas/Roadmap.md", "---\n---\nseed\n")
    c = tmp_vault.note("People/Grace.md", "---\n---\nseed\n")

    def fake_lint(note_name, op_type="", hub=""):
        return {"success": "Roadmap" not in note_name, "errors": ["bad"]}
    monkeypatch.setattr("silica.tools.composed.silica_lint", fake_lint)

    ops = [_patch_op(a), _patch_op(b), _patch_op(c)]
    result = bulk_write_atomic(ops, lint=True)

    assert isinstance(result, AtomicBulkResult)
    assert result.total == 3
    assert {r.path for r in result.committed} == {a, c}
    assert [r.path for r in result.failed] == [b]
    assert "body" in tmp_vault.read(a)
    assert "body" in tmp_vault.read(c)
    assert "body" not in tmp_vault.read(b)
    assert result.ok is False
