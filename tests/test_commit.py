"""Tests for the commit_ops micro-gate (silica/agent/commit.py)."""
from unittest.mock import patch

import pytest

from silica.agent.commit import commit_ops
from silica.agent.bounds import dedup_bounds
from silica.kernel.ops import Op, OpType
import silica.kernel.checkpoints as checkpoints


def _patch_op(path="Concepts/Big.md"):
    return Op(op=OpType.patch, heading="Concept", source_basename="i.md",
              path=path, snippet="appended info")


def test_commit_happy_path():
    with patch("silica.tools.composed.silica_validate_ops", return_value={"validated_count": 1, "success": True}), \
         patch("silica.tools.wrapped.silica_snapshot", return_value={"txn_id": "t1", "inverses": []}), \
         patch("silica.tools.composed.silica_bulk_write", return_value={"successful": 1, "total": 1, "failed": []}), \
         patch("silica.tools.composed.silica_lint", return_value={"success": True}), \
         patch("silica.tools.wrapped.silica_restore") as restore:
        res = commit_ops([_patch_op()], target_dir="Concepts")
    assert res["status"] == "committed"
    assert res["committed"] == 1
    restore.assert_not_called()


def test_commit_rolls_back_on_lint_failure():
    with patch("silica.tools.composed.silica_validate_ops", return_value={"validated_count": 1, "success": True}), \
         patch("silica.tools.wrapped.silica_snapshot", return_value={"txn_id": "t9", "inverses": [{"kind": "restore_version", "path": "Concepts/Big.md"}]}), \
         patch("silica.tools.composed.silica_bulk_write", return_value={"successful": 1, "total": 1, "failed": []}), \
         patch("silica.tools.composed.silica_lint", return_value={"success": False, "errors": ["bad frontmatter"]}), \
         patch("silica.tools.wrapped.silica_restore", return_value={"success": True}) as restore:
        res = commit_ops([_patch_op()], target_dir="Concepts")
    assert res["status"] == "rolled_back"
    assert res["lint_failures"]
    restore.assert_called_once()


def test_commit_bounds_drops_forbidden_op_before_write():
    # An overwrite is outside the dedup bounds → dropped before any tool runs.
    bounds = dedup_bounds("Concepts/Big.md")
    overwrite = Op(op=OpType.overwrite, heading="C", source_basename="i.md",
                   path="Concepts/Big.md", content="x")
    with patch("silica.tools.composed.silica_validate_ops") as validate:
        res = commit_ops([overwrite], target_dir="Concepts", bounds=bounds)
    # No actionable ops survived the bounds → validate never called.
    validate.assert_not_called()
    assert res["status"] == "no_ops"
    assert len(res["rejected_by_bounds"]) == 1


def test_commit_no_ops_when_all_skip():
    skip = Op(op=OpType.skip, heading="C", source_basename="i.md", reason="noop")
    res = commit_ops([skip], target_dir="X")
    assert res["status"] == "no_ops"


def test_delete_op_path_is_leased():
    """A delete op must acquire a path lease so concurrent writes are serialized."""
    from contextlib import contextmanager

    leased_paths = []

    @contextmanager
    def fake_path_lease(path):
        leased_paths.append(path)
        yield

    with patch("silica.kernel.workqueue.path_lease", side_effect=fake_path_lease), \
         patch("silica.tools.composed.silica_validate_ops",
               return_value={"validated_count": 1, "status": "ok", "rejected_ops": []}), \
         patch("silica.tools.wrapped.silica_snapshot", return_value={"txn_id": "t1", "inverses": []}), \
         patch("silica.tools.composed.silica_bulk_write", return_value={"successful": 1, "total": 1, "failed": []}), \
         patch("silica.tools.composed.silica_lint", return_value={"success": True}):
        delete_op = Op(op=OpType.delete, heading="Foo", source_basename="inbox.md", path="notes/Foo.md")
        commit_ops([delete_op], target_dir="notes")

    assert "notes/Foo.md" in leased_paths, (
        f"Delete op path 'notes/Foo.md' was not leased; got: {leased_paths}"
    )


# ---------------------------------------------------------------------------
# commit_derived — machine-generated notes with prior metadata preservation
# ---------------------------------------------------------------------------


@pytest.fixture
def derived_vault(tmp_path, monkeypatch):
    """Isolated fs vault for commit_derived tests; patches DRIVER + checkpoint store."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    (vault_dir / "Wiki.md").write_text(
        "---\ntags:\n  - custom\npinned: yes\n---\n\n# Wiki\n\nold\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SILICA_BACKEND", "fs")
    monkeypatch.setattr("silica.config.CONFIG.backend", "fs")
    monkeypatch.setattr("silica.config.CONFIG.vault_path", str(vault_dir))
    monkeypatch.setattr("silica.driver._driver", None)
    monkeypatch.setattr("silica.kernel.checkpoints._store", None)
    checkpoints.get_checkpoint_store(tmp_path / "checkpoints.db")
    yield vault_dir
    monkeypatch.setattr("silica.driver._driver", None)
    monkeypatch.setattr("silica.kernel.checkpoints._store", None)


def test_commit_derived_floors_bare_regen_with_prior_frontmatter(derived_vault):
    """A derived regen that omits the YAML block keeps the note's prior
    metadata and lands lint-green instead of rolling back."""
    from silica.agent.commit import commit_derived
    res = commit_derived("Wiki.md", "# Wiki\n\nregenerated\n")
    assert res["status"] == "committed", res
    landed = (derived_vault / "Wiki.md").read_text(encoding="utf-8")
    head = landed.split("\n---\n")[0]
    assert "tags:\n  - custom" in head and "pinned: yes" in head
    assert "AI: true" in head
    assert "regenerated" in landed and "old" not in landed


def test_commit_derived_first_write_gets_minimal_floor(derived_vault):
    from silica.agent.commit import commit_derived
    res = commit_derived("Fresh.md", "# Fresh\n\nbody\n")
    assert res["status"] == "committed", res
    landed = (derived_vault / "Fresh.md").read_text(encoding="utf-8")
    assert landed.startswith("---\nAI: true\nlast modified: ")
