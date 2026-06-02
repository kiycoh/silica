"""Tests for the commit_ops micro-gate (silica/agent/commit.py)."""
from unittest.mock import patch

from silica.agent.commit import commit_ops
from silica.agent.leash import dedup_leash
from silica.kernel.ops import Op, OpType


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


def test_commit_leash_drops_forbidden_op_before_write():
    # An overwrite is outside the dedup leash → dropped before any tool runs.
    leash = dedup_leash("Concepts/Big.md")
    overwrite = Op(op=OpType.overwrite, heading="C", source_basename="i.md",
                   path="Concepts/Big.md", content="x")
    with patch("silica.tools.composed.silica_validate_ops") as validate:
        res = commit_ops([overwrite], target_dir="Concepts", leash=leash)
    # No actionable ops survived the leash → validate never called.
    validate.assert_not_called()
    assert res["status"] == "no_ops"
    assert len(res["rejected_leash"]) == 1


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

    with patch("silica.planner.workqueue.path_lease", side_effect=fake_path_lease), \
         patch("silica.tools.composed.silica_validate_ops",
               return_value={"validated_count": 1, "status": "ok", "rejected_ops": []}), \
         patch("silica.tools.wrapped.silica_snapshot", return_value={"txn_id": "t1", "inverses": []}), \
         patch("silica.tools.composed.silica_bulk_write", return_value={"successful": 1, "total": 1, "failed": []}), \
         patch("silica.tools.composed.silica_lint", return_value={"success": True}):
        delete_op = Op(op=OpType.delete, heading="Foo", source_basename="inbox.md", path="notes/Foo.md")
        commit_ops([delete_op], target_dir="notes")

    assert any("Foo" in p for p in leased_paths), (
        f"Delete op path 'notes/Foo.md' was not leased; leased paths: {leased_paths}"
    )
