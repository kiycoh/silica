"""Tests for the LeashedSubAgent dedup behaviour (silica/agent/subagent.py)."""
from unittest.mock import patch, MagicMock

from silica.agent.subagent import (
    LeashedSubAgent, DedupDecision, RefineResult, OrphanLinkDecision,
)
from silica.kernel.ops import OpType
from silica.planner.workqueue import WorkItem


def _item():
    return WorkItem(
        kind="dedup",
        target_path="Concepts/Gradient Descent.md",
        context={
            "concept": "Discesa del gradiente",
            "excerpt": "Variante mini-batch con momentum.",
            "candidate": "Gradient Descent",
            "inbox_file": "Inbox/ml.md",
            "hub": "Concepts",
        },
        reason="borderline_similarity score=0.78",
    )


def test_dedup_merge_builds_single_patch_under_leash():
    agent = LeashedSubAgent()
    decision = DedupDecision(is_duplicate=True, rationale="same concept", addition="### Momentum\nNew info.")

    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="existing body")), \
         patch.object(agent, "_decide_dedup", return_value=decision), \
         patch("silica.agent.subagent.commit_ops", return_value={"status": "committed", "committed": 1}) as commit:
        res = agent.handle(_item())

    assert res["status"] == "committed"
    # commit_ops called with exactly one patch op + a dedup leash on the candidate.
    ops_arg = commit.call_args.args[0]
    assert len(ops_arg) == 1
    assert ops_arg[0].op == OpType.patch
    assert ops_arg[0].path == "Concepts/Gradient Descent.md"
    leash = commit.call_args.kwargs["leash"]
    assert leash.name == "dedup"
    assert OpType.patch in leash.allowed_ops and OpType.overwrite not in leash.allowed_ops


def test_dedup_no_merge_when_not_duplicate():
    agent = LeashedSubAgent()
    decision = DedupDecision(is_duplicate=False, rationale="different topics")
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="body")), \
         patch.object(agent, "_decide_dedup", return_value=decision), \
         patch("silica.agent.subagent.commit_ops") as commit:
        res = agent.handle(_item())
    assert res["status"] == "no_merge"
    commit.assert_not_called()


def test_dedup_no_merge_when_addition_empty():
    agent = LeashedSubAgent()
    decision = DedupDecision(is_duplicate=True, rationale="dup but nothing new", addition="   ")
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="body")), \
         patch.object(agent, "_decide_dedup", return_value=decision), \
         patch("silica.agent.subagent.commit_ops") as commit:
        res = agent.handle(_item())
    assert res["status"] == "no_merge"
    commit.assert_not_called()


def test_unknown_kind_is_skipped():
    agent = LeashedSubAgent()
    res = agent.handle(WorkItem(kind="mystery", target_path="X.md"))
    assert res["status"] == "skipped"


def test_unreadable_candidate_is_skipped():
    agent = LeashedSubAgent()
    with patch("silica.driver.DRIVER.read_note", side_effect=RuntimeError("missing")):
        res = agent.handle(_item())
    assert res["status"] == "skipped"


# --- refine behaviour ------------------------------------------------------

def _refine_item():
    return WorkItem(kind="refine", target_path="Notes/Target.md", context={"hub": "Concepts"})


def test_refine_builds_overwrite_under_refiner_leash():
    agent = LeashedSubAgent()
    refined = RefineResult(content="# Target\n\n> [!note]\nBody with [[Link]].")
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="old body [[Link]]")), \
         patch.object(agent, "_refine_note", return_value=refined), \
         patch("silica.agent.subagent.commit_ops", return_value={"status": "committed", "committed": 1}) as commit:
        res = agent.handle(_refine_item())
    assert res["status"] == "committed"
    ops_arg = commit.call_args.args[0]
    assert ops_arg[0].op == OpType.overwrite
    leash = commit.call_args.kwargs["leash"]
    assert leash.name == "refiner"
    assert leash.content_guard is not None  # anti-info-loss enforced


def test_refine_skips_empty_note():
    agent = LeashedSubAgent()
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="   ")):
        res = agent.handle(_refine_item())
    assert res["status"] == "skipped"


# --- orphan connector behaviour --------------------------------------------

def _orphan_item():
    return WorkItem(
        kind="orphan",
        target_path="Notes/Lonely.md",
        context={"candidates": [
            {"name": "Gradient Descent", "path": "Concepts/Gradient Descent"},
            {"name": "Backprop", "path": "Concepts/Backprop"},
        ]},
        reason="residual_orphan",
    )


def test_orphan_links_only_to_offered_candidates():
    agent = LeashedSubAgent()
    # Model returns one valid candidate + one hallucinated name.
    decision = OrphanLinkDecision(links=["Gradient Descent", "Made Up Note"], rationale="related")
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="orphan body")), \
         patch.object(agent, "_decide_links", return_value=decision), \
         patch("silica.agent.subagent.commit_ops", return_value={"status": "committed", "committed": 1}) as commit:
        res = agent.handle(_orphan_item())
    assert res["status"] == "committed"
    op = commit.call_args.args[0][0]
    assert op.op == OpType.patch and op.path == "Notes/Lonely.md"
    # Hallucinated target filtered out; only the offered candidate is linked.
    assert "[[Gradient Descent]]" in op.snippet
    assert "Made Up Note" not in op.snippet
    assert commit.call_args.kwargs["leash"].name == "orphan"


def test_orphan_no_link_when_model_picks_nothing_valid():
    agent = LeashedSubAgent()
    decision = OrphanLinkDecision(links=["Nonexistent"], rationale="nothing fits")
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="body")), \
         patch.object(agent, "_decide_links", return_value=decision), \
         patch("silica.agent.subagent.commit_ops") as commit:
        res = agent.handle(_orphan_item())
    assert res["status"] == "no_link"
    commit.assert_not_called()


def test_orphan_no_candidates():
    agent = LeashedSubAgent()
    res = agent.handle(WorkItem(kind="orphan", target_path="X.md", context={"candidates": []}))
    assert res["status"] == "no_candidates"
