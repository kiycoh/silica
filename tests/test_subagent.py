"""Tests for capability dispatch (silica/agent/subagent.py) and the per-capability
behaviours (silica/capabilities/*).

Dispatch is a keyed lookup: ``BoundedSubAgent.handle()`` selects the capability
registered under ``item.kind`` and runs it. Each behaviour is a plain
``run(item, config) -> dict`` function in its own module, and its LLM-decision
seam is a module-level function the tests patch directly.
"""
from unittest.mock import patch, MagicMock

from silica.agent.subagent import BoundedSubAgent
from silica.capabilities.dedup import run_dedup, DedupDecision
from silica.capabilities.refine import run_refine
from silica.capabilities.enrich import run_enrich
from silica.capabilities.orphan import run_orphan, OrphanLinkDecision
from silica.capabilities._base import NoteContent
from silica.config import SilicaConfig
from silica.kernel.ops import OpType
from silica.kernel.workqueue import WorkItem

CONFIG = SilicaConfig()


# --- dispatch --------------------------------------------------------------

def test_handle_dispatches_to_capability_by_kind():
    """handle() routes to the capability registered under item.kind."""
    seen = {}

    def fake_run(item, config):
        seen["called"] = item.kind
        return {"status": "ok"}

    agent = BoundedSubAgent(CONFIG, capabilities={"mystery": fake_run})
    res = agent.handle(WorkItem(kind="mystery", target_path="X.md"))
    assert res == {"status": "ok"}
    assert seen["called"] == "mystery"


def test_handle_skips_unknown_kind():
    agent = BoundedSubAgent(CONFIG, capabilities={})
    res = agent.handle(WorkItem(kind="nope", target_path="X.md"))
    assert res["status"] == "skipped"


def test_handle_catches_capability_errors():
    def boom(item, config):
        raise RuntimeError("kaboom")

    agent = BoundedSubAgent(CONFIG, capabilities={"boom": boom})
    res = agent.handle(WorkItem(kind="boom", target_path="X.md"))
    assert res["status"] == "error"
    assert "kaboom" in res["error"]


def test_default_registry_covers_builtin_kinds():
    from silica.capabilities import CAPABILITIES
    # Note-level behaviours plus every worker profile (kind == profile name):
    # one registry dispatches all background work.
    assert set(CAPABILITIES) == {"dedup", "refine", "enrich", "orphan", "reader", "router"}


# --- dedup behaviour -------------------------------------------------------

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
    decision = DedupDecision(verdict="duplicate", rationale="same concept", addition="### Momentum\nNew info.")

    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="existing body")), \
         patch("silica.capabilities.dedup._decide_dedup", return_value=decision), \
         patch("silica.capabilities.dedup.commit_ops", return_value={"status": "committed", "committed": 1}) as commit:
        res = run_dedup(_item(), CONFIG)

    assert res["status"] == "committed"
    # commit_ops called with exactly one patch op + a dedup leash on the candidate.
    ops_arg = commit.call_args.args[0]
    assert len(ops_arg) == 1
    assert ops_arg[0].op == OpType.patch
    assert ops_arg[0].path == "Concepts/Gradient Descent.md"
    bounds = commit.call_args.kwargs["bounds"]
    assert bounds.name == "dedup"
    assert OpType.patch in bounds.allowed_ops and OpType.overwrite not in bounds.allowed_ops


def test_dedup_no_merge_when_not_duplicate():
    decision = DedupDecision(verdict="distinct", rationale="different topics")
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="body")), \
         patch("silica.capabilities.dedup._decide_dedup", return_value=decision), \
         patch("silica.capabilities.dedup.commit_ops") as commit:
        res = run_dedup(_item(), CONFIG)
    assert res["status"] == "no_merge"
    commit.assert_not_called()


def test_dedup_no_merge_when_addition_empty():
    decision = DedupDecision(verdict="duplicate", rationale="dup but nothing new", addition="   ")
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="body")), \
         patch("silica.capabilities.dedup._decide_dedup", return_value=decision), \
         patch("silica.capabilities.dedup.commit_ops") as commit:
        res = run_dedup(_item(), CONFIG)
    assert res["status"] == "no_merge"
    commit.assert_not_called()


def test_dedup_contradicts_builds_contested_patch():
    """Third verdict: the conflicting claim lands as ONE contested patch op —
    warning callout in the snippet, contested_by set for the frontmatter mark."""
    decision = DedupDecision(
        verdict="contradicts",
        rationale="conflicting dosage",
        addition="Il dosaggio raccomandato è 50mg/die.",
    )
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="existing body")), \
         patch("silica.capabilities.dedup._decide_dedup", return_value=decision), \
         patch("silica.capabilities.dedup.commit_ops", return_value={"status": "committed", "committed": 1}) as commit:
        res = run_dedup(_item(), CONFIG)

    assert res["status"] == "committed"
    assert res["verdict"] == "contradicts"
    ops_arg = commit.call_args.args[0]
    assert len(ops_arg) == 1
    op = ops_arg[0]
    assert op.op == OpType.patch
    assert op.path == "Concepts/Gradient Descent.md"
    assert op.contested_by == "fonte: ml.md"
    assert op.snippet.startswith("> [!warning]")
    assert "50mg/die" in op.snippet
    # Same leash as the merge path: the model never escalates beyond a patch.
    bounds = commit.call_args.kwargs["bounds"]
    assert bounds.name == "dedup"


def test_dedup_contradicts_without_claim_is_no_merge():
    """A contradiction verdict with no quoted claim is unactionable — never write."""
    decision = DedupDecision(verdict="contradicts", rationale="conflict", addition="  ")
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="body")), \
         patch("silica.capabilities.dedup._decide_dedup", return_value=decision), \
         patch("silica.capabilities.dedup.commit_ops") as commit:
        res = run_dedup(_item(), CONFIG)
    assert res["status"] == "no_merge"
    commit.assert_not_called()


def _decide_with_raw(raw_text: str):
    """Run _decide_dedup with a provider whose response is `raw_text`."""
    from silica.capabilities.dedup import _decide_dedup

    provider = MagicMock()
    provider.call_llm.return_value = MagicMock(text=raw_text)
    with patch("silica.agent.providers.get_provider", return_value=provider):
        return _decide_dedup(
            CONFIG, concept="X", excerpt="e", candidate_name="Y", candidate_body="b",
        )


def test_decide_dedup_unparseable_defaults_to_distinct():
    """Conservative default: garbage output must never become a contradicts/merge."""
    decision = _decide_with_raw("non-JSON garbage")
    assert decision.verdict == "distinct"


def test_decide_dedup_unknown_verdict_defaults_to_distinct():
    decision = _decide_with_raw('{"verdict": "maybe", "rationale": "", "addition": "x"}')
    assert decision.verdict == "distinct"


def test_decide_dedup_parses_contradicts():
    decision = _decide_with_raw(
        '{"verdict": "contradicts", "rationale": "r", "addition": "claim"}'
    )
    assert decision.verdict == "contradicts"
    assert decision.addition == "claim"


def test_decide_dedup_legacy_is_duplicate_still_maps():
    """A model answering with the old binary schema degrades gracefully."""
    assert _decide_with_raw('{"is_duplicate": true, "addition": "x"}').verdict == "duplicate"
    assert _decide_with_raw('{"is_duplicate": false}').verdict == "distinct"


def test_unreadable_candidate_is_skipped():
    with patch("silica.driver.DRIVER.read_note", side_effect=RuntimeError("missing")):
        res = run_dedup(_item(), CONFIG)
    assert res["status"] == "skipped"


# --- refine behaviour ------------------------------------------------------

def _refine_item():
    return WorkItem(kind="refine", target_path="Notes/Target.md", context={"hub": "Concepts"})


def test_refine_builds_overwrite_under_refiner_bounds():
    refined = NoteContent(content="# Target\n\n> [!note]\nBody with [[Link]].")
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="old body [[Link]]")), \
         patch("silica.capabilities.refine._refine_note", return_value=refined), \
         patch("silica.capabilities.refine.commit_ops", return_value={"status": "committed", "committed": 1}) as commit:
        res = run_refine(_refine_item(), CONFIG)
    assert res["status"] == "committed"
    ops_arg = commit.call_args.args[0]
    assert ops_arg[0].op == OpType.overwrite
    bounds = commit.call_args.kwargs["bounds"]
    assert bounds.name == "refiner"
    assert bounds.content_guard is not None  # anti-info-loss enforced


def test_refine_skips_empty_note():
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="   ")):
        res = run_refine(_refine_item(), CONFIG)
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
    # Model returns one valid candidate + one hallucinated name.
    decision = OrphanLinkDecision(links=["Gradient Descent", "Made Up Note"], rationale="related")
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="orphan body")), \
         patch("silica.capabilities.orphan._decide_links", return_value=decision), \
         patch("silica.capabilities.orphan.commit_ops", return_value={"status": "committed", "committed": 1}) as commit:
        res = run_orphan(_orphan_item(), CONFIG)
    assert res["status"] == "committed"
    op = commit.call_args.args[0][0]
    assert op.op == OpType.patch and op.path == "Notes/Lonely.md"
    # Hallucinated target filtered out; only the offered candidate is linked.
    assert "[[Gradient Descent]]" in op.snippet
    assert "Made Up Note" not in op.snippet
    assert commit.call_args.kwargs["bounds"].name == "orphan"


def test_orphan_no_link_when_model_picks_nothing_valid():
    decision = OrphanLinkDecision(links=["Nonexistent"], rationale="nothing fits")
    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="body")), \
         patch("silica.capabilities.orphan._decide_links", return_value=decision), \
         patch("silica.capabilities.orphan.commit_ops") as commit:
        res = run_orphan(_orphan_item(), CONFIG)
    assert res["status"] == "no_link"
    commit.assert_not_called()


def test_orphan_no_candidates():
    res = run_orphan(WorkItem(kind="orphan", target_path="X.md", context={"candidates": []}), CONFIG)
    assert res["status"] == "no_candidates"


def test_orphan_hub_is_none_when_context_has_no_hub():
    """When context has no hub key, hub must be None (not basename of target_path)."""
    import silica.capabilities.orphan as orphan_module
    from silica.agent.bounds import orphan_bounds as real_orphan_bounds

    item = WorkItem(
        kind="orphan",
        target_path="notes/MyNote.md",
        context={"candidates": [{"name": "Other", "path": "notes/Other.md"}]},
        reason="test",
    )

    captured_hubs = []

    def capture_orphan_bounds(target, *, hub):
        captured_hubs.append(hub)
        return real_orphan_bounds(target, hub=hub)

    with patch.object(orphan_module, "orphan_bounds", side_effect=capture_orphan_bounds), \
         patch("silica.capabilities.orphan.commit_ops", return_value={"status": "no_ops"}), \
         patch("silica.capabilities.orphan._decide_links", return_value=OrphanLinkDecision(links=["Other"], rationale="test")), \
         patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="# MyNote\n")):
        run_orphan(item, CONFIG)

    assert captured_hubs == [None], f"Expected hub=None when context has no 'hub' key, got {captured_hubs}"
