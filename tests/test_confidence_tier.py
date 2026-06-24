"""Confidence → tier: the tier follows an edge's provenance, not a flat default.

Vocabulary ported from Graphify (MIT): EXTRACTED / INFERRED / AMBIGUOUS. Applied
to embedding-proposed missing links — "embeddings propose, the graph disposes":
a proposal the graph also corroborates (shares a neighbour) is EXTRACTED → auto;
a strong-but-uncorroborated cosine is INFERRED → propose.
"""
from __future__ import annotations

from silica.kernel.graph_report import AutolinkCandidate, MissingLink, _empty_report
from silica.kernel.analyst_plan import (
    build_task_plan,
    classify_autolink,
    classify_missing_link,
    tier_for_confidence,
)


def test_tier_for_confidence_maps_provenance_to_tier() -> None:
    assert tier_for_confidence("EXTRACTED") == "auto"
    assert tier_for_confidence("INFERRED") == "propose"
    assert tier_for_confidence("AMBIGUOUS") == "escalate"


def test_classify_missing_link_extracted_when_graph_corroborates() -> None:
    # d_prev == 2 → source and target share a neighbour → structural corroboration.
    ml = MissingLink(source="A", target="B", cosine=0.90, d_prev=2)
    assert classify_missing_link(ml, tau_high=0.85) == "EXTRACTED"


def test_classify_missing_link_inferred_when_embedding_only() -> None:
    # d_prev == 0 → unreachable → embedding signal only, no graph corroboration.
    ml = MissingLink(source="A", target="B", cosine=0.90, d_prev=0)
    assert classify_missing_link(ml, tau_high=0.85) == "INFERRED"


def test_classify_missing_link_inferred_when_cosine_below_tau_high() -> None:
    ml = MissingLink(source="A", target="B", cosine=0.70, d_prev=2)
    assert classify_missing_link(ml, tau_high=0.85) == "INFERRED"


def test_classify_autolink_inferred_when_concepts_shared() -> None:
    # A directly shared concept is textual evidence the two notes cover the same thing.
    cand = AutolinkCandidate(source="A", target="B", weight=3.0, shared=["neural network"])
    assert classify_autolink(cand) == "INFERRED"


def test_classify_autolink_ambiguous_when_associative_only() -> None:
    # No shared concept → related only through transitive expansion → needs a human.
    cand = AutolinkCandidate(source="A", target="B", weight=3.0, shared=[])
    assert classify_autolink(cand) == "AMBIGUOUS"


def test_evidenced_autolink_candidate_is_proposed() -> None:
    # Embedder-free leg: a co-occurrence autolink with shared concepts enters the
    # plan as a propose (INFERRED), even with no embedding missing-links present.
    r = _empty_report()
    r.autolink_candidates = [AutolinkCandidate(source="A", target="B", weight=3.0, shared=["x"])]
    plan = build_task_plan(r)

    propose_sources = [p for c in plan.propose for p in c.payload.get("note_paths", [])]
    assert "A" in propose_sources
    assert any(
        c.confidence == "INFERRED"
        for c in plan.propose if "A" in c.payload.get("note_paths", [])
    )


def test_associative_autolink_candidate_is_escalated() -> None:
    # An associative-only pair (no shared concept) is flagged for human review.
    r = _empty_report()
    r.autolink_candidates = [AutolinkCandidate(source="A", target="B", weight=3.0, shared=[])]
    plan = build_task_plan(r)

    assert any(c.confidence == "AMBIGUOUS" for c in plan.escalate)
    # AMBIGUOUS is review-only: it must not auto-link.
    auto_sources = [p for c in plan.auto for p in c.payload.get("note_paths", [])]
    assert "A" not in auto_sources


def test_corroborated_missing_link_is_auto_not_propose() -> None:
    r = _empty_report()
    r.missing_links = [MissingLink(source="A", target="B", cosine=0.90, d_prev=2)]
    plan = build_task_plan(r)

    auto_sources = [p for c in plan.auto for p in c.payload.get("note_paths", [])]
    propose_sources = [p for c in plan.propose for p in c.payload.get("note_paths", [])]
    assert "A" in auto_sources
    assert "A" not in propose_sources
    # The candidate carries the provenance that drove the tier.
    assert any(c.confidence == "EXTRACTED" for c in plan.auto if "A" in c.payload.get("note_paths", []))
