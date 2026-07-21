# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

from unittest.mock import MagicMock, patch

from silica.capabilities.dedup import (
    DedupDecision,
    _gated_batch_decisions,
    passes_dedup_gate,
    run_dedup,
)
from silica.config import SilicaConfig
from silica.kernel.ops import OpType
from silica.kernel.workqueue import WorkItem


def test_high_cosine_similar_size_passes():
    assert passes_dedup_gate(0.90, incoming_len=500, candidate_len=600) is True


def test_size_guard_rejects_spoke_in_hub():
    # High cosine but the "candidate" is a 10x-larger hub -> not a merge pair.
    assert passes_dedup_gate(0.90, incoming_len=200, candidate_len=5000) is False


def test_below_threshold_rejected():
    assert passes_dedup_gate(0.70, incoming_len=500, candidate_len=600) is False


def test_threshold_boundary_inclusive():
    assert passes_dedup_gate(0.85, incoming_len=500, candidate_len=600) is True


# ---------------------------------------------------------------------------
# Gate-ON ROUTING (Task 9): the failure mode being pinned here is data loss /
# verdict mis-routing, not the pure predicate above.
# ---------------------------------------------------------------------------


def test_gate_on_low_score_skips_llm_and_authors_spoke(monkeypatch):
    """Single path (run_dedup), SILICA_DEDUP_GATE=1, effective score below the
    0.85 threshold: the gated-out pair must route a synthetic
    DedupDecision(verdict="distinct") through _route_verdict WITHOUT ever
    calling the LLM judge (_decide_dedup). Because this is a pipeline item
    (ctx["target_dir"] set), _route_distinct must author the incoming concept
    as a spoke WRITE op — not degrade to a bare no_merge that silently drops
    the concept."""
    monkeypatch.setenv("SILICA_DEDUP_GATE", "1")

    item = WorkItem(
        kind="dedup",
        target_path="Concepts/Gradient Descent.md",
        context={
            "concept": "Discesa del gradiente",
            "excerpt": "Variante mini-batch con momentum.",
            "candidate": "Gradient Descent",
            "inbox_file": "Inbox/ml.md",
            "hub": "Concepts",
            "target_dir": "Concepts",   # pipeline item
            "score": 0.30,              # well below the 0.85 gate threshold
        },
        reason="test",
    )

    llm_called = MagicMock(side_effect=AssertionError(
        "gated-out pair must not reach the LLM judge"))

    with patch("silica.driver.DRIVER.read_note", return_value=MagicMock(content="candidate body")), \
         patch("silica.capabilities.dedup._decide_dedup", llm_called), \
         patch("silica.capabilities.dedup.commit_ops",
               return_value={"status": "committed", "committed": 1}) as commit:
        res = run_dedup(item, SilicaConfig())

    assert llm_called.call_count == 0, "LLM judge must be skipped when the gate rejects"
    assert res["status"] == "committed"
    assert res["verdict"] == "distinct"
    assert "gate" in res["rationale"]

    ops_arg = commit.call_args.args[0]
    assert len(ops_arg) == 1
    op = ops_arg[0]
    # Spoke authored (write), NOT a dropped no_merge / patch on the candidate.
    assert op.op == OpType.write
    assert op.path == "Concepts/Discesa del gradiente.md"
    assert "Variante mini-batch con momentum." in op.snippet  # excerpt authored verbatim
    assert "[[Gradient Descent]]" in op.snippet                # born linked


def test_gated_batch_decisions_full_length_and_index_aligned():
    """Batch path: _gated_batch_decisions is the pure alignment logic under
    test. 4 concepts, interleaved gate outcomes (fail at 0: low score, fail
    at 2: size mismatch; pass at 1 and 3). Only the passing subset must reach
    _decide_dedup_batch, and the returned list must stay full-length and
    index-aligned to `concepts` so the caller's zip(concepts, decisions)
    never silently drops a concept."""
    item = WorkItem(
        kind="dedup",
        target_path="Reti/MQTT.md",
        context={"candidate": "MQTT", "hub": "Reti", "target_dir": "Reti"},
        reason="test",
    )
    candidate_body = "x" * 200

    concepts = [
        {"concept": "C0", "excerpt": "e" * 150, "score": 0.30},   # gate: low score
        {"concept": "C1", "excerpt": "e" * 150, "score": 0.90},   # gate: passes
        {"concept": "C2", "excerpt": "e" * 1000, "score": 0.90},  # gate: size mismatch
        {"concept": "C3", "excerpt": "e" * 150, "score": 0.92},   # gate: passes
    ]

    llm_decisions = [
        DedupDecision(verdict="duplicate", rationale="C1 merge", addition="new info"),
        DedupDecision(verdict="distinct", rationale="C3 distinct", title="T3", body="body3"),
    ]

    with patch("silica.capabilities.dedup._decide_dedup_batch",
               return_value=llm_decisions) as mock_batch:
        result = _gated_batch_decisions(SilicaConfig(), item, concepts, candidate_body)

    # Only the gated-IN subset (C1, C3) is sent to the LLM.
    mock_batch.assert_called_once()
    assert mock_batch.call_args.kwargs["concepts"] == [concepts[1], concepts[3]]

    # Full-length, index-aligned to `concepts` — no concept dropped.
    assert len(result) == len(concepts)
    assert result[0].verdict == "distinct" and "gate" in result[0].rationale
    assert result[2].verdict == "distinct" and "gate" in result[2].rationale
    assert result[1] == llm_decisions[0]
    assert result[3] == llm_decisions[1]
