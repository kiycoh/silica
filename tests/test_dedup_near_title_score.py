"""Near-title dedup WorkItems must carry the real fuzzy title ratio as
title_score, so SILICA_DEDUP_GATE judges them on their true similarity instead
of the degenerate score=0.0 (which force-routes every near-title pair distinct).
"""
from types import SimpleNamespace

from silica.capabilities.dedup import passes_dedup_gate
from silica.router.states.distill import _enqueue_near_title_dedups


class _FakeQueue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.append(item)


def _fsm(queue):
    return SimpleNamespace(
        work_queue=queue, inbox_file="in.md", hub="Hub",
        _current_content_hash="h", target_dir="TargetDir",
    )


def test_ratio_becomes_title_score_and_clears_gate():
    q = _FakeQueue()
    rejected = [{
        "op": {"heading": "Beta", "snippet": "beta body"},
        "reason": "near_title candidate='Betas' path='TargetDir/Betas.md' "
                  "ratio=0.90 — deferred for dedup review",
    }]
    _enqueue_near_title_dedups(_fsm(q), rejected)
    assert len(q.items) == 1
    ctx = q.items[0].context
    assert ctx["title_score"] == 0.90              # real ratio, not 0.0
    # ratio 0.90 >= 0.85 threshold, similar sizes -> reaches the LLM judge
    assert passes_dedup_gate(ctx["title_score"], 500, 600) is True
    # sanity: the old degenerate score=0.0 would have force-gated it out
    assert passes_dedup_gate(0.0, 500, 600) is False


def test_missing_ratio_stays_zero():
    q = _FakeQueue()
    rejected = [{
        "op": {"heading": "Beta", "snippet": "x"},
        "reason": "near_title candidate='Betas' path='TargetDir/Betas.md'",
    }]
    _enqueue_near_title_dedups(_fsm(q), rejected)
    assert q.items[0].context["title_score"] == 0.0  # graceful when absent
