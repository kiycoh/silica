"""Eval harness — Phase 6: autolink precision/recall on the golden set.

Measures how accurately autolink() links titles that should be linked
and avoids linking titles that should not be linked.

Run with: uv run pytest tests/golden/test_eval_autolink.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from silica.kernel.autolink import autolink, build_title_index

GOLDEN_PATH = Path(__file__).parent / "autolink_cases.json"


def _load_cases():
    return json.loads(GOLDEN_PATH.read_text())


@pytest.fixture(scope="module")
def golden_cases():
    return _load_cases()


@pytest.mark.parametrize("case", _load_cases(), ids=[c["id"] for c in _load_cases()])
def test_autolink_golden_case(case):
    """Each golden case checks that expected_links appear and expected_no_links don't."""
    body = case["body"]
    title_index = build_title_index(case["title_index"])

    _, added = autolink(body, title_index)
    added_set = set(added)

    for expected in case.get("expected_links", []):
        assert expected in added_set, (
            f"[{case['id']}] Expected '{expected}' to be linked, but got: {added_set}"
        )

    for not_expected in case.get("expected_no_links", []):
        assert not_expected not in added_set, (
            f"[{case['id']}] Expected '{not_expected}' NOT to be linked, but it was: {added_set}"
        )


def test_autolink_precision_recall_summary(golden_cases):
    """Compute and report precision/recall across all golden cases.

    This test always passes — it's a measurement tool, not a gate.
    The numbers should be checked manually and used to calibrate autolink.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for case in golden_cases:
        body = case["body"]
        title_index = build_title_index(case["title_index"])
        _, added = autolink(body, title_index)
        added_set = set(added)

        expected = set(case.get("expected_links", []))
        not_expected = set(case.get("expected_no_links", []))

        true_positives += len(added_set & expected)
        false_positives += len(added_set & not_expected)
        false_negatives += len(expected - added_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n[AUTOLINK EVAL] Precision={precision:.2%}  Recall={recall:.2%}  F1={f1:.2%}")
    print(f"  TP={true_positives}  FP={false_positives}  FN={false_negatives}")

    # Soft gate: warn but don't fail (thresholds are for monitoring, not CI blocking)
    assert precision >= 0.70, f"Autolink precision {precision:.2%} below 70% — review golden set"
    assert recall >= 0.70, f"Autolink recall {recall:.2%} below 70% — review golden set"
