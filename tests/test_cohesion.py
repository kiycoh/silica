"""Tests for the cohesion_pass sibling-linking step."""
from __future__ import annotations

import pytest
from silica.kernel.cohesion import cohesion_pass, _content_tokens


# ---------------------------------------------------------------------------
# _content_tokens unit tests
# ---------------------------------------------------------------------------

def test_tokens_strips_stopwords():
    tokens = _content_tokens("II Framework PEAS Actuators")
    assert "ii" not in tokens
    assert "peas" in tokens
    assert "actuators" in tokens
    # "framework" is NOT a stopword — specific framework names (STRIPS, PEAS) need it
    assert "framework" in tokens


def test_tokens_empty_string():
    assert _content_tokens("") == frozenset()


def test_tokens_only_stopwords():
    assert _content_tokens("di e in con su") == frozenset()


def test_tokens_min_length():
    # Single-character words are excluded (< 2 chars)
    tokens = _content_tokens("IA e ML")
    assert "e" not in tokens
    assert "ia" in tokens
    assert "ml" in tokens


# ---------------------------------------------------------------------------
# cohesion_pass — basic sibling detection
# ---------------------------------------------------------------------------

def _write_op(heading: str, title: str | None = None, related: list | None = None, source: str = "src.md") -> dict:
    op: dict = {
        "op": "write",
        "heading": heading,
        "path": f"Notes/{title or heading}.md",
        "source_basename": source,
        "hub": "Hub",
        "snippet": "...",
    }
    if title:
        op["title"] = title
    if related:
        op["related"] = related
    return op


def test_peas_siblings_all_linked():
    ops = [
        _write_op("II Framework PEAS Actuators", "PEAS Actuators"),
        _write_op("II Framework PEAS Sensors", "PEAS Sensors"),
        _write_op("II Framework PEAS Environment", "PEAS Environment"),
        _write_op("II Framework PEAS Performance Measure", "PEAS Performance Measure"),
    ]
    result = cohesion_pass(ops)
    # Every PEAS note should have the other three in its related list
    for i, op in enumerate(result):
        others = {result[j]["title"] for j in range(4) if j != i}
        assert set(op["related"]) >= others, f"op {i} missing siblings: {others - set(op['related'])}"


def test_two_sibling_ops_linked_bidirectionally():
    ops = [
        _write_op("Reti Neurali Convoluzionali"),
        _write_op("Reti Neurali Ricorrenti"),
    ]
    result = cohesion_pass(ops)
    assert "Reti Neurali Ricorrenti" in result[0]["related"]
    assert "Reti Neurali Convoluzionali" in result[1]["related"]


def test_no_siblings_when_no_shared_token():
    ops = [
        _write_op("Backpropagation"),
        _write_op("Alberi Decisionali"),
    ]
    result = cohesion_pass(ops)
    # No related injected — original dicts returned
    assert result[0].get("related") is None
    assert result[1].get("related") is None


def test_single_write_op_unchanged():
    ops = [_write_op("Backpropagation")]
    result = cohesion_pass(ops)
    assert result is ops  # fast path: same list returned


def test_zero_write_ops_unchanged():
    ops: list[dict] = []
    result = cohesion_pass(ops)
    assert result is ops


def test_non_write_ops_passed_through():
    ops = [
        {"op": "skip", "heading": "Foo", "source_basename": "s.md", "reason": "off-axis"},
        {"op": "patch", "heading": "Bar", "path": "Notes/Bar.md", "source_basename": "s.md", "snippet": "x"},
    ]
    result = cohesion_pass(ops)
    # No write ops → fast path
    assert result is ops


def test_existing_related_preserved_and_deduplicated():
    ops = [
        _write_op("PEAS Sensors", related=["Existing Note"]),
        _write_op("PEAS Actuators"),
    ]
    result = cohesion_pass(ops)
    related_sensors = result[0]["related"]
    assert "Existing Note" in related_sensors   # pre-existing entry preserved
    assert "PEAS Actuators" in related_sensors  # sibling injected
    assert related_sensors.count("PEAS Actuators") == 1  # no duplicate


def test_no_self_link():
    """An op must never appear in its own related list."""
    ops = [
        _write_op("PEAS Sensors"),
        _write_op("PEAS Actuators"),
    ]
    result = cohesion_pass(ops)
    assert "PEAS Sensors" not in result[0].get("related", [])
    assert "PEAS Actuators" not in result[1].get("related", [])


def test_title_preferred_over_heading_for_injection():
    """When title is set, siblings inject the title (not the raw heading)."""
    ops = [
        _write_op("II Framework PEAS Actuators", title="PEAS Actuators"),
        _write_op("II Framework PEAS Sensors", title="PEAS Sensors"),
    ]
    result = cohesion_pass(ops)
    # Injected name should be the title, not the compound heading
    assert "PEAS Sensors" in result[0]["related"]
    assert "II Framework PEAS Sensors" not in result[0]["related"]


def test_mixed_write_and_patch_only_writes_receive_siblings():
    ops = [
        _write_op("PEAS Sensors"),
        {"op": "patch", "heading": "PEAS Actuators", "path": "Notes/PEAS Actuators.md",
         "source_basename": "src.md", "snippet": "extra"},
        _write_op("PEAS Environment"),
    ]
    result = cohesion_pass(ops)
    # Only the two write ops are siblings
    assert "PEAS Environment" in result[0]["related"]
    assert "PEAS Sensors" in result[2]["related"]
    # Patch op is unchanged
    assert result[1] is ops[1]
