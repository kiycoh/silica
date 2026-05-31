"""Tests for silica/kernel/graph_report.py.

Uses a synthetic deterministic graph (2 Louvain clusters connected by one bridge
edge, one orphan note) without touching a live driver or Obsidian.
"""
from __future__ import annotations

import dataclasses

import pytest

from silica.kernel.graph_report import (
    BridgeStat,
    ClusterStat,
    MissingLink,
    NodeStat,
    VaultReport,
    _empty_report,
    compute_report,
    to_digest,
    to_facts,
    to_markdown,
    write_report,
)


# ---------------------------------------------------------------------------
# Synthetic graph fixture
#
# Layout:
#   Cluster 0: A ↔ B ↔ C   (triangle-ish)
#   Cluster 1: D ↔ E
#   Bridge:    C → D        (cross-cluster, single shared neighbour: none)
#   Orphan:    F             (no incoming links)
#
# Nodes: A, B, C (cluster 0), D, E (cluster 1), F (orphan, no cluster)
# EXTRACTED edges: A↔B, B↔C, A↔C, D↔E, C→D
# Ghost/AMBIGUOUS: F → __unresolved__Ghost
# ---------------------------------------------------------------------------

def _make_node(nid: str, label: str, group: int, note_type: str = "note") -> dict:
    return {"id": nid, "label": label, "group": group, "type": note_type}


def _make_edge(eid: str, src: str, dst: str, edge_type: str = "EXTRACTED") -> dict:
    return {"id": eid, "from": src, "to": dst, "type": edge_type}


@pytest.fixture()
def synthetic_graph():
    """Return (nodes, edges) for the synthetic test vault."""
    nodes = [
        _make_node("A", "Alpha",   group=0),
        _make_node("B", "Beta",    group=0),
        _make_node("C", "Gamma",   group=0),
        _make_node("D", "Delta",   group=1),
        _make_node("E", "Epsilon", group=1),
        _make_node("F", "Phi",     group=-1),  # orphan, no cluster
        # Ghost node for the unresolved link from F
        {"id": "__unresolved__Ghost", "label": "Ghost", "group": -1, "type": "ghost"},
    ]
    edges = [
        _make_edge("e0", "A", "B"),
        _make_edge("e1", "B", "C"),
        _make_edge("e2", "A", "C"),
        _make_edge("e3", "D", "E"),
        _make_edge("e4", "C", "D"),  # cross-cluster bridge
        _make_edge("e5", "F", "__unresolved__Ghost", "AMBIGUOUS"),
    ]
    return nodes, edges


@pytest.fixture()
def report(synthetic_graph):
    nodes, edges = synthetic_graph
    return compute_report(_nodes_edges_override=(nodes, edges))


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

def test_report_is_vault_report(report):
    assert isinstance(report, VaultReport)


def test_totals(report):
    t = report.totals
    assert t["notes"] == 6
    assert t["links"] == 5        # 5 EXTRACTED edges
    assert t["unresolved"] == 1   # 1 AMBIGUOUS edge
    assert t["orphans"] >= 1      # F has no incoming links; D has 1 (from C)


def test_orphan_F_present(report):
    assert "F" in report.orphans


def test_god_nodes_sorted_by_degree(report):
    # The highest-degree node should appear first
    assert len(report.god_nodes) > 0
    degrees = [n.degree for n in report.god_nodes]
    assert degrees == sorted(degrees, reverse=True)


def test_god_nodes_no_ghost(report):
    """Ghost nodes must never appear in god_nodes."""
    for n in report.god_nodes:
        assert not n.id.startswith("__unresolved__")


def test_bridges_detected(report):
    """C→D is a cross-cluster bridge; the report must contain at least one bridge."""
    assert len(report.bridges) >= 1
    bridge_pairs = {(b.source, b.target) for b in report.bridges} | \
                   {(b.target, b.source) for b in report.bridges}
    assert ("C", "D") in bridge_pairs or ("D", "C") in bridge_pairs


def test_bridges_different_clusters(report):
    for b in report.bridges:
        assert b.source_cluster != b.target_cluster


def test_clusters_present(report):
    assert len(report.clusters) >= 1


def test_dangling_ghost_aggregated(report):
    """Ghost link from F should appear in dangling as target='Ghost', refs=1."""
    targets = {d["target"]: d["refs"] for d in report.dangling}
    assert "Ghost" in targets
    assert targets["Ghost"] == 1


# ---------------------------------------------------------------------------
# Empty vault / no edges degrades gracefully
# ---------------------------------------------------------------------------

def test_empty_vault_no_exception():
    nodes = [_make_node("X", "X", group=-1)]
    edges = []
    r = compute_report(_nodes_edges_override=(nodes, edges))
    assert isinstance(r, VaultReport)
    assert r.totals["notes"] == 1
    assert r.totals["links"] == 0
    # A single isolated node has degree=0 but still appears in god_nodes
    assert len(r.god_nodes) <= 1
    assert r.bridges == []
    assert r.clusters == []


def test_empty_report_helper():
    r = _empty_report("some/folder")
    assert r.scope == "some/folder"
    assert all(v == 0 for v in r.totals.values())


# ---------------------------------------------------------------------------
# to_facts
# ---------------------------------------------------------------------------

def test_to_facts_keys(report):
    facts = to_facts(report)
    assert set(facts.keys()) == {"scope", "totals", "god_nodes", "top_bridges", "orphan_count", "dangling_top"}


def test_to_facts_god_nodes_are_ids(report):
    facts = to_facts(report)
    # Each entry should be a string (node id)
    for gn in facts["god_nodes"]:
        assert isinstance(gn, str)


def test_to_facts_dangling_top_capped(report):
    facts = to_facts(report)
    assert len(facts["dangling_top"]) <= 5


# ---------------------------------------------------------------------------
# to_digest
# ---------------------------------------------------------------------------

def test_to_digest_non_empty(report):
    digest = to_digest(report)
    assert len(digest) > 0
    assert "VAULT AUDIT" in digest


def test_to_digest_empty_vault():
    r = _empty_report()
    digest = to_digest(r)
    assert "VAULT AUDIT" in digest
    assert "notes=0" in digest


def test_to_digest_contains_orphan(report):
    digest = to_digest(report)
    assert "ORPHANS" in digest
    assert "Phi" in digest or "F" in digest  # F's label is "Phi"


# ---------------------------------------------------------------------------
# to_markdown
# ---------------------------------------------------------------------------

def test_to_markdown_sections(report):
    md = to_markdown(report)
    assert "## Totals" in md
    assert "## God Nodes" in md
    assert "## Clusters" in md
    assert "## Orphans" in md
    assert "## Dangling Links" in md
    assert "## Surprising Cross-Cluster" in md


def test_to_markdown_no_proposed_section_when_empty(report):
    """Missing links section should be absent when missing_links is empty."""
    assert not report.missing_links
    md = to_markdown(report)
    assert "Proposed Missing Links" not in md


def test_to_markdown_proposed_section_when_present():
    r = _empty_report()
    r.missing_links = [MissingLink(source="X", target="Y", cosine=0.91)]
    md = to_markdown(r)
    assert "Proposed Missing Links" in md


# ---------------------------------------------------------------------------
# write_report
# ---------------------------------------------------------------------------

def test_write_report_creates_files(tmp_path, report):
    out = str(tmp_path / "GRAPH_REPORT.md")
    result = write_report(report, out)
    assert "path_md" in result
    assert "path_json" in result
    import os
    assert os.path.exists(result["path_md"])
    assert os.path.exists(result["path_json"])


def test_write_report_json_deserializable(tmp_path, report):
    import orjson
    out = str(tmp_path / "GRAPH_REPORT.md")
    result = write_report(report, out)
    data = orjson.loads(open(result["path_json"], "rb").read())
    assert "totals" in data
    assert "god_nodes" in data


# ---------------------------------------------------------------------------
# Determinism: same input → same output
# ---------------------------------------------------------------------------

def test_to_facts_byte_stable(synthetic_graph):
    """to_facts on identical input must produce identical dicts."""
    import orjson
    nodes, edges = synthetic_graph
    r1 = compute_report(_nodes_edges_override=(nodes, edges))
    r2 = compute_report(_nodes_edges_override=(nodes, edges))
    # generated_at will differ — compare only structural fields
    f1 = to_facts(r1)
    f2 = to_facts(r2)
    f1.pop("totals", None)  # totals are deterministic, but keep the check focused
    f2.pop("totals", None)
    assert orjson.dumps(f1, option=orjson.OPT_SORT_KEYS) == orjson.dumps(f2, option=orjson.OPT_SORT_KEYS)
