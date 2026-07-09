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
    # Full report: these tests assert god_nodes/bridges/cohesion, which are the
    # analytics signals the on-demand /graph and /report commands consume.
    return compute_report(_nodes_edges_override=(nodes, edges), analytics=True)


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

def test_report_is_vault_report(report):
    assert isinstance(report, VaultReport)


def test_totals(report):
    t = report.totals
    assert t["notes"] == 6
    assert t["links"] == 5        # 5 EXTRACTED edges
    assert t["dangling_links"] == 1   # 1 AMBIGUOUS edge
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


def test_to_markdown_folds_long_lists_into_callouts():
    """Long lists fold into collapsed OFM callouts (`[!kind]-`); the health/tip
    summaries stay open. Wikilinks must survive inside the `>`-quoted callout."""
    r = _empty_report()
    r.totals = {"notes": 3, "orphans": 2, "dangling_links": 1, "lean_notes": 1}
    r.clusters = [ClusterStat(cluster_id=0, hub="Hub", size=2, cohesion=0.5, members=["A", "B"])]
    r.orphans = ["A", "B"]
    r.dangling = [{"target": "X", "refs": 2}]
    r.lean_notes = ["L"]
    md = to_markdown(r)
    assert "> [!warning] Health" in md          # health summary — open, not folded
    assert "> [!tip] Suggestions" in md         # fixes summary — open
    assert "> [!abstract]- [[Hub]]" in md       # cluster — folded
    assert "> [!warning]- 2 orphans" in md      # orphans — folded
    assert "> [!bug]- 1 broken links" in md     # dangling — folded
    assert "> - [[A]]" in md                    # wikilink bullet survives inside callout


def test_is_vault_artifact_matches_root_only():
    from silica.kernel.graph_export import is_vault_artifact
    assert is_vault_artifact("GRAPH_REPORT.md")
    assert is_vault_artifact("log")
    assert not is_vault_artifact("Concepts/log.md")   # a real note in a subfolder
    assert not is_vault_artifact("Statistica.md")


def test_build_graph_data_excludes_vault_artifacts(tmp_vault):
    """GRAPH_REPORT.md/log.md are Silica's own output — they must stay out of the
    graph, or the report's own `[[...]]` would zero the orphan count next run."""
    from silica.kernel.graph_export import build_graph_data

    tmp_vault.note("Real.md", "# Real\nNo links here.\n")
    tmp_vault.note("GRAPH_REPORT.md", "# Report\n[[Real]]\n")   # report links Real
    tmp_vault.note("log.md", "# Log\n[[Real]]\n")

    nodes, edges = build_graph_data()
    ids = {n["id"] for n in nodes}
    assert "GRAPH_REPORT.md" not in ids and "log.md" not in ids
    assert "Real.md" in ids
    # Real is linked ONLY by the artifacts -> excluded, it stays an orphan
    assert not any(e["to"] == "Real.md" for e in edges)


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


# ---------------------------------------------------------------------------
# _compute_missing_links — common_neighbors structural boost (paper #2)
# ---------------------------------------------------------------------------

def test_missing_links_common_neighbors_boosts_ranking(monkeypatch):
    """Two candidates with equal cosine rank by shared-neighbor count.

    Paper (Marwitz et al. 2026) Baseline uses sum_i A^2_u,i (2-length path
    count) as a core feature. Silica equivalent: candidates sharing more
    common neighbors with the source are likelier to form a real link and
    must rank strictly higher than equally-similar but structurally-isolated
    candidates.
    """
    import networkx as nx
    from silica.kernel import graph_report as gr

    # S reaches A through two shared neighbors (X, Y) and B through one (Z).
    # Both A and B sit at shortest-path distance 2 from S (so both clear the
    # d_prev > 1 gate), and both will be returned with identical cosine.
    G = nx.Graph()
    G.add_edges_from([
        ("S", "X"), ("S", "Y"), ("S", "Z"),
        ("A", "X"), ("A", "Y"),
        ("B", "Z"),
    ])

    class _Store:
        _notes: dict = {}

        def __len__(self):
            return 6

        def get_vec(self, p):
            return [1.0, 0.0] if p == "S" else None

        def cosine_top_k(self, vec, k=10, exclude=None):
            return [
                {"path": "A", "score": 0.90},
                {"path": "B", "score": 0.90},
            ]

    monkeypatch.setattr("silica.kernel.embed.EmbedStore", _Store)
    monkeypatch.setattr("silica.agent.providers.get_embedder", lambda cfg: object())

    report = VaultReport(
        generated_at="x", scope="", totals={},
        god_nodes=[NodeStat(id="S", label="S", cluster=0,
                            out_degree=3, in_degree=0, degree=3, pagerank=0.0)],
        bridges=[], orphans=[], dangling=[], clusters=[],
    )

    links = gr._compute_missing_links(report, G, tau=0.5, k=10)
    by_target = {l.target: l for l in links}

    assert "A" in by_target and "B" in by_target
    # A shares 2 neighbors with S, B shares 1 → A must score strictly higher.
    assert by_target["A"].cosine > by_target["B"].cosine
    # …and the result list is ordered accordingly.
    pairs = [(l.source, l.target) for l in links]
    assert pairs.index(("S", "A")) < pairs.index(("S", "B"))


def test_duplicate_pairs_split_confirmed_vs_borderline(monkeypatch):
    """≥ τ_high → confirmed (merge candidate); τ_low..τ_high → borderline; ≤ τ_low dropped."""
    from silica.kernel import graph_report as gr

    nn = {  # each note's single nearest neighbour: (target, cosine)
        "a": ("b", 0.92),  # ≥ 0.85  → confirmed
        "c": ("d", 0.70),  # 0.65..0.85 → borderline
        "e": ("f", 0.50),  # ≤ 0.65  → dropped
    }

    class _Store:
        _notes: dict = {}
        def __len__(self): return len(nn)
        def paths(self): return list(nn)
        def get_vec(self, p): return [p] if p in nn else None
        def cosine_top_k(self, vec, k=1, exclude=None):
            tgt, score = nn[vec[0]]
            return [{"path": tgt, "score": score}]

    monkeypatch.setattr("silica.kernel.embed.EmbedStore", _Store)

    report = VaultReport(
        generated_at="x", scope="", totals={},
        god_nodes=[], bridges=[], orphans=[], dangling=[], clusters=[],
    )
    borderline, confirmed = gr._compute_duplicate_pairs(report)

    assert [(d.source, d.target) for d in confirmed] == [("a", "b")]
    assert [(d.source, d.target) for d in borderline] == [("c", "d")]


# ---------------------------------------------------------------------------
# Contested notes (spec-hermes-coherence §1): analytics triage surfaces
# contested frontmatter so contradictions never silently harden into fact.
# ---------------------------------------------------------------------------

CONTESTED_NOTE = """---
AI: true
tags:
  - farmacologia
last modified: 2026, 07, 02
related:
  - "[[B]]"
contested: true
contradictions:
  - "fonte: appunti.md"
---

# A

corpo con [[B]]
"""

PLAIN_NOTE = """---
AI: true
tags:
  - t
last modified: 2026, 07, 02
related:
  - "[[A]]"
---

# B

corpo
"""


def test_contested_notes_surface_in_analytics_report(tmp_vault):
    tmp_vault.note("A.md", CONTESTED_NOTE)
    tmp_vault.note("B.md", PLAIN_NOTE)
    nodes = [_make_node("A", "A", group=0), _make_node("B", "B", group=0)]
    edges = [_make_edge("e0", "A", "B")]

    r = compute_report(_nodes_edges_override=(nodes, edges), analytics=True)

    assert [c.path for c in r.contested] == ["A"]
    assert r.contested[0].refs == ["fonte: appunti.md"]
    assert r.totals["contested"] == 1


def test_contested_skipped_without_analytics(tmp_vault):
    tmp_vault.note("A.md", CONTESTED_NOTE)
    nodes = [_make_node("A", "A", group=0)]
    r = compute_report(_nodes_edges_override=(nodes, []))
    assert r.contested == []


def test_contested_section_rendered(tmp_vault):
    tmp_vault.note("A.md", CONTESTED_NOTE)
    tmp_vault.note("B.md", PLAIN_NOTE)
    nodes = [_make_node("A", "A", group=0), _make_node("B", "B", group=0)]
    edges = [_make_edge("e0", "A", "B")]
    r = compute_report(_nodes_edges_override=(nodes, edges), analytics=True)

    md = to_markdown(r)
    assert "Contested" in md and "appunti.md" in md
    digest = to_digest(r)
    assert "contested" in digest.lower()


# ---------------------------------------------------------------------------
# Source drift (spec-hermes-coherence §3): note<->source drift via sha256
# provenance records. Embedder-free, pure read of .silica/provenance.json —
# no note bodies touched, unlike Contested above.
# ---------------------------------------------------------------------------

def test_source_drift_acceptance_v2_touching_half_drifts_the_other_half(tmp_vault):
    """Ingest v1 (A,B) -> modify source -> re-ingest v2 (A only) -> graph_report
    lists B as drifted from lezione-03.md."""
    from silica.kernel.provenance import append_record

    append_record("lezione-03.md", "sha-v1", "run1", ["A", "B"])
    append_record("lezione-03.md", "sha-v2", "run2", ["A"])

    nodes = [_make_node("A", "A", group=0), _make_node("B", "B", group=0)]
    r = compute_report(_nodes_edges_override=(nodes, []), analytics=True)

    assert [(d.note, d.source) for d in r.source_drift] == [("B", "lezione-03.md")]
    assert r.totals["source_drift"] == 1


def test_source_drift_empty_without_provenance_file(tmp_vault):
    """No .silica/provenance.json -> no drift, nothing fails (additive)."""
    nodes = [_make_node("A", "A", group=0)]
    r = compute_report(_nodes_edges_override=(nodes, []), analytics=True)
    assert r.source_drift == []
    assert r.totals["source_drift"] == 0


def test_source_drift_skipped_without_analytics(tmp_vault):
    from silica.kernel.provenance import append_record

    append_record("a.md", "sha1", "run1", ["A", "B"])
    append_record("a.md", "sha2", "run2", ["A"])

    nodes = [_make_node("A", "A", group=0), _make_node("B", "B", group=0)]
    r = compute_report(_nodes_edges_override=(nodes, []))
    assert r.source_drift == []


def test_source_drift_section_rendered(tmp_vault):
    from silica.kernel.provenance import append_record

    append_record("lezione-03.md", "sha-v1", "run1", ["A", "B"])
    append_record("lezione-03.md", "sha-v2", "run2", ["A"])

    nodes = [_make_node("A", "A", group=0), _make_node("B", "B", group=0)]
    r = compute_report(_nodes_edges_override=(nodes, []), analytics=True)

    md = to_markdown(r)
    assert "Source Drift" in md
    assert "lezione-03.md" in md
    assert "[[B]]" in md
    digest = to_digest(r)
    assert "drift" in digest.lower()


def test_source_drift_no_section_when_empty(tmp_vault):
    nodes = [_make_node("A", "A", group=0)]
    r = compute_report(_nodes_edges_override=(nodes, []), analytics=True)
    md = to_markdown(r)
    assert "Source Drift" not in md


def test_source_drift_matches_despite_md_suffix_on_node_ids(tmp_vault):
    """Regression: real vault node ids carry `.md` (driver index keys) while
    provenance notes are recorded WITHOUT it (RunManifestEntry.path strips
    the suffix). The id-form mismatch must not swallow the intersection —
    drift has to surface even when the graph node id is `Concepts/A.md` and
    the provenance note is `Concepts/A`."""
    from silica.kernel.provenance import append_record

    append_record("lezione-03.md", "sha-v1", "run1", ["Concepts/A", "Concepts/B"])
    append_record("lezione-03.md", "sha-v2", "run2", ["Concepts/A"])

    nodes = [
        _make_node("Concepts/A.md", "A", group=0),
        _make_node("Concepts/B.md", "B", group=0),
    ]
    r = compute_report(_nodes_edges_override=(nodes, []), analytics=True)

    assert [(d.note, d.source) for d in r.source_drift] == [("Concepts/B", "lezione-03.md")]
    assert r.totals["source_drift"] == 1
