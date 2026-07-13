"""Tests for the co-occurrence vs wikilink DELTA report (graph_report.py).

The delta is the "hidden advantage": comparing the deterministic co-occurrence
graph against the authoritative wikilink graph yields an autonomous work plan,
computed with zero embedding tokens.

  - co-occurrence − wikilink  -> AUTOLINK candidates (related in text, unlinked)
  - wikilink − co-occurrence   -> STALE links (linked, but no textual co-presence)
  - high cooccur centrality + no hub note -> MISSING HUB (next note to create)
"""
from __future__ import annotations

import pytest

from silica.kernel.cooccurrence import CooccurStore, build_contribution
from silica.kernel.graph_report import (
    AutolinkCandidate,
    IntegrationDeficit,
    MissingHub,
    StaleLink,
    VaultReport,
    compute_report,
    to_markdown,
    _compute_cooccur_delta,
)


# ---------------------------------------------------------------------------
# Synthetic vault: wikilink structure + matching co-occurrence corpus.
#
# Wikilinks (EXTRACTED):  A↔B, B↔C, A↔C, D↔E, C→D ;  F orphan (ghost)
# Co-occurrence corpus (note body text, keyed by the SAME ids):
#   A: neural network        E: neural network    -> A,E share concepts, UNLINKED (3 hops)
#   B: beta cooking          C: beta cooking       -> B,C share concepts, and ARE linked
#   D: sailing boat          F: isolated
# So:
#   AUTOLINK: A–E (related in text, not wikilinked)
#   STALE:    A–B, A–C, C–D, D–E (linked, no shared concepts); NOT B–C (shared)
#   MISSING HUB: "neural"/"network"/"cooking" are central but have no note titled
#                so; "beta" is NOT missing (note B is titled "Beta").
# ---------------------------------------------------------------------------


def _make_node(nid, label, group, note_type="note"):
    return {"id": nid, "label": label, "group": group, "type": note_type}


def _make_edge(eid, src, dst, edge_type="EXTRACTED"):
    return {"id": eid, "from": src, "to": dst, "type": edge_type}


@pytest.fixture()
def synthetic_graph():
    nodes = [
        _make_node("A", "Alpha",   0),
        _make_node("B", "Beta",    0),
        _make_node("C", "Gamma",   0),
        _make_node("D", "Delta",   1),
        _make_node("E", "Epsilon", 1),
        _make_node("F", "Phi",    -1),
        {"id": "__unresolved__Ghost", "label": "Ghost", "group": -1, "type": "ghost"},
    ]
    edges = [
        _make_edge("e0", "A", "B"),
        _make_edge("e1", "B", "C"),
        _make_edge("e2", "A", "C"),
        _make_edge("e3", "D", "E"),
        _make_edge("e4", "C", "D"),
        _make_edge("e5", "F", "__unresolved__Ghost", "AMBIGUOUS"),
    ]
    return nodes, edges


@pytest.fixture()
def cooccur_store(tmp_path):
    st = CooccurStore(path=tmp_path / "c.json", lang="english")
    st.upsert_note("A", build_contribution("A", "neural network architecture"))
    st.upsert_note("B", build_contribution("B", "beta cooking pasta"))
    st.upsert_note("C", build_contribution("C", "beta cooking pizza"))
    st.upsert_note("D", build_contribution("D", "sailing boat harbour"))
    st.upsert_note("E", build_contribution("E", "neural network training"))
    st.upsert_note("F", build_contribution("F", "isolated lonely topic"))
    return st


@pytest.fixture()
def delta_report(synthetic_graph, cooccur_store):
    nodes, edges = synthetic_graph
    return compute_report(
        _nodes_edges_override=(nodes, edges),
        with_cooccurrence=True,
        _cooccur_store_override=cooccur_store,
    )


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

def test_delta_absent_by_default(synthetic_graph, cooccur_store):
    nodes, edges = synthetic_graph
    r = compute_report(_nodes_edges_override=(nodes, edges))
    assert r.autolink_candidates == []
    assert r.stale_links == []
    assert r.missing_hubs == []


def test_delta_empty_store_no_exception(synthetic_graph, tmp_path):
    nodes, edges = synthetic_graph
    empty = CooccurStore(path=tmp_path / "empty.json")
    r = compute_report(
        _nodes_edges_override=(nodes, edges),
        with_cooccurrence=True,
        _cooccur_store_override=empty,
    )
    assert r.autolink_candidates == []
    assert r.stale_links == []
    assert r.missing_hubs == []


# ---------------------------------------------------------------------------
# co-occurrence − wikilink  ->  AUTOLINK candidates
# ---------------------------------------------------------------------------

def test_autolink_proposes_unlinked_text_related_pair(delta_report):
    pairs = {(a.source, a.target) for a in delta_report.autolink_candidates}
    assert ("A", "E") in pairs or ("E", "A") in pairs


def test_autolink_carries_shared_concept_evidence(delta_report):
    cand = next(
        a for a in delta_report.autolink_candidates
        if {a.source, a.target} == {"A", "E"}
    )
    assert any("neural" in s or "network" in s for s in cand.shared)
    assert cand.weight > 0


# --- CORRELATE (ADR-0013): direct note_edges provenance + IDF evidence ------

def test_autolink_provenance_is_expanded_without_edges(delta_report):
    # No note_edges built on the fixture store -> today's behaviour: the pair
    # surfaces only through the expanded ranking.
    cand = next(a for a in delta_report.autolink_candidates if {a.source, a.target} == {"A", "E"})
    assert cand.provenance == "expanded"


def test_autolink_provenance_is_direct_when_edges_present(synthetic_graph, cooccur_store):
    from silica.kernel.correlate import recompute_all_edges
    recompute_all_edges(cooccur_store)  # A-E is a direct edge (jaccard 0.5 >= tau)
    nodes, edges = synthetic_graph
    r = compute_report(
        _nodes_edges_override=(nodes, edges),
        with_cooccurrence=True,
        _cooccur_store_override=cooccur_store,
    )
    cand = next(a for a in r.autolink_candidates if {a.source, a.target} == {"A", "E"})
    assert cand.provenance == "direct"


def test_autolink_bridges_md_suffixed_graph_ids(cooccur_store):
    # Real vaults: graph node ids carry '.md' while store keys are stripped.
    # The delta must bridge the two keyspaces — a raw `nid in G_und` matches
    # nothing and AUTOLINK silently comes out empty (regression 2026-07-10).
    from silica.kernel.correlate import recompute_all_edges
    recompute_all_edges(cooccur_store)
    nodes = [_make_node(f"{x}.md", x, 0) for x in ("A", "B", "C", "D", "E", "F")]
    edges = [
        _make_edge("e0", "A.md", "B.md"),
        _make_edge("e1", "B.md", "C.md"),
        _make_edge("e2", "A.md", "C.md"),
        _make_edge("e3", "D.md", "E.md"),
        _make_edge("e4", "C.md", "D.md"),
    ]
    r = compute_report(
        _nodes_edges_override=(nodes, edges),
        with_cooccurrence=True,
        _cooccur_store_override=cooccur_store,
    )
    cand = next(a for a in r.autolink_candidates if {a.source, a.target} == {"A", "E"})
    # candidates keep STORE keys; the direct leg bridged too (A-E is an edge)
    assert cand.provenance == "direct"


def test_direct_evidence_orders_shared_stems_by_idf(tmp_path):
    from silica.kernel.correlate import recompute_all_edges
    st = CooccurStore(path=tmp_path / "idf.json", lang="english")
    # "common" is ubiquitous (low IDF); "rare"/"scarce" live only in P,Q (high IDF).
    st.upsert_note("P", build_contribution("P", "common rare scarce alpha"))
    st.upsert_note("Q", build_contribution("Q", "common rare scarce beta"))
    st.upsert_note("R", build_contribution("R", "common ordinary usual"))
    st.upsert_note("S", build_contribution("S", "common typical routine"))
    recompute_all_edges(st)  # P-Q share {common,rare,scarce}: jaccard 3/5, a direct edge
    # P and Q are isolated nodes (no wikilink path) -> a valid >2-hop candidate.
    nodes = [_make_node(x, x, 0) for x in ("P", "Q", "R", "S")]
    r = compute_report(
        _nodes_edges_override=(nodes, []),
        with_cooccurrence=True,
        _cooccur_store_override=st,
    )
    cand = next(a for a in r.autolink_candidates if {a.source, a.target} == {"P", "Q"})
    assert cand.provenance == "direct"
    # discriminative (high-IDF) stems precede the ubiquitous one; top 5 only.
    assert cand.shared.index("common") == len(cand.shared) - 1
    assert "rare" in cand.shared and "scarce" in cand.shared
    assert len(cand.shared) <= 5


def test_autolink_quota_keeps_direct_above_the_expanded_flood(tmp_path):
    # Direct weights are Jaccard (<=1); expanded overlap weights run orders of
    # magnitude higher. Without the per-leg quota one mixed sort floods the
    # top-k with expanded pairs and the direct leg never renders.
    from silica.kernel.correlate import recompute_all_edges
    st = CooccurStore(path=tmp_path / "quota.json", lang="english")
    st.upsert_note("P", build_contribution("P", "rare scarce alpha beta"))
    st.upsert_note("Q", build_contribution("Q", "rare scarce alpha gamma"))
    hub = "common " * 10  # a heavy shared stem -> big expanded overlaps
    st.upsert_note("R", build_contribution("R", hub + "red rope rust rain rock"))
    st.upsert_note("S", build_contribution("S", hub + "sand salt song sail seed"))
    st.upsert_note("T", build_contribution("T", hub + "tree tent tide turf twig"))
    recompute_all_edges(st)  # P-Q is the ONLY direct edge (jaccard 0.6)
    nodes = [_make_node(x, x, 0) for x in ("P", "Q", "R", "S", "T")]
    r = compute_report(
        _nodes_edges_override=(nodes, []),
        with_cooccurrence=True,
        _cooccur_store_override=st,
        top_k=2,
    )
    provs = [a.provenance for a in r.autolink_candidates]
    assert "direct" in provs      # the Jaccard-scaled leg holds its slot
    assert "expanded" in provs    # without starving the high-recall leg


def test_autolink_excludes_already_wikilinked_pairs(delta_report):
    pairs = {frozenset((a.source, a.target)) for a in delta_report.autolink_candidates}
    # B and C share concepts but are already wikilinked -> never an autolink
    assert frozenset(("B", "C")) not in pairs


# ---------------------------------------------------------------------------
# wikilink − co-occurrence  ->  STALE links
# ---------------------------------------------------------------------------

def test_stale_flags_wikilink_without_shared_concepts(delta_report):
    pairs = {frozenset((s.source, s.target)) for s in delta_report.stale_links}
    assert frozenset(("A", "B")) in pairs   # linked, but neural vs cooking: no overlap


def test_stale_excludes_wikilink_with_shared_concepts(delta_report):
    pairs = {frozenset((s.source, s.target)) for s in delta_report.stale_links}
    assert frozenset(("B", "C")) not in pairs   # linked AND share "beta cooking"


# ---------------------------------------------------------------------------
# high cooccur centrality + no hub note  ->  MISSING HUB
# ---------------------------------------------------------------------------

def test_missing_hub_surfaces_central_unhubbed_concept(delta_report):
    concepts = {h.concept for h in delta_report.missing_hubs}
    assert any(c in concepts for c in ("neural", "network"))


def test_missing_hub_excludes_concept_with_a_titled_note(delta_report):
    # note B is titled "Beta", so the concept "beta" is already formalised
    concepts = {h.concept for h in delta_report.missing_hubs}
    assert "beta" not in concepts


def test_missing_hubs_sorted_by_centrality_desc(delta_report):
    cents = [h.centrality for h in delta_report.missing_hubs]
    assert cents == sorted(cents, reverse=True)


# ---------------------------------------------------------------------------
# Unit: _compute_cooccur_delta is injectable and returns four lists
# ---------------------------------------------------------------------------

def test_compute_cooccur_delta_returns_four_lists(synthetic_graph, cooccur_store):
    import networkx as nx
    nodes, edges = synthetic_graph
    real_ids = {n["id"] for n in nodes if n.get("type") != "ghost"}
    G = nx.Graph()
    G.add_nodes_from(real_ids)
    for e in edges:
        if e["type"] == "EXTRACTED":
            G.add_edge(e["from"], e["to"])
    node_label = {n["id"]: n["label"] for n in nodes if n.get("type") != "ghost"}
    report = VaultReport(
        generated_at="", scope="", totals={}, god_nodes=[], bridges=[],
        orphans=[], dangling=[], clusters=[],
    )
    al, sl, mh, idf = _compute_cooccur_delta(
        report, G, node_label, cooccur_store=cooccur_store, k=10
    )
    assert all(isinstance(x, AutolinkCandidate) for x in al)
    assert all(isinstance(x, StaleLink) for x in sl)
    assert all(isinstance(x, MissingHub) for x in mh)
    assert all(isinstance(x, IntegrationDeficit) for x in idf)


# ---------------------------------------------------------------------------
# Output / totals
# ---------------------------------------------------------------------------

def test_totals_include_delta_counts(delta_report):
    assert "autolink_candidates" in delta_report.totals
    assert "stale_links" in delta_report.totals
    assert "missing_hubs" in delta_report.totals
    assert "integration_deficits" in delta_report.totals


# ---------------------------------------------------------------------------
# INTEGRATION DEFICIT: concept-rich note, weakly wikilinked
#
# F carries 3 concepts ("isolated lonely topic") but has ZERO resolved
# wikilinks (its only edge is AMBIGUOUS) -> highest divergence, ranks first.
# A carries the same 3 concepts but 2 wikilinks -> lower score.
# ---------------------------------------------------------------------------

def test_integration_deficit_ranks_unlinked_rich_note_first(delta_report):
    deficits = delta_report.integration_deficits
    assert deficits, "expected a non-empty integration-deficit ranking"
    top = deficits[0]
    assert top.path == "F"
    assert top.degree == 0
    # score is the pure ranking formula, no hidden weights
    for d in deficits:
        assert d.score == pytest.approx(d.concepts / (1 + d.degree), abs=1e-3)
    # descending order
    scores = [d.score for d in deficits]
    assert scores == sorted(scores, reverse=True)


def test_markdown_renders_integration_deficit_section(delta_report):
    md = to_markdown(delta_report)
    assert "Integration Deficit" in md


def test_markdown_renders_delta_sections(delta_report):
    md = to_markdown(delta_report)
    assert "Autolink" in md
    assert "Stale" in md
    assert "Hub" in md  # "Missing Hubs" section header


def test_markdown_omits_delta_sections_when_empty(synthetic_graph):
    nodes, edges = synthetic_graph
    r = compute_report(_nodes_edges_override=(nodes, edges))
    md = to_markdown(r)
    assert "Autolink Candidates" not in md
    assert "Stale Links" not in md


# ---------------------------------------------------------------------------
# Tool surface: the delta is reachable via silica_vault_report
# ---------------------------------------------------------------------------

def test_vault_report_tool_exposes_with_cooccurrence_flag():
    from silica.tools.composed import VaultReportArgs
    args = VaultReportArgs()
    assert args.with_cooccurrence is False  # default off, opt-in like with_embeddings


def test_delta_report_json_serializable(delta_report, tmp_path):
    import dataclasses
    import orjson
    from silica.kernel.graph_report import write_report
    paths = write_report(delta_report, str(tmp_path / "GRAPH_REPORT.md"))
    data = orjson.loads((tmp_path / "GRAPH_REPORT.json").read_bytes())
    # nested delta dataclasses survive the asdict -> orjson round-trip
    assert "autolink_candidates" in data
    assert isinstance(data["autolink_candidates"], list)
    assert dataclasses.asdict(delta_report)["stale_links"] == data["stale_links"]


# ---------------------------------------------------------------------------
# #8 cross-concept convergence  ->  S_(many_own)×other
#
# Paper (Marwitz et al. 2026, Table 2): the report section linking a candidate
# to MANY of the researcher's own concepts had the highest "interesting" rate
# (61.5%). Silica equivalent: an autolink candidate touching more god-node hubs
# earns a higher `convergence` and must rank ahead of an equally-weighted but
# single-hub candidate.
# ---------------------------------------------------------------------------

@pytest.fixture()
def convergence_store(tmp_path):
    st = CooccurStore(path=tmp_path / "conv.json", lang="english")
    # Three hubs each carry a distinct concept PLUS a common "shared" concept.
    st.upsert_note("H1", build_contribution("H1", "alpha shared"))
    st.upsert_note("H2", build_contribution("H2", "beta shared"))
    st.upsert_note("H3", build_contribution("H3", "gamma shared"))
    # X co-mentions "shared" -> co-occurs with ALL three hubs (convergence 3).
    st.upsert_note("X", build_contribution("X", "shared concept"))
    # Y co-mentions only "alpha" -> co-occurs with H1 alone (convergence 1).
    st.upsert_note("Y", build_contribution("Y", "alpha extra"))
    return st


def test_autolink_convergence_counts_hubs_and_drives_ranking(convergence_store):
    import networkx as nx
    from silica.kernel.graph_report import NodeStat, _compute_cooccur_delta

    ids = ["H1", "H2", "H3", "X", "Y"]
    G = nx.Graph()
    G.add_nodes_from(ids)
    G.add_edges_from([("H1", "H2"), ("H2", "H3")])  # hubs linked; X, Y isolated
    node_label = {i: i for i in ids}

    report = VaultReport(
        generated_at="", scope="", totals={},
        god_nodes=[
            NodeStat(id="H1", label="H1", cluster=0, out_degree=1, in_degree=1, degree=2, pagerank=0.0),
            NodeStat(id="H2", label="H2", cluster=0, out_degree=2, in_degree=2, degree=4, pagerank=0.0),
            NodeStat(id="H3", label="H3", cluster=0, out_degree=1, in_degree=1, degree=2, pagerank=0.0),
        ],
        bridges=[], orphans=[], dangling=[], clusters=[],
    )

    al, _sl, _mh, _idf = _compute_cooccur_delta(
        report, G, node_label, cooccur_store=convergence_store, k=20
    )

    x_cand = next(c for c in al if "X" in (c.source, c.target))
    y_cand = next(c for c in al if "Y" in (c.source, c.target))

    # X bridges into all three hub neighbourhoods; Y only into H1's.
    assert x_cand.convergence == 3
    assert y_cand.convergence == 1

    # Convergence drives ranking: the multi-hub candidate sorts ahead.
    x_idx = min(i for i, c in enumerate(al) if "X" in (c.source, c.target))
    y_idx = next(i for i, c in enumerate(al) if "Y" in (c.source, c.target))
    assert x_idx < y_idx


def test_markdown_autolink_table_shows_convergence():
    """The autolink table surfaces the #8 convergence (hub reach) column."""
    r = VaultReport(
        generated_at="", scope="", totals={}, god_nodes=[], bridges=[],
        orphans=[], dangling=[], clusters=[],
    )
    r.autolink_candidates = [
        AutolinkCandidate(source="A", target="B", weight=3.0, shared=["x"], convergence=2),
    ]
    md = to_markdown(r)
    assert "Hubs" in md          # convergence column header
    assert "| 2 |" in md         # the convergence value rendered
    assert "Via" in md           # provenance column header
    assert "expanded" in md      # default provenance rendered
