"""Tests for the relatedness facade (kernel/relatedness.py).

The facade fuses two PROPOSE-layers into a single note-level ranking:
  - embeddings (EmbedStore.cosine_top_k)  — semantic similarity
  - co-occurrence (CooccurStore + inverted index) — associative reach
via Reciprocal Rank Fusion, with degenerate proponents abstaining so the
survivor's ranking passes through unchanged ("embedder down -> cooccur routing").
"""
from __future__ import annotations


from silica.kernel.embed import EmbedStore
from silica.kernel.cooccurrence import CooccurStore, build_contribution


# ---------------------------------------------------------------------------
# RRF fusion (pure)
# ---------------------------------------------------------------------------

from silica.kernel.relatedness import _rrf_fuse, RRF_K


def test_rrf_fuse_single_ranking_orders_by_rank():
    fused = _rrf_fuse([[("A", 9.0), ("B", 4.0), ("C", 1.0)]])
    # earlier rank -> higher RRF contribution
    assert fused["A"] > fused["B"] > fused["C"]


def test_rrf_fuse_rewards_agreement_across_rankings():
    # X is rank-2 in both lists; Y is rank-1 in one and absent in the other.
    embed = [("Y", 0.9), ("X", 0.8), ("Z", 0.1)]
    cooc = [("W", 50.0), ("X", 30.0), ("Q", 1.0)]
    fused = _rrf_fuse([embed, cooc])
    # X appears in both -> accumulates two reciprocal-rank terms -> beats
    # single-list leaders Y and W.
    assert fused["X"] > fused["Y"]
    assert fused["X"] > fused["W"]


def test_rrf_fuse_uses_standard_damping_constant():
    fused = _rrf_fuse([[("A", 1.0)]])
    assert fused["A"] == 1.0 / (RRF_K + 1)


def test_rrf_fuse_empty_is_empty():
    assert _rrf_fuse([]) == {}
    assert _rrf_fuse([[]]) == {}


# --- CORRELATE (ADR-0013): third fusion leg from note_edges -----------------

def test_fuse_includes_edges_leg():
    from silica.kernel.relatedness import _fuse
    out = _fuse(None, None, edges_rank=[("B", 0.31)], k=5)
    assert out and out[0].path == "B"
    assert out[0].edge_score == 0.31
    assert "edge:0.31" in out[0].evidence


def test_fuse_drops_vault_root_artifacts():
    # A stale GRAPH_REPORT vector can outlive its index-build exclusion (the
    # store is upsert-only); _fuse must never surface it. "real" survives.
    from silica.kernel.relatedness import _fuse
    out = _fuse([("GRAPH_REPORT", "Graph Report", 0.99), ("real", "Real", 0.5)], None, k=5)
    assert [r.path for r in out] == ["real"]


# ---------------------------------------------------------------------------
# Embed leg + abstention
# ---------------------------------------------------------------------------

from silica.kernel.relatedness import _embed_ranking


def _embed_store(tmp_path) -> EmbedStore:
    es = EmbedStore(path=tmp_path / "e.json")
    es.upsert("A", "A note", [1.0, 0.0])
    es.upsert("B", "B note", [0.9, 0.1])   # close to A
    es.upsert("C", "C note", [0.0, 1.0])   # orthogonal to A
    return es


def test_embed_ranking_returns_path_name_score(tmp_path):
    es = _embed_store(tmp_path)
    ranking = _embed_ranking(es, "A", k=5, exclude={"A"})
    assert ranking is not None
    paths = [p for p, _n, _s in ranking]
    assert paths[0] == "B"            # nearest neighbour first
    assert ("B", "B note") == (ranking[0][0], ranking[0][1])


def test_embed_ranking_abstains_when_note_not_indexed(tmp_path):
    es = _embed_store(tmp_path)
    assert _embed_ranking(es, "DOES_NOT_EXIST", k=5, exclude=set()) is None


def test_embed_ranking_abstains_on_degenerate_all_zero_scores(tmp_path):
    es = EmbedStore(path=tmp_path / "e.json")
    es.upsert("Z", "Z", [0.0, 0.0])   # zero query vector -> every score 0.0
    es.upsert("B", "B", [1.0, 0.0])
    # degenerate output must abstain, NOT return a flat zero ranking (poison for RRF)
    assert _embed_ranking(es, "Z", k=5, exclude={"Z"}) is None


def test_embed_ranking_handles_md_suffixed_query(tmp_path):
    es = _embed_store(tmp_path)
    # graph_report-style callers may pass paths with a trailing .md
    ranking = _embed_ranking(es, "A.md", k=5, exclude={"A"})
    assert ranking is not None
    assert ranking[0][0] == "B"


# ---------------------------------------------------------------------------
# Co-occurrence leg + abstention
# ---------------------------------------------------------------------------

from silica.kernel.relatedness import _cooccur_ranking


def _cooc_store(tmp_path) -> CooccurStore:
    st = CooccurStore(path=tmp_path / "c.json", lang="english")
    st.upsert_note("A", build_contribution("A", "alpha beta gamma"))
    st.upsert_note("B", build_contribution("B", "beta gamma delta"))  # shares beta, gamma
    st.upsert_note("C", build_contribution("C", "zeta eta theta"))    # disjoint
    return st


def test_cooccur_ranking_ranks_notes_sharing_concepts(tmp_path):
    st = _cooc_store(tmp_path)
    ranking = _cooccur_ranking(st, "A", k=5, exclude=set(), scope=None, expand=False)
    assert ranking is not None
    paths = [p for p, _w in ranking]
    assert paths[0] == "B"      # shares two concepts with A
    assert "C" not in paths     # shares nothing -> not a candidate


def test_cooccur_ranking_excludes_query_and_exclude_set(tmp_path):
    st = _cooc_store(tmp_path)
    ranking = _cooccur_ranking(st, "A", k=5, exclude={"B"}, scope=None, expand=False)
    paths = [p for p, _w in ranking or []]
    assert "A" not in paths     # never returns the query itself
    assert "B" not in paths     # honours the exclude set


def test_cooccur_ranking_abstains_when_query_absent(tmp_path):
    st = _cooc_store(tmp_path)
    assert _cooccur_ranking(st, "UNKNOWN", k=5, exclude=set(), scope=None) is None


def test_cooccur_ranking_idf_beats_hub_over_rare_match(tmp_path):
    # 'hub' appears in every note (zero discriminating power); 'rare' is shared
    # by only the query and TWIN. IDF must rank the rare-sharing TWIN above a
    # note that merely piles on the ubiquitous hub concept.
    st = CooccurStore(path=tmp_path / "c.json", lang="english")
    st.upsert_note("Q",    build_contribution("Q", "hub rare"))
    st.upsert_note("TWIN", build_contribution("TWIN", "hub rare"))       # shares the rare concept
    st.upsert_note("HUB1", build_contribution("HUB1", "hub hub hub"))    # only the ubiquitous hub
    st.upsert_note("HUB2", build_contribution("HUB2", "hub filler"))     # keeps 'hub' near-ubiquitous

    ranking = _cooccur_ranking(st, "Q", k=5, exclude=set(), scope=None, expand=False)
    paths = [p for p, _w in ranking or []]
    assert paths and paths[0] == "TWIN"   # rare shared concept wins over hub breadth


def test_cooccur_ranking_expansion_reaches_associative_notes(tmp_path):
    # A is about alpha. Elsewhere alpha co-occurs strongly with omega.
    # A note about omega (but not alpha) is associatively related ONLY via expansion.
    st = CooccurStore(path=tmp_path / "c.json", lang="english")
    st.upsert_note("A", build_contribution("A", "alpha alpha"))
    st.upsert_note("BRIDGE", build_contribution("BRIDGE", "alpha omega"))  # links alpha<->omega
    st.upsert_note("OMEGA", build_contribution("OMEGA", "omega omega"))    # no alpha at all

    direct = _cooccur_ranking(st, "A", k=5, exclude={"BRIDGE"}, scope=None, expand=False)
    expanded = _cooccur_ranking(st, "A", k=5, exclude={"BRIDGE"}, scope=None, expand=True)

    direct_paths = [p for p, _w in direct or []]
    expanded_paths = [p for p, _w in expanded or []]
    assert "OMEGA" not in direct_paths      # no shared concept without expansion
    assert "OMEGA" in expanded_paths        # reached via alpha->omega neighbour edge


# ---------------------------------------------------------------------------
# Facade integration: related_notes
# ---------------------------------------------------------------------------

from silica.kernel.relatedness import related_notes, RelatedNote


def test_related_notes_fuses_both_legs_with_evidence(tmp_path):
    es = _embed_store(tmp_path)
    st = _cooc_store(tmp_path)
    out = related_notes("A", embed_store=es, cooccur_store=st, k=5)
    assert out and isinstance(out[0], RelatedNote)
    by_path = {r.path: r for r in out}
    # B is both A's nearest embed neighbour AND its strongest cooccur overlap
    assert "B" in by_path
    ev = by_path["B"].evidence
    assert any(e.startswith("embed:") for e in ev)
    assert any(e.startswith("cooccur:") for e in ev)


def test_related_notes_embedder_down_routes_on_cooccurrence(tmp_path):
    # No embed store at all -> embed leg abstains -> pure cooccurrence ranking.
    st = _cooc_store(tmp_path)
    out = related_notes("A", embed_store=None, cooccur_store=st, k=5)
    paths = [r.path for r in out]
    assert paths and paths[0] == "B"
    # provenance is cooccur-only when the embedder is down
    assert all(e.startswith("cooccur:") for r in out for e in r.evidence)


def test_related_notes_cooccur_empty_routes_on_embeddings(tmp_path):
    es = _embed_store(tmp_path)
    out = related_notes("A", embed_store=es, cooccur_store=None, k=5)
    paths = [r.path for r in out]
    assert paths and paths[0] == "B"
    assert all(e.startswith("embed:") for r in out for e in r.evidence)


def test_related_notes_both_abstain_returns_empty(tmp_path):
    out = related_notes("A", embed_store=None, cooccur_store=None, k=5)
    assert out == []


def test_related_notes_respects_k(tmp_path):
    es = _embed_store(tmp_path)
    st = _cooc_store(tmp_path)
    out = related_notes("A", embed_store=es, cooccur_store=st, k=1)
    assert len(out) <= 1


def test_related_notes_never_returns_the_query(tmp_path):
    es = _embed_store(tmp_path)
    st = _cooc_store(tmp_path)
    out = related_notes("A", embed_store=es, cooccur_store=st, k=10)
    assert "A" not in [r.path for r in out]


def test_related_notes_evidence_score_formats(tmp_path):
    es = _embed_store(tmp_path)
    st = _cooc_store(tmp_path)
    out = related_notes("A", embed_store=es, cooccur_store=st, k=5)
    ev_all = [e for r in out for e in r.evidence]
    # embed evidence carries a 2-decimal cosine; cooccur carries an integer weight
    assert any(e.startswith("embed:0.") or e.startswith("embed:1.") for e in ev_all)
    assert any(e.startswith("cooccur:w") for e in ev_all)


def test_related_notes_direct_edge_leg_carries_edge_score(tmp_path):
    from silica.kernel.correlate import recompute_all_edges
    st = CooccurStore(path=tmp_path / "c.json", lang="english")
    st.upsert_note("A", build_contribution("A", "alpha beta gamma"))
    st.upsert_note("B", build_contribution("B", "alpha beta delta"))  # jaccard 0.5 -> edge
    recompute_all_edges(st)
    out = related_notes("A", embed_store=None, cooccur_store=st, k=5)
    b = next(r for r in out if r.path == "B")
    assert b.edge_score is not None
    assert any(e.startswith("edge:") for e in b.evidence)


def test_related_notes_edge_leg_abstains_without_edges(tmp_path):
    st = _cooc_store(tmp_path)  # contributions only, no note_edges built
    out = related_notes("A", embed_store=None, cooccur_store=st, k=5)
    assert out  # cooccur leg still carries the fusion
    assert all(r.edge_score is None for r in out)


def test_related_note_exposes_structured_per_leg_scores(tmp_path):
    es = _embed_store(tmp_path)
    st = _cooc_store(tmp_path)
    out = related_notes("A", embed_store=es, cooccur_store=st, k=5)
    b = next(r for r in out if r.path == "B")
    # raw signals are accessible without parsing the evidence strings
    assert b.embed_score is not None and b.embed_score > 0.9
    assert b.cooccur_weight is not None and b.cooccur_weight > 0


# ---------------------------------------------------------------------------
# Fresh-query facade: related_notes_for_query (vec + text, no indexed path)
# ---------------------------------------------------------------------------

from silica.kernel.relatedness import related_notes_for_query


def test_for_query_embed_only_ranks_by_vector(tmp_path):
    es = _embed_store(tmp_path)
    out = related_notes_for_query(query_vec=[0.9, 0.1], embed_store=es, k=5)
    assert out and out[0].path == "B"          # nearest to the query vector
    assert out[0].embed_score is not None and out[0].cooccur_weight is None
    assert all(e.startswith("embed:") for r in out for e in r.evidence)


def test_for_query_cooccur_only_from_text(tmp_path):
    st = _cooc_store(tmp_path)                  # A:alpha beta gamma, B:beta gamma delta
    out = related_notes_for_query(query_text="alpha beta gamma", cooccur_store=st, k=5)
    paths = [r.path for r in out]
    assert "A" in paths and "B" in paths        # both share concepts with the text
    assert all(r.embed_score is None for r in out)
    assert any(r.cooccur_weight for r in out)


def test_for_query_fuses_vec_and_text(tmp_path):
    es = _embed_store(tmp_path)
    st = _cooc_store(tmp_path)
    out = related_notes_for_query(
        query_vec=es.get_vec("A"), query_text="alpha beta gamma",
        embed_store=es, cooccur_store=st, k=5, exclude={"A"},
    )
    b = next(r for r in out if r.path == "B")
    assert b.embed_score is not None and b.cooccur_weight is not None


def test_for_query_degenerate_vector_abstains_cooccur_carries(tmp_path):
    es = _embed_store(tmp_path)
    st = _cooc_store(tmp_path)
    out = related_notes_for_query(
        query_vec=[0.0, 0.0], query_text="alpha beta gamma",
        embed_store=es, cooccur_store=st, k=5, exclude={"A"},
    )
    # zero query vector -> embed leg abstains rather than poisoning the fusion
    assert out
    assert all(r.embed_score is None for r in out)
    assert any(r.cooccur_weight for r in out)


def test_for_query_respects_exclude(tmp_path):
    st = _cooc_store(tmp_path)
    out = related_notes_for_query(query_text="alpha beta gamma", cooccur_store=st, k=5, exclude={"B"})
    assert "B" not in [r.path for r in out]


def test_for_query_never_has_edge_leg(tmp_path):
    # Structural abstention (ADR-0013 Q5): fresh query text has no note_edges
    # row, so the third leg NEVER fires here — even when the store has edges.
    from silica.kernel.correlate import recompute_all_edges
    st = _cooc_store(tmp_path)
    recompute_all_edges(st)
    out = related_notes_for_query(query_text="alpha beta gamma", cooccur_store=st, k=5)
    assert out
    assert all(r.edge_score is None for r in out)


def test_for_query_both_absent_returns_empty(tmp_path):
    assert related_notes_for_query(k=5) == []
    assert related_notes_for_query(query_text="alpha", k=5) == []        # no cooccur store
    assert related_notes_for_query(query_vec=[1.0, 0.0], k=5) == []      # no embed store


# ---------------------------------------------------------------------------
# Boundary / robustness contract
# ---------------------------------------------------------------------------

