"""Tests for the co-occurrence graph kernel (kernel/cooccurrence.py)."""
from __future__ import annotations

from pathlib import Path

from silica.config import SilicaConfig


def test_config_has_cooccurrence_lang_default_english():
    cfg = SilicaConfig()
    assert cfg.cooccurrence_lang == "english"


from silica.kernel.cooccurrence import tokenize, _split_sentences


def test_split_sentences_breaks_on_terminators():
    text = "Prima frase. Seconda frase! Terza?\nQuarta"
    assert _split_sentences(text) == ["Prima frase", "Seconda frase", "Terza", "Quarta"]


def test_tokenize_lowercases_and_drops_short_and_stopwords_english():
    # "the" is a stopword, "a" and "is" too; "of" stopword; "ai" is < 3 chars
    sents = tokenize("The cat is on a mat", lang="english")
    # one sentence, stopwords/short removed, remaining stemmed (cat, mat)
    stems = [stem for sent in sents for (stem, _surface) in sent]
    assert "cat" in stems
    assert "mat" in stems
    assert all(s not in stems for s in ("the", "is", "on", "a"))


def snow_stem_it(word: str) -> str:
    import snowballstemmer
    return snowballstemmer.stemmer("italian").stemWord(word)


def test_tokenize_collapses_italian_inflections():
    sents = tokenize("La rete e le reti neurali", lang="italian")
    stems = [stem for sent in sents for (stem, _surface) in sent]
    # rete and reti must collapse to the same stem
    assert stems.count(snow_stem_it("rete")) >= 1
    # both inflections map to one stem
    assert snow_stem_it("rete") == snow_stem_it("reti")


def test_tokenize_keeps_surface_form():
    sents = tokenize("Neural networks", lang="english")
    surfaces = [surface for sent in sents for (_stem, surface) in sent]
    assert "neural" in surfaces  # surface is lowercased original token


from silica.kernel.cooccurrence import build_contribution


def _edge_weight(contribution, a, b):
    """Sum directed edge weight a->b in a contribution's edge list."""
    return sum(w for (f, t, w) in contribution["edges"] if f == a and t == b)


def test_build_contribution_narrative_adjacent_weight_3():
    # four distinct content words, no stopwords, single sentence
    c = build_contribution("N", "alpha beta gamma delta", lang="english")
    st = __import__("snowballstemmer").stemmer("english").stemWord
    # adjacent pair alpha->beta has narrative weight 3
    assert _edge_weight(c, st("alpha"), st("beta")) == 3


def test_build_contribution_gap_scan_decays_3_2_1():
    c = build_contribution("N", "alpha beta gamma delta", lang="english")
    st = __import__("snowballstemmer").stemmer("english").stemWord
    a, b, g, d = st("alpha"), st("beta"), st("gamma"), st("delta")
    # delta links back: to gamma (dist1=3), beta (dist2=2), alpha (dist3=1)
    assert _edge_weight(c, g, d) == 3
    assert _edge_weight(c, b, d) == 2
    assert _edge_weight(c, a, d) == 1


def test_build_contribution_no_edge_across_sentence_boundary():
    c = build_contribution("N", "alpha beta. gamma delta", lang="english")
    st = __import__("snowballstemmer").stemmer("english").stemWord
    # beta (end of sentence 1) must NOT link to gamma (start of sentence 2)
    assert _edge_weight(c, st("beta"), st("gamma")) == 0


def test_build_contribution_nodes_have_label_and_count():
    c = build_contribution("N", "alpha alpha beta", lang="english")
    st = __import__("snowballstemmer").stemmer("english").stemWord
    assert c["nodes"][st("alpha")]["count"] == 2
    assert c["nodes"][st("alpha")]["label"] == "alpha"


from silica.kernel.cooccurrence import CooccurStore


def test_store_empty_on_missing_file(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    assert len(store) == 0


def test_store_upsert_and_len(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha beta gamma"))
    assert len(store) == 1
    assert "A" in store.paths()


def test_store_roundtrip(tmp_path):
    idx = tmp_path / "cooc.json"
    store = CooccurStore(path=idx, lang="english")
    store.upsert_note("A", build_contribution("A", "alpha beta gamma"))
    store.save()

    store2 = CooccurStore(path=idx)
    assert len(store2) == 1
    assert store2.lang == "english"


def test_store_delete_note(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha beta"))
    store.delete_note("A")
    assert len(store) == 0


def test_neighbors_returns_sorted_candidates(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    # gamma co-occurs strongly with beta (dist1) and weaker with alpha (dist2)
    store.upsert_note("A", build_contribution("A", "alpha beta gamma"))
    cands = store.neighbors("gamma", k=5)
    assert cands[0]["evidence"] == "cooccur"
    labels = [c["concept"] for c in cands]
    # beta (weight 3) ranks above alpha (weight 2)
    assert labels.index("beta") < labels.index("alpha")


def test_neighbors_undirected_sums_both_directions(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha beta"))   # alpha->beta w3
    store.upsert_note("B", build_contribution("B", "beta alpha"))   # beta->alpha w3
    cands = store.neighbors("alpha", k=5)
    beta = next(c for c in cands if c["concept"] == "beta")
    assert beta["weight"] == 6  # 3 + 3, undirected


def test_neighbors_respects_k(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha beta gamma delta epsilon"))
    assert len(store.neighbors("alpha", k=2)) <= 2


def test_neighbors_missing_concept_returns_empty(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha beta"))
    assert store.neighbors("nonexistentword", k=5) == []


def test_neighbors_empty_store_returns_empty(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    assert store.neighbors("alpha", k=5) == []


def test_note_nodes_returns_stem_counts_for_one_note(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha alpha beta"))
    st = __import__("snowballstemmer").stemmer("english").stemWord
    nodes = store.note_nodes("A")
    assert nodes[st("alpha")] == 2
    assert nodes[st("beta")] == 1


def test_note_nodes_missing_note_returns_empty(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    assert store.note_nodes("NOPE") == {}


def test_to_networkx_builds_weighted_undirected_graph(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("A", build_contribution("A", "alpha beta"))
    G = store.to_networkx()
    st = __import__("snowballstemmer").stemmer("english").stemWord
    assert G.has_edge(st("alpha"), st("beta"))
    assert G[st("alpha")][st("beta")]["weight"] == 3


def test_scope_restricts_aggregation(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    store.upsert_note("Robotica/A", build_contribution("A", "alpha beta"))
    store.upsert_note("Cucina/B", build_contribution("B", "gamma delta"))
    # within Robotica/, gamma has no neighbors
    assert store.neighbors("gamma", scope="Robotica") == []
    # but alpha does
    assert store.neighbors("alpha", scope="Robotica") != []


from silica.kernel.cooccurrence import build_index, refresh_note


def test_build_index_bulk(tmp_path):
    idx = tmp_path / "cooc.json"
    notes = [
        ("A", "A", "alpha beta gamma"),
        ("B", "B", "beta gamma delta"),
    ]
    store = build_index(notes, store=CooccurStore(path=idx, lang="english"))
    assert len(store) == 2
    assert idx.exists()


def test_refresh_note_replaces_contribution_no_inflation(tmp_path):
    idx = tmp_path / "cooc.json"
    store = CooccurStore(path=idx, lang="english")
    store.upsert_note("A", build_contribution("A", "alpha beta"))
    store.save()

    st = __import__("snowballstemmer").stemmer("english").stemWord
    before = store.neighbors("alpha", k=5)
    w_before = next(c["weight"] for c in before if c["concept"] == "beta")

    # refresh the SAME note with identical content — weight must NOT double
    refresh_note("A", "A", "alpha beta", store=store)
    after = store.neighbors("alpha", k=5)
    w_after = next(c["weight"] for c in after if c["concept"] == "beta")
    assert w_after == w_before  # replacement, not accumulation


def test_refresh_note_reflects_new_content(tmp_path):
    idx = tmp_path / "cooc.json"
    store = CooccurStore(path=idx, lang="english")
    store.upsert_note("A", build_contribution("A", "alpha beta"))
    refresh_note("A", "A", "alpha gamma", store=store)
    # beta no longer co-occurs with alpha; gamma now does
    labels = [c["concept"] for c in store.neighbors("alpha", k=5)]
    assert "gamma" in labels
    assert "beta" not in labels


import ast


def test_module_never_imports_embedder():
    """cooccurrence.py is the stable leg: it must not depend on the embedder
    or provider stack (works with LM Studio down)."""
    src = (Path(__file__).parent.parent / "silica" / "kernel" / "cooccurrence.py").read_text()
    tree = ast.parse(src)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
    assert not any("providers" in m for m in imported)
    assert not any("embed" in m for m in imported)


def test_neighbors_never_raises_on_garbage(tmp_path):
    store = CooccurStore(path=tmp_path / "cooc.json")
    # empty/garbage queries must return [] rather than raising
    assert store.neighbors("", k=5) == []
    assert store.neighbors("   ", k=5) == []


def test_corrupt_index_loads_empty(tmp_path):
    idx = tmp_path / "cooc.json"
    idx.write_text("{ this is not valid json ")
    store = CooccurStore(path=idx)
    assert len(store) == 0


# ---------------------------------------------------------------------------
# #9 LLM concept augmentation — concepts reinforce the co-occurrence graph
#
# Paper (Marwitz et al. 2026, Table 1): LLM-extracted concept phrases beat
# rule-based extraction (nominalization, formula cleanup, synonym resolution).
# build_contribution accepts optional `concepts`; they enter the SAME tokenize
# pipeline so their stems become nodes and their words co-occur, lifting
# LLM-validated concepts above body noise. `concepts=None` is byte-identical to
# today (graceful degradation).
# ---------------------------------------------------------------------------

def test_build_contribution_concepts_add_nodes_absent_from_body():
    st = __import__("snowballstemmer").stemmer("english").stemWord
    c = build_contribution("N", "alpha beta", concepts=["quantum entanglement"], lang="english")
    assert st("quantum") in c["nodes"]
    assert st("entanglement") in c["nodes"]


def test_build_contribution_concepts_create_intra_concept_edge():
    st = __import__("snowballstemmer").stemmer("english").stemWord
    c = build_contribution("N", "alpha beta", concepts=["quantum entanglement"], lang="english")
    # the two words of one concept are adjacent -> narrative weight 3
    assert _edge_weight(c, st("quantum"), st("entanglement")) == 3


def test_build_contribution_concepts_none_is_identical_to_today():
    base = build_contribution("N", "alpha beta gamma", lang="english")
    none = build_contribution("N", "alpha beta gamma", concepts=None, lang="english")
    empty = build_contribution("N", "alpha beta gamma", concepts=[], lang="english")
    assert none == base
    assert empty == base


def test_build_index_threads_concepts_by_path(tmp_path):
    """#9: build_index forwards per-path LLM concepts into build_contribution."""
    st = __import__("snowballstemmer").stemmer("english").stemWord
    store = CooccurStore(path=tmp_path / "c.json", lang="english")
    build_index(
        [("Notes/A", "A", "alpha beta")],
        store=store,
        concepts_by_path={"Notes/A": ["quantum entanglement"]},
        force=True,
    )
    nodes = store.note_nodes("Notes/A")
    assert st("quantum") in nodes
    assert st("entanglement") in nodes
