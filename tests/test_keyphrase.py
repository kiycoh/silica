"""Tests for silica.kernel.keyphrase — content-based concept extraction (Fase 1).

The thesis: markup-only extraction (recon.extract_concepts) returns ~0 real
concepts on prose with no headings/bold/acronyms; YAKE recovers them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_EXAMPLE_OVERLAYS = Path(__file__).resolve().parent.parent / "examples" / "overlays"

# Italian prose, NO markup: the case that broke the old markup-only recon.
_PROSE = (
    "La discesa del gradiente stocastico ottimizza la funzione di perdita "
    "aggiornando i pesi della rete neurale a ogni iterazione del training. "
    "Il tasso di apprendimento controlla l'ampiezza del passo di aggiornamento. "
    "La retropropagazione calcola i gradienti rispetto a ciascun parametro del modello."
)


@pytest.fixture
def it_overlay():
    path = _EXAMPLE_OVERLAYS / "it-academic.yaml"
    if not path.exists():
        pytest.skip(f"examples overlay not found: {path}")
    from silica.kernel.overlay import load_overlay
    return load_overlay(path)


def test_prose_extracts_content_concepts(it_overlay):
    """Prose with no markup yields real domain concepts (markup-only gave ~0)."""
    from silica.kernel.keyphrase import extract_keyphrases

    cands = extract_keyphrases(_PROSE, overlay=it_overlay, lang="italian")
    phrases = " ".join(c.phrase.lower() for c in cands)

    assert cands, "no concepts extracted from prose"
    assert "gradiente" in phrases or "rete neurale" in phrases


def _fake_ranked(n):
    from silica.kernel.keyphrase import ConceptCandidate
    return [ConceptCandidate(phrase=f"c{i}", score=float(i)) for i in range(n)]


def test_cutoff_scales_with_tokens_and_caps():
    """k = clamp(n_tok / TOKENS_PER_CONCEPT, MIN, MAX), capped at candidates."""
    from silica.kernel.keyphrase import (
        MAX_CONCEPTS, MIN_CONCEPTS, TOKENS_PER_CONCEPT, _cutoff,
    )
    pool = _fake_ranked(100)

    huge = "w " * (TOKENS_PER_CONCEPT * (MAX_CONCEPTS + 10))   # well past MAX
    assert len(_cutoff(huge, pool)) == MAX_CONCEPTS

    mid = "w " * (TOKENS_PER_CONCEPT * 12)                     # 12 in [MIN, MAX]
    assert len(_cutoff(mid, pool)) == 12

    tiny = "w " * 5                                            # below MIN => floor
    assert len(_cutoff(tiny, pool)) == MIN_CONCEPTS

    assert len(_cutoff(huge, _fake_ranked(7))) == 7           # never exceed candidates


def test_frontmatter_ignored(it_overlay):
    """YAML front matter is metadata, not content: it must not change concepts."""
    from silica.kernel.keyphrase import extract_keyphrases

    body = _PROSE
    with_fm = "---\ntitle: ZzzParolaSegreta\ntags: [nascosto]\n---\n" + body
    a = [c.phrase for c in extract_keyphrases(with_fm, overlay=it_overlay, lang="italian")]
    b = [c.phrase for c in extract_keyphrases(body, overlay=it_overlay, lang="italian")]

    assert a == b


def test_empty_content_abstains(it_overlay):
    """No content => empty list (silica_recon handles it as an empty report)."""
    from silica.kernel.keyphrase import extract_keyphrases

    assert extract_keyphrases("", overlay=it_overlay, lang="italian") == []


# ---------------------------------------------------------------------------
# Fase 2: YAKE = pool generator, embedder + MMR = ranker, structural = boost
# ---------------------------------------------------------------------------

_AXES = ("graph", "memory", "planning", "noise")


class FakeEmbedder:
    """Deterministic embedder: vector over topic axes by word presence."""
    def embed(self, texts):
        return [[float(ax in t.lower()) for ax in _AXES] for t in texts]


def test_structural_concepts_from_markup():
    """Heading / bold / acronym concepts are extracted and overlay-filtered (lowercased)."""
    from silica.kernel.keyphrase import _structural_concepts
    from silica.kernel.overlay import DEFAULT_OVERLAY

    body = "# Reti Neurali\n\nUso di **Gradient Descent** e il PID controller."
    concs = _structural_concepts(body, DEFAULT_OVERLAY)

    assert "reti neurali" in concs   # heading
    assert "gradient descent" in concs  # bold
    assert "pid" in concs            # acronym


def test_mmr_demotes_near_duplicate():
    """MMR picks a diverse candidate over a near-duplicate of an already-selected one."""
    from silica.kernel.keyphrase import _mmr

    vecs = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]  # 0 and 1 identical, 2 orthogonal
    order = _mmr(vecs, theme=[1.0, 1.0], k=2, lam=0.6)

    assert order[0] in (0, 1)
    assert 2 in order                       # diversity reaches the orthogonal item
    assert not (0 in order and 1 in order)  # not both duplicates


def test_rerank_orders_thematic_above_junk_and_abstains_without_embedder():
    from silica.kernel.keyphrase import _rerank, ConceptCandidate
    from silica.kernel.overlay import DEFAULT_OVERLAY

    pool = [ConceptCandidate("promise to enhance", 0.0),
            ConceptCandidate("knowledge graph", 0.0),
            ConceptCandidate("graph memory", 0.0)]
    body = "graph memory planning graph memory knowledge graph"

    ranked = _rerank(pool, body, DEFAULT_OVERLAY, FakeEmbedder())
    phrases = [c.phrase for c in ranked]
    assert phrases.index("knowledge graph") < phrases.index("promise to enhance")

    assert _rerank(pool, body, DEFAULT_OVERLAY, None) is None  # no embedder => abstain


def test_structural_boost_promotes_markup_concept():
    """A thematically-flat concept that appears in a heading is lifted by the structural boost."""
    from silica.kernel.keyphrase import _rerank, ConceptCandidate
    from silica.kernel.overlay import DEFAULT_OVERLAY

    pool = [ConceptCandidate("alpha widget", 0.0), ConceptCandidate("beta gadget", 0.0)]
    body = "# Beta Gadget\n\nsome unrelated prose"  # both flat on theme; beta is in a heading

    ranked = _rerank(pool, body, DEFAULT_OVERLAY, FakeEmbedder())
    phrases = [c.phrase for c in ranked]
    assert phrases.index("beta gadget") < phrases.index("alpha widget")


# ---------------------------------------------------------------------------
# Fase A: structural markup is also a *candidate source*, not only a boost
# ---------------------------------------------------------------------------

def test_structural_phrase_beyond_yake_ngram_enters_pool():
    """A markup-marked phrase longer than YAKE's max n-gram (n=3) can never be a
    YAKE candidate, yet the author bolded it. The structural leg must seed it into
    the pool so it survives even in the embedder-down fallback."""
    from silica.kernel.keyphrase import _yake_leg, extract_keyphrases
    from silica.kernel.overlay import DomainOverlay

    overlay = DomainOverlay(stopwords=frozenset(), noise_patterns=())
    body = ("This work studies sequential decision making in agents. "
            "The setting is a **partially observable markov decision process** "
            "and we evaluate planning under it across many tasks and domains.")

    # precondition: YAKE (n=3) cannot produce the 4+ word phrase
    pool = _yake_leg(body, overlay, "english") or []
    assert all("partially observable markov decision" not in c.phrase.lower() for c in pool)

    # behaviour: the embedder-down fallback still surfaces the bolded concept
    out = [c.phrase.lower()
           for c in extract_keyphrases(body, overlay=overlay, lang="english", embedder=None)]
    assert any("partially observable markov decision process" in p for p in out)


def test_extract_keyphrases_rerank_end_to_end():
    """With an embedder, extract_keyphrases reranks; without, it falls back to YAKE order."""
    from silica.kernel.keyphrase import extract_keyphrases
    from silica.kernel.overlay import DEFAULT_OVERLAY

    body = ("The knowledge graph stores memory. Planning over the graph memory improves "
            "planning. A knowledge graph is a memory structure for planning.")
    with_emb = [c.phrase for c in extract_keyphrases(body, overlay=DEFAULT_OVERLAY, lang="english", embedder=FakeEmbedder())]
    no_emb = [c.phrase for c in extract_keyphrases(body, overlay=DEFAULT_OVERLAY, lang="english", embedder=None)]

    assert with_emb and no_emb
    assert with_emb != no_emb  # reranking actually changed the order
