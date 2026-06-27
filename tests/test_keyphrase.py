"""Tests for silica.kernel.keyphrase — content-based concept extraction (Fase 1).

The thesis: markup-only extraction (recon.extract_concepts) returns ~0 real
concepts on prose with no headings/bold/acronyms; YAKE recovers them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BUNDLED_OVERLAYS = Path(__file__).resolve().parent.parent / "silica" / "overlays"

# Italian prose, NO markup: the case that broke the old markup-only recon.
_PROSE = (
    "La discesa del gradiente stocastico ottimizza la funzione di perdita "
    "aggiornando i pesi della rete neurale a ogni iterazione del training. "
    "Il tasso di apprendimento controlla l'ampiezza del passo di aggiornamento. "
    "La retropropagazione calcola i gradienti rispetto a ciascun parametro del modello."
)


@pytest.fixture
def it_overlay():
    path = _BUNDLED_OVERLAYS / "italian.yaml"
    if not path.exists():
        pytest.skip(f"bundled overlay not found: {path}")
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


def test_min_concepts_default_is_one_no_forced_padding():
    """A thin note maps to as few as one concept — the floor never manufactures."""
    from silica.kernel.keyphrase import MIN_CONCEPTS, _cutoff
    assert MIN_CONCEPTS == 1
    tiny = "w w w"                                  # 3 tokens → k = max(1, 3//20) = 1
    assert len(_cutoff(tiny, _fake_ranked(50))) == 1


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


# ---------------------------------------------------------------------------
# Corroboration tier: structural markup is a *second axis*, not only a boost.
# EXTRACTED <=> structurally corroborated (embedder-free); else INFERRED.
# ---------------------------------------------------------------------------

def test_embedder_down_structural_is_extracted_yake_only_is_inferred(it_overlay):
    """Embedder-down — the only corroboration available is author markup.

    A heading concept has a second independent signal → EXTRACTED. A prose-only
    YAKE concept has a single (junk-prone) signal → INFERRED. This is the gate
    the salience path cannot supply when the embedder is down.
    """
    from silica.kernel.keyphrase import extract_keyphrases

    body = (
        "# Discesa Del Gradiente Stocastico\n\n"
        "La discesa del gradiente stocastico ottimizza la funzione di perdita "
        "aggiornando i pesi della rete neurale a ogni iterazione del training. "
        "Il tasso di apprendimento controlla l'ampiezza del passo di aggiornamento. "
        "La retropropagazione calcola i gradienti rispetto a ciascun parametro del modello. "
        "La regolarizzazione riduce il sovradattamento penalizzando i pesi troppo grandi. "
        "La convalida incrociata stima la capacita di generalizzazione del modello."
    )
    cands = extract_keyphrases(body, overlay=it_overlay, lang="italian", embedder=None)
    by = {c.phrase.lower(): c.confidence for c in cands}

    assert by.get("discesa del gradiente stocastico") == "EXTRACTED"  # heading → second axis
    assert any(conf == "INFERRED" for conf in by.values())           # prose-only → single signal


def test_embedder_up_tier_independent_of_ranking_axis():
    """With an embedder, the tier still follows markup, not the theme cosine the
    ranker already uses: a heading concept is EXTRACTED, a theme-only one INFERRED."""
    from silica.kernel.keyphrase import extract_keyphrases
    from silica.kernel.overlay import DEFAULT_OVERLAY

    body = (
        "# Graph Memory\n\n"
        "The planning module reads the graph memory and writes planning results back. "
        "Planning over the graph memory improves planning quality and memory recall. "
        "A planning agent stores planning state in graph memory for later planning. "
        "The memory layer indexes planning episodes so planning can resume from memory."
    )
    cands = extract_keyphrases(body, overlay=DEFAULT_OVERLAY, lang="english", embedder=FakeEmbedder())
    by = {c.phrase.lower(): c.confidence for c in cands}

    assert by.get("graph memory") == "EXTRACTED"          # in a heading → corroborated
    assert any(conf == "INFERRED" for conf in by.values())  # theme-only candidates stay single-signal


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


def test_latex_body_yields_no_math_token_concepts():
    """LaTeX commands in the body never surface as concepts (stripped pre-YAKE)."""
    from silica.kernel.keyphrase import extract_keyphrases
    body = (
        "# Gradient descent\n\n"
        "The loss function $\\mathcal{L}$ is minimized by gradient descent. "
        "We compute $$\\sum_{i} \\nabla_w \\mathcal{L}_i \\leq \\epsilon$$ each step, "
        "updating the weights of the neural network until convergence. " * 3
    )
    cands = extract_keyphrases(body)  # default overlay/lang, no embedder
    phrases = " ".join(c.phrase.lower() for c in cands)
    for junk in ("mathcal", "sum", "nabla", "leq", "epsilon"):
        assert junk not in phrases, f"{junk!r} leaked from LaTeX"
    assert cands, "prose should still yield concepts"


def test_auto_lang_resolves_so_yake_drops_italian_function_words():
    """lang='auto' is resolved to a real Snowball language before YAKE, so YAKE
    drops Italian function words at candidate generation (no bogus 'au' ISO)."""
    from silica.kernel.keyphrase import extract_keyphrases
    from silica.kernel.overlay import DomainOverlay
    # Empty overlay isolates the YAKE-language effect: is_concept filters nothing,
    # so a leaked function word would survive to the output if lang were wrong.
    empty = DomainOverlay(stopwords=frozenset(), noise_patterns=())
    # Longer body ensures "della" survives _cutoff and makes it to the final output
    # if YAKE doesn't filter it (which happens with bogus "au" code).
    body = (
        "La discesa del gradiente della rete neurale aggiorna i pesi della rete. "
        "La funzione di perdita della rete dipende dai pesi della rete neurale. "
        "Il tasso di apprendimento della rete regola il passo. "
        "La retropropagazione della rete calcola i gradienti. " * 8
    )
    cands = extract_keyphrases(body, overlay=empty, lang="auto")
    phrases = {c.phrase.lower() for c in cands}
    assert "della" not in phrases  # IT function word dropped by YAKE(it), not 'au'


def test_yake_leg_augments_not_replaces_builtin_stopwords():
    """YAKE's built-in Italian stopword 'ancora' is filtered even though it is
    absent from the Italian overlay — proving union semantics, not replace.

    Word verified:
      'ancora' in yake.KeywordExtractor(lan='it').stopword_set  → True
      'ancora' in overlay_for_lang('italian').stopwords          → False

    With replace semantics (the bug): 'ancora' is NOT in the stopword set
    → YAKE produces it as a candidate → it leaks to output.
    With union semantics (the fix): 'ancora' IS in the unioned stopword set
    → YAKE never proposes it → it cannot appear in output.
    """
    from silica.kernel.keyphrase import extract_keyphrases
    from silica.kernel.overlay import overlay_for_lang

    overlay = overlay_for_lang("italian")
    # Precondition: 'ancora' is in YAKE's built-in Italian set but not the overlay
    import yake as _yake
    assert "ancora" in _yake.KeywordExtractor(lan="it", n=3, top=100, dedupLim=0.9).stopword_set
    assert "ancora" not in overlay.stopwords

    # Body: 'ancora' repeated many times alongside a real content word so that
    # if YAKE doesn't filter it, it would rank highly and survive _cutoff.
    body = (
        "Il percettrone e ancora un modello ancora usato ancora nella rete neurale. "
        "Ancora oggi il percettrone e ancora studiato e ancora applicato ancora. "
        "La regola di apprendimento del percettrone e ancora fondamentale ancora. " * 6
    )
    cands = extract_keyphrases(body, overlay=overlay, lang="italian")
    phrases = {c.phrase.lower() for c in cands}

    # 'ancora' must not surface as a standalone concept (built-in stopword)
    assert not any(p == "ancora" or p.startswith("ancora ") or p.endswith(" ancora")
                   for p in phrases), f"'ancora' leaked as concept: {phrases}"
    # Real content word must still appear
    assert any("percettrone" in p for p in phrases), f"content word lost: {phrases}"


def test_recon_italian_drops_latex_and_structural_keeps_content():
    """End-to-end (overlay=None -> overlay_for_lang('italian')): LaTeX, the
    'Lezione' heading, and the CFU acronym are gone; an Italian content word
    survives."""
    from silica.kernel.keyphrase import extract_keyphrases
    from silica.kernel.overlay import reset_overlay_cache
    reset_overlay_cache()
    body = (
        "## Lezione 10\n\n"
        "Il percettrone e un modello della rete neurale. "
        "Per ogni CFU si studia il percettrone e la sua regola di apprendimento. "
        "La funzione $\\mathbb{R} \\to \\mathbb{R}$ con $\\sum_i w_i x_i \\leq \\theta$ "
        "definisce l'attivazione del percettrone. " * 3
    )
    cands = extract_keyphrases(body, lang="italian")  # overlay=None on purpose
    phrases = " ".join(c.phrase.lower() for c in cands)
    for junk in ("mathbb", "lezione", "cfu", "sum", "leq", "theta"):
        assert junk not in phrases, f"{junk!r} should be filtered"
    assert "percettrone" in phrases, "content word lost"
