"""Content-based concept extraction (UKE).

The old `recon.extract_concepts` keyed concepts on markdown markup, so prose
papers — concepts living in unmarked sentences — extracted to nearly nothing.
This module instead generates candidates from the *content* via YAKE and ranks
them. Design split (validated on a real corpus, see the eval and the spec):

  - **YAKE = candidate generator.** Its own ranking is junk-prone (it floats
    rare contiguous n-grams like "promise to enhance"), so its rank is discarded
    once an embedder is available — YAKE only supplies the candidate *pool*.
  - **embedder + MMR = the ranker.** Candidates are ordered by cosine to the
    document theme, with MMR for diversity (plain cosine collapses onto
    near-synonym clusters). This is the primary signal.
  - **structural (markup) = boost.** Concepts that appear in a heading/bold/
    acronym get a relevance bonus — lifts lecture-genre concepts; on prose with
    no markup the boost set is empty (no effect).
  - **embedder down => fall back to YAKE rank** (degraded, deterministic).

Return shape is `list[ConceptCandidate]`, ranked best-first.
See docs/superpowers/specs/2026-06-19-concept-recon-design.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from silica.kernel.embed import _cosine, document_theme_vector
from silica.kernel.overlay import DomainOverlay, get_active_overlay
from silica.kernel.recon import _strip_frontmatter, is_concept, normalize

# Cutoff knobs (calibration — tune on a real paper + lecture via the eval).
# Tuned on the eval (3 real docs). Concept density varies wildly across genres
# (~40 tok/concept for a paper, ~8 for a dense distilled lecture note), so one
# linear ratio is a compromise; a density-aware cutoff (cosine elbow) would serve
# short dense notes better — deferred until the linear clamp proves insufficient.
TOKENS_PER_CONCEPT = 20   # ponytail: linear clamp; tune via the eval
MIN_CONCEPTS = 1          # a note may map to a single concept — no forced padding
MAX_CONCEPTS = 40
YAKE_POOL = 100           # candidates YAKE proposes (also the rerank pool)

# Rerank knobs (Fase 2 — tune via the eval).
MMR_LAMBDA = 0.6          # relevance vs diversity in MMR; lower = more diverse
STRUCT_BOOST = 0.3        # relevance bonus for a concept present in markup

# CONFIG.cooccurrence_lang is Snowball-style ("italian"); YAKE wants ISO ("it").
_SNOWBALL_TO_ISO = {
    "arabic": "ar", "danish": "da", "dutch": "nl", "english": "en",
    "finnish": "fi", "french": "fr", "german": "de", "hungarian": "hu",
    "italian": "it", "norwegian": "no", "portuguese": "pt", "romanian": "ro",
    "russian": "ru", "spanish": "es", "swedish": "sv",
}


@dataclass
class ConceptCandidate:
    phrase: str
    score: float                       # ordering only (YAKE cost; lower = better). NOT calibrated.
    evidence: list[str] = field(default_factory=list)  # provenance/debug, e.g. ["yake:0.12"]
    # Corroboration tier (vocabulary mirrors links, see analyst_plan.py):
    #   EXTRACTED — structurally corroborated (author markup; second, embedder-free axis)
    #   INFERRED  — single signal only (embedding cosine or YAKE rank), uncorroborated
    confidence: str = "INFERRED"


def _yake_leg(text: str, overlay: DomainOverlay, lang: str) -> list[ConceptCandidate] | None:
    """YAKE-ranked candidates (best-first), filtered through the overlay.

    Abstains (None) if YAKE is unimportable or yields nothing. YAKE returns
    (phrase, cost) ascending (lower cost = more relevant), already deduplicated.
    """
    try:
        import yake
    except ImportError:
        return None

    iso = _SNOWBALL_TO_ISO.get(lang.lower(), lang.lower()[:2] or "en")
    kw = yake.KeywordExtractor(lan=iso, n=3, top=YAKE_POOL, dedupLim=0.9)
    raw = kw.extract_keywords(text)  # already sorted ascending (best-first)
    if not raw:
        return None

    out: list[ConceptCandidate] = []
    for phrase, cost in raw:
        norm = normalize(phrase)
        if is_concept(norm, overlay=overlay):
            out.append(ConceptCandidate(phrase=norm, score=float(cost),
                                        evidence=[f"yake:{cost:.3f}"]))
    return out or None


# ---------------------------------------------------------------------------
# Structural (markup) signal — re-added for the boost; pure regex extractors.
# ---------------------------------------------------------------------------

def from_headings(content: str) -> set:
    return {m.group(1) for m in re.finditer(r'^#{1,4}\s+(.+?)\s*$', content, re.MULTILINE)}


def from_bold(content: str) -> set:
    return {m.group(1) for m in re.finditer(r'\*\*(.+?)\*\*', content)}


def from_acronyms(content: str) -> set:
    return set(re.findall(r'\b[A-Z]{2,6}\b', content))


def _structural_concepts(body: str, overlay: DomainOverlay) -> set[str]:
    """Lowercased concepts present in markup (heading/bold/acronym), overlay-filtered.

    Empty on prose with no markup — that is the leg "abstaining" for the boost.
    """
    raw = from_headings(body) | from_bold(body) | from_acronyms(body)
    out: set[str] = set()
    for r in raw:
        n = normalize(r)
        if is_concept(n, overlay=overlay):
            out.add(n.lower())
    return out


# ---------------------------------------------------------------------------
# Embedder + MMR ranker
# ---------------------------------------------------------------------------

def _mmr(vecs, theme, k, lam: float = MMR_LAMBDA, rel=None) -> list[int]:
    """Maximal Marginal Relevance selection. Returns selected indices, best-first.

    `rel[i]` is the relevance of candidate i (default: cosine to `theme`). Each
    pick maximises `lam*rel - (1-lam)*max similarity to already-picked`, so
    near-duplicates of a selected candidate are demoted.
    """
    if rel is None:
        rel = [_cosine(v, theme) for v in vecs]
    cand = list(range(len(vecs)))
    sel: list[int] = []
    while cand and len(sel) < k:
        if not sel:
            i = max(cand, key=lambda i: rel[i])
        else:
            i = max(cand, key=lambda i: lam * rel[i]
                    - (1 - lam) * max(_cosine(vecs[i], vecs[j]) for j in sel))
        sel.append(i)
        cand.remove(i)
    return sel


def _rerank(
    pool: list[ConceptCandidate],
    body: str,
    overlay: DomainOverlay,
    embedder,
) -> list[ConceptCandidate] | None:
    """Rerank the YAKE pool by embedder cosine-to-theme + MMR + structural boost.

    Returns None (abstain -> caller falls back to YAKE rank) when no embedder, an
    empty document theme, or an embedding failure.
    """
    if embedder is None:
        return None
    theme = document_theme_vector(embedder, body)
    if not theme:
        return None
    phrases = [c.phrase for c in pool]
    try:
        vecs = embedder.embed(phrases)
    except Exception:
        return None
    if not vecs:
        return None

    structural = _structural_concepts(body, overlay)
    rel = [
        _cosine(vecs[i], theme) + (STRUCT_BOOST if phrases[i].lower() in structural else 0.0)
        for i in range(len(pool))
    ]
    order = _mmr(vecs, theme, k=len(pool), lam=MMR_LAMBDA, rel=rel)

    out: list[ConceptCandidate] = []
    for i in order:
        ev = [f"embed:{_cosine(vecs[i], theme):.2f}"]
        if phrases[i].lower() in structural:
            ev.append("struct")
        out.append(ConceptCandidate(phrase=pool[i].phrase, score=rel[i], evidence=ev))
    return out


def _seed_structural(
    body: str, overlay: DomainOverlay, pool: list[ConceptCandidate],
) -> list[ConceptCandidate]:
    """Prepend markup concepts (heading/bold/acronym) absent from the YAKE pool.

    Author markup is frequency-independent: it recovers concepts YAKE can't reach
    — single-occurrence terms, and phrases longer than its max n-gram (n=3). With
    an embedder these are reranked by cosine like any candidate; in the fallback
    they lead, since author markup is the strongest deterministic signal we have.
    """
    structural = _structural_concepts(body, overlay)  # lowercased, overlay-filtered
    have = {c.phrase.lower() for c in pool}
    seeded = [ConceptCandidate(phrase=s, score=0.0, evidence=["struct"])
              for s in sorted(structural) if s not in have]
    return seeded + pool


def _cutoff(content: str, ranked: list[ConceptCandidate]) -> list[ConceptCandidate]:
    n_tok = len(content.split())
    k = max(MIN_CONCEPTS, min(MAX_CONCEPTS, n_tok // TOKENS_PER_CONCEPT))
    return ranked[:min(k, len(ranked))]


def extract_keyphrases(
    content: str,
    *,
    overlay: DomainOverlay | None = None,
    lang: str = "english",
    embedder=None,
) -> list[ConceptCandidate]:
    """Ranked concept candidates from *content*.

    YAKE generates the candidate pool, seeded with markup concepts it can't reach
    (see `_seed_structural`); if an `embedder` is given it ranks the pool (cosine-
    to-theme + MMR + structural boost), otherwise the structural-first / YAKE rank
    is used (degraded fallback). Returns [] only when both legs abstain, which
    `silica_recon` already handles as an empty report.
    """
    if overlay is None:
        overlay = get_active_overlay()
    body = _strip_frontmatter(content)  # metadata, not content concepts
    pool = _seed_structural(body, overlay, _yake_leg(body, overlay, lang) or [])
    if not pool:
        return []
    ranked = _rerank(pool, body, overlay, embedder)
    if ranked is None:
        ranked = pool  # fallback: structural-first, then YAKE rank

    # Stamp the corroboration tier from the second (embedder-free) axis: a concept
    # present in author markup is corroborated → EXTRACTED; otherwise INFERRED.
    # One rule, both paths — survives the embedder-down fallback, where it is the
    # only gate left (salience needs the embedder). ponytail: recompute structural
    # here (cheap regex) rather than thread it through _rerank/_seed_structural.
    structural = _structural_concepts(body, overlay)
    for c in ranked:
        if c.phrase.lower() in structural:
            c.confidence = "EXTRACTED"
    return _cutoff(body, ranked)


if __name__ == "__main__":  # ponytail: self-check, no framework
    txt = ("La discesa del gradiente ottimizza la funzione di perdita aggiornando "
           "i pesi della rete neurale a ogni iterazione del training. " * 3)
    from silica.kernel.overlay import DomainOverlay as _DO
    cands = extract_keyphrases(txt, overlay=_DO(stopwords=frozenset(), noise_patterns=()),
                               lang="italian")
    assert cands, "expected concepts from prose"
    assert len(cands) <= MAX_CONCEPTS  # lower bound not guaranteed: cutoff caps at available
    print(f"OK: {len(cands)} concepts; top={cands[0].phrase!r}")
