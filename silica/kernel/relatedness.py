# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Relatedness facade — fuses the two PROPOSE-layers into one note ranking.

Silica has two independent signals about which notes belong together:

  - **embeddings** (`kernel/embed.py`)        — holistic semantic similarity
  - **co-occurrence** (`kernel/cooccurrence.py`) — how the author actually
    co-mentions concepts (deterministic, embedder-free)

They live at different granularities (note-level vs concept-level) and on
incomparable scales (cosine in [0,1] vs unbounded integer weight). This module
is the single place where they meet. It:

  1. Reconciles granularity via a concept->notes **inverted index**, turning the
     concept-level co-occurrence graph into a note-level ranking.
  2. Fuses the two note rankings with **Reciprocal Rank Fusion** (RRF), which
     only consults rank position, so incomparable scores combine cleanly.
  3. Lets a degenerate proponent **abstain** (return None) instead of emitting a
     flat zero ranking that would poison RRF. A leg that abstains contributes no
     reciprocal-rank terms, so fusion degrades automatically to the survivor's
     ranking — "embedder down => routing on co-occurrence", with no special-case
     branch.

Provenance is preserved: every returned note carries an `evidence` list
(`embed:0.83`, `cooccur:w9`, or both).

Generalises the existing rule "embeddings PROPOSE, graph DISPOSES" into
"the proponents propose, the graph disposes": this facade is a proposer, never
authoritative about vault structure.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Callable

from silica.kernel.cooccurrence import CooccurStore
from silica.kernel.embed import EmbedStore
from silica.kernel.graph_export import is_vault_artifact

# Standard RRF damping constant (Cormack et al. 2009). Larger -> flatter weight
# decay across ranks; 60 is the widely-used default.
RRF_K = 60

# A leg whose best candidate scores at or below this is treated as signal-free
# (e.g. a zero query vector makes every cosine 0.0) and abstains.
_NOISE_FLOOR = 1e-6

# Cooccur confidence gate (retrieval-gates spec 2026-07-14). ponytail: dormant —
# 0.0 never fires. Phase-0 (2026-07-17, bench/phase0_gates.json) recorded the
# no-fire reference only: vault coverage p10 0.259 / lme_s p10 0.432, so any
# future threshold <=0.1 is home-turf-safe. The fire side (MuSiQue, vocabulary
# mismatch) is no longer on disk; freeze only after a MuSiQue re-run, and shelve
# the gate if that run shows no wide separation (spec abort criterion).
_COOCCUR_MIN_CONFIDENCE = 0.0
# Calibration hook: harnesses set it to capture per-query
# {"coverage", "flatness", "fired"}; production leaves it None.
COOCCUR_GATE_PROBE: Callable[[dict], None] | None = None

# Neighbour edges are associative, not direct membership: discount their pull on
# the query concept profile so notes literally sharing concepts still dominate.
_EXPANSION_DISCOUNT = 0.25

# Minimum per-leg candidate pool fed to RRF, independent of the caller's k, so
# fusion has enough material to find agreement before the final top-k cut.
_POOL_MIN = 25


@dataclass
class RelatedNote:
    """A fused related-note candidate with its provenance.

    `score` is the RRF score (only meaningful for ordering, not as a similarity).
    `evidence` records which legs proposed it and their native scores, as
    display strings; `embed_score` / `cooccur_weight` expose the same raw signals
    structurally (None when that leg did not propose the note) so callers can
    threshold or render without parsing the evidence strings.
    """
    path: str
    name: str
    score: float
    evidence: list[str]
    embed_score: float | None = None
    cooccur_weight: float | None = None
    edge_score: float | None = None  # CORRELATE (ADR-0013): direct note_edges Jaccard
    # ADR-0019: "vault" = active vault, "memory" = personal-memory lane. A
    # memory result's `path` is relative to the MEMORY vault — consumers must
    # respect this marker (open the right note in the right vault) and never
    # write through it.
    origin: str = "vault"


# ---------------------------------------------------------------------------
# RRF fusion (pure)
# ---------------------------------------------------------------------------

def _rrf_fuse(rankings: list[list[tuple[str, float]]]) -> dict[str, float]:
    """Reciprocal Rank Fusion over several ranked lists of (key, _score).

    Each list must be sorted best-first. The native score is ignored — only the
    rank position counts — so lists on incomparable scales combine cleanly. A key
    appearing in multiple lists accumulates a reciprocal-rank term from each.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (key, _score) in enumerate(ranking):
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
    return fused


# ---------------------------------------------------------------------------
# Embedding leg
# ---------------------------------------------------------------------------

def _rank_embeddings_from_vec(
    embed_store: EmbedStore | None,
    vec: list[float] | None,
    *,
    k: int,
    exclude: set[str],
) -> list[tuple[str, str, float]] | None:
    """Note ranking from a query vector, or None if the embed leg abstains.

    Abstains when: no store, no vector, the search errors, or the output is
    degenerate (every score at the noise floor — a flat zero ranking that would
    poison RRF rather than inform it).
    """
    if embed_store is None or vec is None:
        return None
    try:
        cands = embed_store.cosine_top_k(vec, k=k, exclude=exclude)
    except Exception:
        return None
    if not cands:
        return None
    if max((c.get("score", 0.0) for c in cands), default=0.0) <= _NOISE_FLOOR:
        return None
    return [(c["path"], c["name"], float(c.get("score", 0.0))) for c in cands]


def _embed_ranking(
    embed_store: EmbedStore | None,
    query_path: str,
    *,
    k: int,
    exclude: set[str],
) -> list[tuple[str, str, float]] | None:
    """Embed ranking for an INDEXED note: resolve its vector by path, then rank."""
    if embed_store is None:
        return None
    vec = embed_store.get_vec(query_path)
    if vec is None:
        vec = embed_store.get_vec(query_path.removesuffix(".md"))
    return _rank_embeddings_from_vec(embed_store, vec, k=k, exclude=exclude)


# ---------------------------------------------------------------------------
# Co-occurrence leg (granularity reconciliation)
# ---------------------------------------------------------------------------

def _path_in_scope(path: str, scope: str | None) -> bool:
    if not scope:
        return True
    s = scope.strip("/").lower()
    p = path.strip("/").lower()
    return p == s or p.startswith(s + "/")


def _profile_from_seeds(
    cooccur_store: CooccurStore,
    seeds: dict[str, float],
    *,
    scope: str | None,
    expand: bool,
) -> dict[str, float]:
    """Weighted concept profile {stem: weight} from seed concepts.

    When `expand`, adds each seed's co-occurrence neighbours at a discounted
    weight (associative reach: a note about a strongly-linked neighbour concept
    is related even without a literal shared concept).
    """
    if not seeds:
        return {}
    profile: dict[str, float] = dict(seeds)
    if expand:
        adj = cooccur_store.adjacency(scope=scope)
        for stem, weight in list(profile.items()):
            for neighbour, edge_weight in adj.get(stem, {}).items():
                profile[neighbour] = (
                    profile.get(neighbour, 0.0)
                    + weight * edge_weight * _EXPANSION_DISCOUNT
                )
    return profile


def _seed_from_text(text: str, lang: str) -> dict[str, float]:
    """Seed concepts {stem: count} from raw query text (for fresh queries)."""
    from silica.kernel.cooccurrence import tokenize

    seeds: dict[str, float] = {}
    for sentence in tokenize(text, stem_lang=lang):
        for stem, _surface in sentence:
            seeds[stem] = seeds.get(stem, 0.0) + 1.0
    return seeds


def _concept_idf(
    cooccur_store: CooccurStore,
    stems: set[str],
    *,
    scope: str | None,
) -> dict[str, float]:
    """Inverse document frequency per stem: log((N+1) / df), N = in-scope notes.

    Without this, a hub concept present in hundreds of notes (e.g. "data
    science", "statistica") dominates every ranking purely by breadth, burying
    the discriminative concepts that actually make two notes the same. IDF is the
    standard fix — a stem in every note scores ~0, a rare stem scores high — and
    on the real vault it lifts true twins from rank 6/miss into the visible top-k
    where plain overlap left them buried. Rarity is a corpus property, so `blocked`
    (query + excludes) still counts toward df.

    The `N+1` numerator (smoothed IDF) keeps the weight strictly positive even
    when a stem sits in every note — the raw `log(N/df)` collapses to exactly 0
    there, which on a tiny or brand-new vault (N=1, every stem ubiquitous) would
    zero the whole co-occurrence signal and silently drop the leg. On a real
    corpus the smoothing is negligible, so hub suppression is unchanged.

    ponytail: one extra O(notes) pass, recomputed per query; memoise on the store
    if a profiler ever shows it hot.
    """
    df: dict[str, int] = {}
    n = 0
    for path in cooccur_store.paths():
        if not _path_in_scope(path, scope):
            continue
        n += 1
        for stem in cooccur_store.note_nodes(path):
            if stem in stems:
                df[stem] = df.get(stem, 0) + 1
    import math

    return {stem: math.log((n + 1) / c) for stem, c in df.items() if c > 0}


def _rank_cooccur_from_profile(
    cooccur_store: CooccurStore,
    profile: dict[str, float],
    *,
    k: int,
    blocked: set[str],
    scope: str | None,
) -> list[tuple[str, float]] | None:
    """Rank in-scope notes by IDF-weighted concept overlap with `profile`
    (implicit concept->notes inverted index). Returns None when nothing overlaps.
    """
    if not profile:
        return None
    idf = _concept_idf(cooccur_store, set(profile), scope=scope)
    note_scores: dict[str, float] = {}
    for path in cooccur_store.paths():
        if path in blocked or not _path_in_scope(path, scope):
            continue
        overlap = 0.0
        for stem, count in cooccur_store.note_nodes(path).items():
            weight = profile.get(stem)
            if weight:
                overlap += weight * count * idf.get(stem, 0.0)
        if overlap > 0.0:
            note_scores[path] = overlap
    if not note_scores:
        return None
    ranked = sorted(note_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    # Confidence signals (retrieval-gates spec): coverage measures the diagnosed
    # cause (query/corpus vocabulary mismatch — IDF mass of profile stems the top
    # hit actually matches), flatness the symptom (indiscriminate near-uniform
    # scores). Values already in hand; no extra corpus pass.
    total_mass = sum(w * idf.get(s, 0.0) for s, w in profile.items())
    top_stems = set(cooccur_store.note_nodes(ranked[0][0]))
    matched = sum(w * idf.get(s, 0.0) for s, w in profile.items() if s in top_stems)
    coverage = (matched / total_mass) if total_mass > 0 else 0.0
    scores = [s for _p, s in ranked]
    flatness = scores[0] / statistics.median(scores)
    fired = coverage < _COOCCUR_MIN_CONFIDENCE
    if COOCCUR_GATE_PROBE:
        COOCCUR_GATE_PROBE({"coverage": coverage, "flatness": flatness, "fired": fired})
    if fired:
        return None
    return ranked


def _cooccur_ranking(
    cooccur_store: CooccurStore | None,
    query_path: str,
    *,
    k: int,
    exclude: set[str],
    scope: str | None = None,
    expand: bool = True,
) -> list[tuple[str, float]] | None:
    """Co-occurrence ranking for an INDEXED note: seed from its own concepts."""
    if cooccur_store is None:
        return None
    profile = _profile_from_seeds(
        cooccur_store, cooccur_store.note_nodes(query_path), scope=scope, expand=expand
    )
    return _rank_cooccur_from_profile(
        cooccur_store, profile, k=k, blocked=set(exclude) | {query_path}, scope=scope
    )


# ---------------------------------------------------------------------------
# Facade entry point
# ---------------------------------------------------------------------------

def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


# Key namespace for memory-lane candidates inside the shared RRF dict: the two
# lanes are different vaults, so identical relative paths are DIFFERENT notes.
# NUL cannot appear in a filename, so the prefix can never collide.
_MEM = "\x00memory:"


def _fuse(
    embed_rank: list[tuple[str, str, float]] | None,
    cooc_rank: list[tuple[str, float]] | None,
    *,
    k: int,
    edges_rank: list[tuple[str, float]] | None = None,
    mem_embed_rank: list[tuple[str, str, float]] | None = None,
    mem_cooc_rank: list[tuple[str, float]] | None = None,
    recall_rank: list[tuple[str, float]] | None = None,
) -> list[RelatedNote]:
    """RRF-fuse the per-leg rankings into RelatedNotes with provenance.

    Shared by both facade entry points. A None ranking is an abstaining leg and
    contributes no terms; [] is returned only when all legs abstain.

    `edges_rank` is the CORRELATE third leg — direct note_edges of the query,
    ranked by Jaccard. Only related_notes(query_path) supplies it; the
    fresh-query facade always abstains here (fresh text has no note_edges row).

    `mem_embed_rank` / `mem_cooc_rank` are the personal-memory lane (ADR-0019):
    same fusion, key-namespaced under `_MEM` so a memory note never collides
    with (or masquerades as) an active-vault note. Its results come out with
    origin="memory" and `memory:`-prefixed evidence.

    `recall_rank` (phase 1 of `improve`, LoCoMo eval-only): notes previously
    confirmed helpful for this vault/conversation, best-first by weight. None
    abstains — every caller except the LoCoMo runner leaves this unset.
    """
    rankings: list[list[tuple[str, float]]] = []
    embed_scores: dict[str, float] = {}
    names: dict[str, str] = {}
    if embed_rank is not None:
        rankings.append([(path, score) for path, _name, score in embed_rank])
        for path, name, score in embed_rank:
            embed_scores[path] = score
            names[path] = name
    if mem_embed_rank is not None:
        rankings.append([(_MEM + path, score) for path, _name, score in mem_embed_rank])
        for path, name, score in mem_embed_rank:
            embed_scores[_MEM + path] = score
            names[_MEM + path] = name

    cooc_scores: dict[str, float] = {}
    if cooc_rank is not None:
        rankings.append(list(cooc_rank))
        cooc_scores = dict(cooc_rank)
    if mem_cooc_rank is not None:
        rankings.append([(_MEM + path, w) for path, w in mem_cooc_rank])
        cooc_scores.update({_MEM + path: w for path, w in mem_cooc_rank})

    edge_scores: dict[str, float] = {}
    if edges_rank is not None:
        rankings.append(list(edges_rank))
        edge_scores = dict(edges_rank)

    recall_scores: dict[str, float] = {}
    if recall_rank is not None:
        rankings.append(list(recall_rank))
        recall_scores = dict(recall_rank)

    fused = _rrf_fuse(rankings)
    # Vault-root artifacts (log.md, GRAPH_REPORT.md) are excluded at index-build,
    # but a stale vector embedded before that exclusion outlives it: the store is
    # upsert-only and never prunes departed notes. Drop them here, before the
    # top-k cut, so no store consumer (map/autolink/dedup) ever surfaces one.
    fused = {
        p: s for p, s in fused.items()
        if not is_vault_artifact(p.removeprefix(_MEM))
    }
    if not fused:
        return []

    out: list[RelatedNote] = []
    for path, score in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0])):
        evidence: list[str] = []
        embed_score = embed_scores.get(path)
        cooc_weight = cooc_scores.get(path)
        edge_score = edge_scores.get(path)
        if embed_score is not None:
            evidence.append(f"embed:{embed_score:.2f}")
        if cooc_weight is not None:
            evidence.append(f"cooccur:w{int(round(cooc_weight))}")
        if edge_score is not None:
            evidence.append(f"edge:{edge_score:.2f}")
        recall_weight = recall_scores.get(path)
        if recall_weight is not None:
            evidence.append(f"recall:{int(recall_weight)}")
        origin = "vault"
        if path.startswith(_MEM):
            origin = "memory"
            evidence = [f"memory:{e}" for e in evidence]
        out.append(
            RelatedNote(
                path=path.removeprefix(_MEM),
                name=names.get(path, _basename(path.removeprefix(_MEM))),
                score=score,
                evidence=evidence,
                embed_score=embed_score,
                cooccur_weight=cooc_weight,
                edge_score=edge_score,
                origin=origin,
            )
        )
    return out[:k]


def related_notes(
    query_path: str,
    *,
    embed_store: EmbedStore | None = None,
    cooccur_store: CooccurStore | None = None,
    memory_embed_store: EmbedStore | None = None,
    memory_cooccur_store: CooccurStore | None = None,
    k: int = 10,
    scope: str | None = None,
    exclude: set[str] | None = None,
    expand: bool = False,
) -> list[RelatedNote]:
    """Return the top-k notes related to an INDEXED note `query_path`.

    Stores are injected (pass None for a leg that is unavailable — that leg
    abstains and fusion degrades to the survivor). Returns [] only when both
    legs abstain. Each result carries `evidence` recording its provenance.

    `memory_*_store` are the personal-memory lane (ADR-0019): the same query
    signals (the note's vector / concept stems) ranked against the memory
    vault's stores. None (the default) ⇒ the lane abstains and fusion is
    bit-identical to single-vault. `scope`/`exclude` are active-vault concepts
    and do not apply to the memory lane.

    `expand` (default off) adds associative co-occurrence neighbours to the
    concept profile. On a real vault this re-inflates hub concepts and buries
    true matches even under IDF weighting, so it stays opt-in for the one caller
    that wants pure associative reach (autolink candidate discovery).
    """
    blocked = set(exclude or ()) | {query_path}
    pool = max(k * 3, _POOL_MIN)

    embed_rank = _embed_ranking(embed_store, query_path, k=pool, exclude=blocked)
    cooc_rank = _cooccur_ranking(
        cooccur_store, query_path, k=pool, exclude=blocked, scope=scope, expand=expand
    )

    mem_embed_rank = None
    if memory_embed_store is not None and embed_store is not None:
        vec = embed_store.get_vec(query_path)
        if vec is None:
            vec = embed_store.get_vec(query_path.removesuffix(".md"))
        mem_embed_rank = _rank_embeddings_from_vec(
            memory_embed_store, vec, k=pool, exclude=set()
        )
    mem_cooc_rank = None
    if memory_cooccur_store is not None and cooccur_store is not None:
        profile = _profile_from_seeds(
            memory_cooccur_store,
            cooccur_store.note_nodes(query_path),
            scope=None,
            expand=expand,
        )
        mem_cooc_rank = _rank_cooccur_from_profile(
            memory_cooccur_store, profile, k=pool, blocked=set(), scope=None
        )
    # CORRELATE third leg: the query's direct note_edges row, ranked by Jaccard.
    # Abstains (None) when the row is empty — 0.57 edges/note means most queries
    # abstain and fusion is identical to before; when it fires, high precision.
    edges_rank = None
    if cooccur_store is not None:
        row = cooccur_store.note_edges_for(query_path)
        ranked = sorted(
            ((p, s) for p, s in row.items() if p not in blocked),
            key=lambda kv: (-kv[1], kv[0]),
        )
        edges_rank = ranked or None
    return _fuse(
        embed_rank,
        cooc_rank,
        k=k,
        edges_rank=edges_rank,
        mem_embed_rank=mem_embed_rank,
        mem_cooc_rank=mem_cooc_rank,
    )


def related_notes_for_query(
    *,
    query_vec: list[float] | None = None,
    query_text: str | None = None,
    embed_store: EmbedStore | None = None,
    cooccur_store: CooccurStore | None = None,
    memory_embed_store: EmbedStore | None = None,
    memory_cooccur_store: CooccurStore | None = None,
    k: int = 10,
    scope: str | None = None,
    exclude: set[str] | None = None,
    expand: bool = False,
    recall_rank: list[tuple[str, float]] | None = None,
) -> list[RelatedNote]:
    """Return the top-k notes related to a FRESH query (not an indexed note).

    The embed leg ranks against `query_vec`; the co-occurrence leg seeds its
    concept profile from `query_text`. This is the fusion path for routing an
    incoming concept (COLLISION) or autolinking a freshly-written note: either
    input may be omitted, and that leg abstains. Returns [] when both abstain.

    `memory_*_store` are the personal-memory lane (ADR-0019); see
    `related_notes`. The memory co-occurrence leg seeds from `query_text`
    using the MEMORY store's frozen language.
    """
    blocked = set(exclude or ())
    pool = max(k * 3, _POOL_MIN)

    embed_rank = _rank_embeddings_from_vec(embed_store, query_vec, k=pool, exclude=blocked)

    cooc_rank = None
    if cooccur_store is not None and query_text:
        profile = _profile_from_seeds(
            cooccur_store,
            _seed_from_text(query_text, cooccur_store.lang),
            scope=scope,
            expand=expand,
        )
        cooc_rank = _rank_cooccur_from_profile(
            cooccur_store, profile, k=pool, blocked=blocked, scope=scope
        )

    mem_embed_rank = _rank_embeddings_from_vec(
        memory_embed_store, query_vec, k=pool, exclude=set()
    )
    mem_cooc_rank = None
    if memory_cooccur_store is not None and query_text:
        profile = _profile_from_seeds(
            memory_cooccur_store,
            _seed_from_text(query_text, memory_cooccur_store.lang),
            scope=None,
            expand=expand,
        )
        mem_cooc_rank = _rank_cooccur_from_profile(
            memory_cooccur_store, profile, k=pool, blocked=set(), scope=None
        )
    return _fuse(
        embed_rank,
        cooc_rank,
        k=k,
        mem_embed_rank=mem_embed_rank,
        mem_cooc_rank=mem_cooc_rank,
        recall_rank=recall_rank,
    )
