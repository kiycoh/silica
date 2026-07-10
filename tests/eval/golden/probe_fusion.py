# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""probe_fusion — masked-pair recovery through the FULL relatedness facade.

Same ground truth as probe_correlate (human body-wikilink pairs, masked, >2-hop
eligibility), different question: does ``related_notes()`` — the real fused
ranking (RRF over embed + cooccur + note_edges, with abstention) — surface the
masked counterpart in its top-k? probe_correlate measures candidate POPULATIONS
per leg; this probe is the only place the fusion itself (pool sizing,
``_rrf_fuse``, third-leg wiring) is exercised end-to-end, so
``fusion.recall_at_10`` is the fusion regression gate.

Tier-adaptive with no tier code: for INDEXED notes the embed leg is a pure
index lookup + cosine (no API call), so the caller passes whatever EmbedStore
exists on disk — absent or empty, the leg abstains per the facade contract and
the probe measures cooccur+edges fusion. ``legs`` reports what was actually
live so the runner can refuse baseline comparison across leg drift;
``embed_coverage`` (fraction of evaluated notes with a stored vector) exposes
stale or key-mismatched embedding indexes that would otherwise read as a
recall drop.

Masking caveat (shared with probe_correlate): the wikilink's surface text stays
in the note body, so recall is an optimistic ceiling — a regression gate, not
an absolute quality claim.
"""
from __future__ import annotations

from tests.eval.golden.probe_correlate import _pair, _wikilink_graph

# Facade depth measured; matches the k of every production related_notes surface.
# The metric name pins it — change both together or not at all.
K = 10

_EMPTY = {
    "pairs_evaluated": 0,
    "recall_at_10": 0.0,
    "mrr": 0.0,
    "embed_coverage": 0.0,
    "legs": "",
}


def _eligible_pairs(adj: dict[str, set[str]]) -> list[tuple[str, str]]:
    """Unordered wikilinked pairs that stay >2 hops apart once masked.

    Same filter as probe_correlate: a shared neighbour leaves a 2-hop path
    (via the hub) after masking, so the pair is not a fair recovery target.
    """
    eligible: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for a, nbrs in adj.items():
        for b in nbrs:
            p = _pair(a, b)
            if p in seen:
                continue
            seen.add(p)
            if (adj.get(p[0], set()) - {p[1]}) & (adj.get(p[1], set()) - {p[0]}):
                continue
            eligible.append(p)
    return eligible


def run(vault, store, *, embed_store=None, k: int = K, verbose: bool = False) -> dict:
    from silica.kernel import correlate
    from silica.kernel.relatedness import related_notes

    es = embed_store if (embed_store is not None and len(embed_store)) else None
    legs = ("embed+" if es is not None else "") + ("cooccur+edges" if len(store) else "")

    if len(store) == 0:
        return {**_EMPTY, "legs": legs}

    # Self-contained: derive note_edges from the current contributions
    # (idempotent — probe order in the runner must not matter).
    correlate.recompute_all_edges(store)

    eligible = _eligible_pairs(_wikilink_graph(vault, store))
    if not eligible:
        return {**_EMPTY, "legs": legs}

    # One facade call per unique endpoint, not per pair — pairs share notes.
    endpoints = sorted({e for pr in eligible for e in pr})
    topk = {
        key: [r.path for r in related_notes(key, embed_store=es, cooccur_store=store, k=k)]
        for key in endpoints
    }

    covered = 0
    if es is not None:
        for key in endpoints:  # mirror _embed_ranking's exact-then-stripped lookup
            if es.get_vec(key) is not None or es.get_vec(key.removesuffix(".md")) is not None:
                covered += 1

    hits = 0
    rr_sum = 0.0
    for a, b in eligible:
        ranks = []
        if b in topk[a]:
            ranks.append(topk[a].index(b) + 1)
        if a in topk[b]:
            ranks.append(topk[b].index(a) + 1)
        if ranks:  # recovered from either direction, best rank feeds MRR
            hits += 1
            rr_sum += 1.0 / min(ranks)

    n = len(eligible)
    res = {
        "pairs_evaluated": n,
        "recall_at_10": round(hits / n, 4),
        "mrr": round(rr_sum / n, 4),
        "embed_coverage": round(covered / len(endpoints), 4) if es is not None else 0.0,
        "legs": legs,
    }
    if verbose:
        print(f"\nfusion[{legs}]: recall@{k} {hits}/{n} = {res['recall_at_10']:.1%}, "
              f"mrr {res['mrr']:.3f}, embed coverage {res['embed_coverage']:.1%}")
    return res
