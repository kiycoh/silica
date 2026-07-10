# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Co-occurrence vs wikilink delta — PROPOSED, embedder-free signals.

AUTOLINK (co-occurrence − wikilink), STALE (wikilink − co-occurrence) and
MISSING HUB (central concept with no hub note). This is the designated
landing zone for the ADR-0013 CORRELATE wiring: changes to the delta logic
must not leak into compute.py or render.py.
"""
from __future__ import annotations

import logging
from typing import Any

from silica.kernel.graph_report.models import (
    AutolinkCandidate,
    MissingHub,
    StaleLink,
    VaultReport,
)

logger = logging.getLogger(__name__)


def _compute_cooccur_delta(
    report: VaultReport,
    G_und: Any,
    node_label: dict[str, str],
    *,
    cooccur_store: Any | None = None,
    k: int = 10,
) -> tuple[list[AutolinkCandidate], list[StaleLink], list[MissingHub]]:
    """Delta between the co-occurrence graph and the wikilink graph.

    Three PROPOSED, embedder-free signals (no network, pure local compute):

      - AUTOLINK  (co-occurrence − wikilink): note pairs the author co-mentions
        in text but never wikilinked, more than 2 hops apart.
      - STALE     (wikilink − co-occurrence): wikilinked pairs whose notes share
        no concepts in text — a structural link without textual co-presence.
      - MISSING HUB (centrality − hub): a concept central in the discourse for
        which no note is titled — the next hub note to create.

    `cooccur_store` is injectable for testing; loaded from disk when None.
    Returns empty lists when the index is empty (best-effort, never raises).
    """
    import networkx as nx
    from silica.kernel.cooccurrence import cooccur_key, get_cooccur_store, tokenize
    from silica.kernel.relatedness import _concept_idf, _cooccur_ranking

    try:
        store = cooccur_store if cooccur_store is not None else get_cooccur_store()
    except Exception as exc:
        logger.debug("graph_report: co-occurrence index unavailable (%s)", exc)
        return [], [], []
    if len(store) == 0:
        return [], [], []

    scope = report.scope or None

    def _shared_labels(a: str, b: str) -> list[str]:
        na, nb = store.note_nodes(a), store.note_nodes(b)
        return sorted(store.node_label(s) for s in (set(na) & set(nb)))

    # Direct-edge evidence orders shared stems by IDF descending (top 5), so the
    # boilerplate-template stems a raw-count metric admits sink below the
    # discriminative ones. IDF lives ONLY here (display), never in the metric.
    # Computed lazily on the first direct edge — one O(N) pass at render.
    idf_map: dict[str, float] | None = None

    def _shared_by_idf(a: str, b: str) -> list[str]:
        nonlocal idf_map
        if idf_map is None:
            all_stems: set[str] = set()
            for p in store.paths():
                all_stems |= set(store.note_nodes(p))
            idf_map = _concept_idf(store, all_stems, scope=scope)
        shared = set(store.note_nodes(a)) & set(store.note_nodes(b))
        ranked = sorted(shared, key=lambda s: (-idf_map.get(s, 0.0), s))
        return [store.node_label(s) for s in ranked[:5]]

    # Keyspace bridge: G_und node ids are graph paths WITH '.md'; store keys are
    # stripped (cooccur_key). Membership and hop checks must cross that boundary
    # here — on a real vault a raw `nid in G_und` matches nothing and the whole
    # AUTOLINK section silently comes out empty.
    gid_by_key = {cooccur_key(n): n for n in G_und.nodes}

    # --- AUTOLINK: direct note_edges (CORRELATE) UNION expanded ranking, both
    #     >2 hops away and not already wikilinked. Direct pairs are a lookup
    #     (free) and win provenance when a pair appears in both legs.
    #     Candidates keep STORE keys (stripped): the cosine-band filter and the
    #     shared-concept evidence below consume them, and render strips anyway.
    autolinks: list[AutolinkCandidate] = []
    seen: set[tuple[str, str]] = set()
    for nid in store.paths():
        src_gid = gid_by_key.get(nid)
        if src_gid is None:
            continue
        direct = store.note_edges_for(nid)  # {tgt: jaccard}, both directions
        expanded = _cooccur_ranking(store, nid, k=k, exclude=set(), scope=scope, expand=True) or []
        legs = (
            [(tgt, w, "direct") for tgt, w in direct.items()]
            + [(tgt, w, "expanded") for tgt, w in expanded]
        )
        for tgt, weight, provenance in legs:
            tgt_gid = gid_by_key.get(tgt)
            if tgt_gid is None:
                continue
            try:
                if nx.shortest_path_length(G_und, src_gid, tgt_gid) <= 2:
                    continue  # already linked or trivially close
            except Exception:
                pass  # no path -> genuinely disconnected -> a valid candidate
            key = (min(nid, tgt), max(nid, tgt))
            if key in seen:
                continue
            seen.add(key)
            shared = _shared_by_idf(nid, tgt) if provenance == "direct" else _shared_labels(nid, tgt)
            autolinks.append(AutolinkCandidate(
                source=key[0], target=key[1],
                weight=round(float(weight), 2),
                shared=shared,
                provenance=provenance,
            ))

    # --- #6 cosine-band: filter trivially-similar or nonsensically-distant ---
    # Paper (Marwitz 2026) S_own×other^filtered: removes pairs whose semantic
    # similarity is too high (trivial, A2 in expert ratings) or too low
    # (nonsensical, B in expert ratings). Best-effort: skipped silently when
    # embeddings are unavailable.
    try:
        from silica.kernel.embed import get_store, _cosine
        _embed_store = get_store()
        if len(_embed_store) > 0:
            _cos_hi = 0.92
            _cos_lo = 0.35
            filtered: list[AutolinkCandidate] = []
            for a in autolinks:
                v_src = _embed_store.get_vec(a.source)
                v_tgt = _embed_store.get_vec(a.target)
                if v_src and v_tgt:
                    cos = _cosine(v_src, v_tgt)
                    if cos > _cos_hi or cos < _cos_lo:
                        continue  # too trivial or too alien
                filtered.append(a)
            autolinks = filtered
    except Exception:
        pass  # embeddings unavailable → no filtering, degrade gracefully

    # --- #8 convergence: S_(many_own)×other --------------------------------
    # Paper (Marwitz 2026, Table 2): the highest-interest section connects a
    # candidate to MANY of the researcher's own concepts. Silica's "own
    # concepts" are the god-node hubs; a candidate touching more hubs (either
    # endpoint co-occurring with the hub, or being a hub itself) earns a higher
    # convergence and is ranked by convergence × weight. Degrades to the prior
    # weight-only ordering when there are no god nodes.
    # Same keyspace bridge: god-node ids are graph paths, candidates carry store
    # keys — normalise once so the hub-exclusion and reach comparisons line up.
    god_ids = [cooccur_key(n.id) for n in report.god_nodes]
    if god_ids:
        god_set = set(god_ids)
        # expand=False: count only DIRECT concept overlap with the hub, so
        # convergence measures genuine reach into distinct hubs rather than
        # transitive bleed through a single shared concept.
        god_related: dict[str, set[str]] = {}
        for g in god_ids:
            ranking = _cooccur_ranking(store, g, k=50, exclude=set(), scope=scope, expand=False)
            god_related[g] = {p for p, _w in (ranking or [])}
        for a in autolinks:
            # The "other" endpoint(s) are those not themselves hubs; convergence
            # counts how many distinct hubs that other concept reaches into.
            others = [e for e in (a.source, a.target) if e not in god_set]
            a.convergence = sum(
                1 for g in god_ids
                if any(o in god_related[g] for o in others)
            )

    # Per-leg quota: direct weights are Jaccard (<=1) while expanded overlaps run
    # to ~1e6 on a real vault — one mixed sort would bury every direct candidate
    # (the high-precision leg CORRELATE exists for). Direct gets up to half the
    # slots ranked by its native Jaccard; expanded keeps the convergence ranking
    # for the rest; either leg backfills when the other runs short.
    direct_leg = sorted(
        (a for a in autolinks if a.provenance == "direct"),
        key=lambda a: (-a.weight, a.source, a.target),
    )
    expanded_leg = sorted(
        (a for a in autolinks if a.provenance != "direct"),
        key=lambda a: (-(a.convergence * a.weight), -a.weight, a.source, a.target),
    )
    take = min(len(direct_leg), k - k // 2)
    autolinks = direct_leg[:take] + expanded_leg[: k - take]
    if len(autolinks) < k:
        autolinks += direct_leg[take: take + k - len(autolinks)]

    # --- STALE: wikilinked but the two notes share no concepts --------------
    stale: list[StaleLink] = []
    for u, v in G_und.edges():
        if not store.note_nodes(u) or not store.note_nodes(v):
            continue  # a note with no concepts can't be assessed -> don't flag
        if not _shared_labels(u, v):
            stale.append(StaleLink(source=min(u, v), target=max(u, v)))
    stale.sort(key=lambda s: (s.source, s.target))
    stale = stale[:k]

    # --- MISSING HUB: central concept with no note titled after it ----------
    Gc = store.to_networkx(scope=scope)
    titled_stems: set[str] = set()
    for label in node_label.values():
        for sentence in tokenize(label, stem_lang=store.lang, stopword_lang=store.lang):
            titled_stems.update(stem for stem, _surface in sentence)

    hubs: list[MissingHub] = []
    for stem in Gc.nodes():
        if stem in titled_stems:
            continue  # a hub note already formalises this concept
        wdeg = sum(d.get("weight", 0.0) for _u, _v, d in Gc.edges(stem, data=True))
        hubs.append(MissingHub(concept=store.node_label(stem), centrality=round(wdeg, 2)))
    hubs.sort(key=lambda h: (-h.centrality, h.concept))
    hubs = hubs[:k]

    return autolinks, stale, hubs
