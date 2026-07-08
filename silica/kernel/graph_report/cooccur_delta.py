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
    from silica.kernel.cooccurrence import get_cooccur_store, tokenize
    from silica.kernel.relatedness import _cooccur_ranking

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

    # --- AUTOLINK: co-occurrence-related but not wikilinked (and >2 hops) ----
    autolinks: list[AutolinkCandidate] = []
    seen: set[tuple[str, str]] = set()
    for nid in store.paths():
        if nid not in G_und:
            continue
        ranking = _cooccur_ranking(store, nid, k=k, exclude=set(), scope=scope, expand=True)
        for tgt, weight in ranking or []:
            if tgt not in G_und:
                continue
            try:
                if nx.shortest_path_length(G_und, nid, tgt) <= 2:
                    continue  # already linked or trivially close
            except Exception:
                pass  # no path -> genuinely disconnected -> a valid candidate
            key = (min(nid, tgt), max(nid, tgt))
            if key in seen:
                continue
            seen.add(key)
            autolinks.append(AutolinkCandidate(
                source=key[0], target=key[1],
                weight=round(float(weight), 2),
                shared=_shared_labels(nid, tgt),
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
    god_ids = [n.id for n in report.god_nodes]
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

    autolinks.sort(key=lambda a: (-(a.convergence * a.weight), -a.weight, a.source, a.target))
    autolinks = autolinks[:k]

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
