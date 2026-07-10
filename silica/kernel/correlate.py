# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""CORRELATE (ADR-0013): prose note-to-note edges from co-occurrence contributions.

L1 kernel: no LLM, no API, no embedder. Pure math here (top-k stem selection +
Jaccard on stem sets); the store-coupled refresh lives below it. `note_edges` is
derived data, never source of truth — discrepancies are resolved by
recomputation, never repair.

Metric (measured, ADR-0013 gate 2026-07-09): Jaccard over each note's top-30
stems by RAW count, edge kept when >= tau. IDF was rejected by the data (it
made the top-k of a note a function of the whole corpus; raw count keeps a
note's row a pure function of that note alone).
"""
from __future__ import annotations

# ponytail: module constants; promote to CONFIG if a second vault ever needs different
_TOP_K = 30
_TAU = 0.25


def topk_set(nodes: dict[str, int], k: int = _TOP_K) -> frozenset[str]:
    """The k highest-count stems of one note, as a set. Tie-break lexicographic.

    `k` defaults to the module constant; production never passes it. It exists
    so a fixture can pin a small k instead of needing 30+ synthetic stems.
    """
    ranked = sorted(nodes.items(), key=lambda kv: (-kv[1], kv[0]))
    return frozenset(stem for stem, _count in ranked[:k])


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    if not union:
        return 0.0  # two empty top-k sets share nothing; 0.0, not a ZeroDivisionError
    return len(a & b) / len(union)


def _inverted_index(store) -> dict[str, set[str]]:
    """top-stem -> {note keys with that stem in their top-k}. In-memory only,
    never persisted; rebuilt in O(N*k) at refresh time (a scan of the store
    already in memory). Used to find edge candidates without an O(N) sweep.
    """
    idx: dict[str, set[str]] = {}
    for path in store.paths():
        for stem in topk_set(store.note_nodes(path)):
            idx.setdefault(stem, set()).add(path)
    return idx


def refresh_edges(store, paths: list[str]) -> None:
    """Recompute the note_edges rows of `paths` in place (does NOT save).

    For each touched note A: clear every existing edge that involves A, then
    among the notes sharing >=1 top-stem with A (via the inverted index,
    O(candidates*k) not O(N*k)) re-add the ones with Jaccard >= tau.

    Contributions are the input and are not mutated here, so the inverted index
    is stable across the loop. Edges are stored once under the ordered pair, so
    refreshing A also updates A's edges seen from a non-refreshed neighbour.
    """
    from silica.kernel.cooccurrence import cooccur_key

    idx = _inverted_index(store)
    for path in paths:
        key = cooccur_key(path)
        a_set = topk_set(store.note_nodes(key))
        store.clear_note_edges(key)
        if not a_set:
            continue
        candidates: set[str] = set()
        for stem in a_set:
            candidates |= idx.get(stem, set())
        candidates.discard(key)
        for other in candidates:
            score = jaccard(a_set, topk_set(store.note_nodes(other)))
            if score >= _TAU:
                store.set_note_edge(key, other, score)


def recompute_all_edges(store) -> None:
    """Rebuild every note_edges row from scratch (/cooccur --force).

    Refreshing all paths clears every edge (each touches some path) and rebuilds
    it, so this needs no separate wipe. Does NOT save — the caller flushes once.
    """
    refresh_edges(store, store.paths())
