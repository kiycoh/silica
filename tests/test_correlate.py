# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Pure math of CORRELATE (ADR-0013): top-k stem selection + Jaccard.

These functions are the metric the spec's admission gate measured
(top-30 by raw count, Jaccard, tau=0.25). They know nothing about the store —
they operate on plain {stem: count} maps and frozensets.
"""
from __future__ import annotations

import pytest

from silica.kernel.correlate import (
    jaccard,
    recompute_all_edges,
    refresh_edges,
    topk_set,
)
from silica.kernel.cooccurrence import CooccurStore


def _nodes(**counts: int) -> dict:
    """A synthetic contribution with exactly these {stem: count} nodes.

    Bypasses the tokenizer so a fixture controls the top-k sets precisely.
    """
    return {"nodes": {s: {"label": s, "count": c} for s, c in counts.items()}, "edges": []}


def test_topk_set_keeps_highest_counts() -> None:
    nodes = {"quick": 10, "sort": 8, "boilerplate": 1}
    assert topk_set(nodes, k=2) == frozenset({"quick", "sort"})


def test_topk_set_tie_break_is_lexicographic() -> None:
    # same count: the lexicographically smaller stem wins the cut, so it is
    # deterministic across runs and machines (no dict-order leakage).
    nodes = {"beta": 5, "alpha": 5, "gamma": 5}
    assert topk_set(nodes, k=2) == frozenset({"alpha", "beta"})


def test_topk_set_returns_all_when_fewer_than_k() -> None:
    nodes = {"quick": 3, "sort": 2}
    assert topk_set(nodes, k=30) == frozenset({"quick", "sort"})


def test_topk_set_of_empty_is_empty() -> None:
    assert topk_set({}) == frozenset()


def test_jaccard_is_intersection_over_union() -> None:
    a = frozenset({"quick", "sort", "array"})
    b = frozenset({"quick", "sort", "tree"})
    # |{quick, sort}| / |{quick, sort, array, tree}| = 2 / 4
    assert jaccard(a, b) == 0.5


def test_jaccard_of_empty_sets_is_zero_not_a_crash() -> None:
    # a note with no stems must not blow up the metric with a ZeroDivisionError.
    assert jaccard(frozenset(), frozenset()) == 0.0


def test_jaccard_disjoint_is_zero_identical_is_one() -> None:
    a = frozenset({"quick", "sort"})
    assert jaccard(a, frozenset({"tree", "heap"})) == 0.0
    assert jaccard(a, a) == 1.0


def test_refresh_creates_edge_for_overlapping_notes() -> None:
    store = CooccurStore()
    store.upsert_note("a", _nodes(quick=1, sort=1, array=1, tree=1))
    store.upsert_note("b", _nodes(quick=1, sort=1, array=1, heap=1))
    # top-k(a)={quick,sort,array,tree}, top-k(b)={quick,sort,array,heap}
    # jaccard = |3 shared| / |5 union| = 0.6 >= tau
    refresh_edges(store, ["a", "b"])
    assert store.note_edges_for("a") == {"b": pytest.approx(0.6)}


def test_refresh_no_edge_below_tau() -> None:
    store = CooccurStore()
    store.upsert_note("a", _nodes(quick=1, sort=1, array=1))
    store.upsert_note("b", _nodes(bread=1, flour=1, yeast=1))
    refresh_edges(store, ["a", "b"])
    assert store.note_edges_for("a") == {}


def test_refresh_is_local_to_touched_paths() -> None:
    store = CooccurStore()
    store.upsert_note("a", _nodes(quick=1, sort=1, array=1, tree=1))
    store.upsert_note("b", _nodes(quick=1, sort=1, array=1, heap=1))
    store.upsert_note("c", _nodes(bread=1, flour=1, yeast=1, water=1))
    store.upsert_note("d", _nodes(bread=1, flour=1, yeast=1, salt=1))
    recompute_all_edges(store)
    assert store.note_edges_for("c") == {"d": pytest.approx(0.6)}
    # rewrite A to be disjoint, refresh ONLY A: A's row changes, C-D untouched
    store.upsert_note("a", _nodes(xxx=1, yyy=1))
    refresh_edges(store, ["a"])
    assert store.note_edges_for("a") == {}
    assert store.note_edges_for("b") == {}, "the a-b edge must be gone after A diverged"
    assert store.note_edges_for("c") == {"d": pytest.approx(0.6)}, "C-D is untouched by an A refresh"


def test_refresh_is_idempotent() -> None:
    store = CooccurStore()
    store.upsert_note("a", _nodes(quick=1, sort=1, array=1, tree=1))
    store.upsert_note("b", _nodes(quick=1, sort=1, array=1, heap=1))
    refresh_edges(store, ["a", "b"])
    once = store.note_edges_for("a")
    refresh_edges(store, ["a", "b"])
    assert store.note_edges_for("a") == once


def test_recompute_all_builds_the_whole_graph() -> None:
    store = CooccurStore()
    store.upsert_note("a", _nodes(quick=1, sort=1, array=1, tree=1))
    store.upsert_note("b", _nodes(quick=1, sort=1, array=1, heap=1))
    store.upsert_note("c", _nodes(bread=1, flour=1, yeast=1))
    recompute_all_edges(store)
    assert store.note_edges_for("a") == {"b": pytest.approx(0.6)}
    assert store.note_edges_for("c") == {}
