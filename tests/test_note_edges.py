# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""note_edges section of CooccurStore (CORRELATE / ADR-0013).

Edges are DERIVED data, stored once under the ordered pair (min, max), looked
up in both directions. The store owns document integrity: prune edges on
delete_note, prune orphans (endpoint with no contribution) on load.
"""
from __future__ import annotations

from silica.kernel.cooccurrence import CooccurStore, build_contribution


def test_set_and_read_edge_in_both_directions() -> None:
    store = CooccurStore()
    store.set_note_edge("a", "b", 0.5)
    assert store.note_edges_for("a") == {"b": 0.5}
    assert store.note_edges_for("b") == {"a": 0.5}


def test_edge_is_stored_once_regardless_of_argument_order() -> None:
    store = CooccurStore()
    store.set_note_edge("b", "a", 0.4)  # reversed args
    store.set_note_edge("a", "b", 0.9)  # same pair, updated score
    assert store.note_edges_for("a") == {"b": 0.9}
    assert store.note_edges_for("b") == {"a": 0.9}


def test_edges_survive_save_and_load(tmp_path) -> None:
    p = tmp_path / "cooccurrence.json"
    store = CooccurStore(path=p)
    store.upsert_note("a", build_contribution("A", "quick sort partitions the array"))
    store.upsert_note("b", build_contribution("B", "merge sort splits the array"))
    store.set_note_edge("a", "b", 0.5)
    store.save()
    reloaded = CooccurStore(path=p)
    assert reloaded.note_edges_for("a") == {"b": 0.5}


def test_orphan_edges_are_pruned_on_load(tmp_path) -> None:
    # A writer deleted note "b" from contributions but never touched edges.
    # note_edges is derived, so the dangling a<->b edge is dropped at load.
    p = tmp_path / "cooccurrence.json"
    store = CooccurStore(path=p)
    store.upsert_note("a", build_contribution("A", "quick sort partitions the array"))
    store.set_note_edge("a", "b", 0.5)  # "b" has no contribution
    store.save()
    reloaded = CooccurStore(path=p)
    assert reloaded.note_edges_for("a") == {}, "edge to a contribution-less note must be pruned"


def test_delete_note_prunes_its_edges() -> None:
    store = CooccurStore()
    store.upsert_note("a", build_contribution("A", "quick sort partitions the array"))
    store.upsert_note("b", build_contribution("B", "merge sort splits the array"))
    store.set_note_edge("a", "b", 0.5)
    store.delete_note("a")
    assert store.note_edges_for("b") == {}, "deleting a note must drop edges that touch it"


def test_clear_note_edges_removes_both_directions() -> None:
    store = CooccurStore()
    store.set_note_edge("a", "b", 0.5)
    store.set_note_edge("b", "c", 0.6)  # b is min here, c is max
    store.clear_note_edges("b")
    assert store.note_edges_for("a") == {}
    assert store.note_edges_for("b") == {}
    assert store.note_edges_for("c") == {}
