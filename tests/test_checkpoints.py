"""Tests for the per-note checkpoint stack (interactive-patch undo)."""
from __future__ import annotations

import pytest

from silica.kernel.checkpoints import CheckpointStore


@pytest.fixture
def store(tmp_path):
    return CheckpointStore(tmp_path / "checkpoints.db")


def test_first_push_seeds_original_floor(store):
    depth = store.push("Concepts/AI.md", "ORIG", "P1")
    # floor (ORIG) + first patch result (P1)
    assert depth == 2
    assert store.depth("Concepts/AI.md") == 2


def test_subsequent_push_appends_only_result(store):
    p = "Concepts/AI.md"
    store.push(p, "ORIG", "P1")
    store.push(p, "P1", "P1P2")
    store.push(p, "P1P2", "P1P2P3")
    # ORIG, P1, P1P2, P1P2P3
    assert store.depth(p) == 4


def test_undo_walks_back_one_patch_at_a_time(store):
    p = "Concepts/AI.md"
    store.push(p, "ORIG", "P1")
    store.push(p, "P1", "P1P2")
    store.push(p, "P1P2", "P1P2P3")
    store.push(p, "P1P2P3", "P1P2P3P4")

    assert store.undo(p) == "P1P2P3"
    assert store.undo(p) == "P1P2"
    assert store.undo(p) == "P1"
    assert store.undo(p) == "ORIG"


def test_floor_is_never_removed(store):
    p = "Concepts/AI.md"
    store.push(p, "ORIG", "P1")
    assert store.undo(p) == "ORIG"   # back to original
    assert store.depth(p) == 1       # floor remains
    assert store.undo(p) is None     # nothing left to undo
    assert store.depth(p) == 1


def test_undo_unknown_path_returns_none(store):
    assert store.undo("Nope.md") is None


def test_most_recent_path_tracks_last_push(store):
    store.push("A.md", "a0", "a1")
    store.push("B.md", "b0", "b1")
    assert store.most_recent_path() == "B.md"
    store.push("A.md", "a1", "a2")
    assert store.most_recent_path() == "A.md"


def test_most_recent_path_empty_store(store):
    assert store.most_recent_path() is None


def test_independent_stacks_per_note(store):
    store.push("A.md", "a0", "a1")
    store.push("B.md", "b0", "b1")
    assert store.undo("A.md") == "a0"
    # B untouched
    assert store.depth("B.md") == 2
    assert store.undo("B.md") == "b0"


def test_persistence_across_instances(tmp_path):
    db = tmp_path / "checkpoints.db"
    s1 = CheckpointStore(db)
    s1.push("A.md", "a0", "a1")
    s1.push("A.md", "a1", "a2")
    # New instance reads the same file
    s2 = CheckpointStore(db)
    assert s2.depth("A.md") == 3
    assert s2.undo("A.md") == "a1"
    assert s2.most_recent_path() == "A.md"


def test_clear_drops_all_for_path(store):
    store.push("A.md", "a0", "a1")
    store.push("B.md", "b0", "b1")
    store.clear("A.md")
    assert store.depth("A.md") == 0
    assert store.depth("B.md") == 2
