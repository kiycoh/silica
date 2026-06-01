"""TDD tests for WS1 — CLI graph snapshot keying (path-canonical, C1.2/C1.3).

Written BEFORE the implementation. All tests should be RED until cli_backend.py
incremental snapshot is refactored.
C1.1 (bulk resolvedLinks read via Obsidian eval/CDP) is implemented as Spike S1.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import networkx as nx

from silica.driver.base import GraphSnapshot, NoteRef, Link


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ref(path: str) -> NoteRef:
    name = path.rsplit("/", 1)[-1].removesuffix(".md")
    return NoteRef(name=name, path=path)


def _mock_cli_backend(vault: Path):
    """Return a configured ObsidianCLIBackend with Obsidian calls patched out."""
    from silica.driver.cli_backend import ObsidianCLIBackend

    backend = ObsidianCLIBackend.__new__(ObsidianCLIBackend)
    backend.vault_path = vault
    backend._is_graph_built = True   # skip _ensure_graph rebuild
    backend._graph = MagicMock()
    backend._unresolved_links = set()
    backend._notes: dict[str, NoteRef] = {}
    backend._notes_by_name: dict[str, list[NoteRef]] = {}
    return backend


# ---------------------------------------------------------------------------
# C1.2 — Incremental snapshot is path-keyed
# ---------------------------------------------------------------------------

def test_cli_incremental_snapshot_is_path_keyed(tmp_path):
    """link_counts and backlink_counts in an incremental snapshot use path-canonical keys."""
    from silica.driver.cli_backend import ObsidianCLIBackend

    backend = _mock_cli_backend(tmp_path)

    # Seed internal state: one note at "Concepts/Backpropagation.md"
    ref_bp = _make_ref("Concepts/Backpropagation.md")
    backend._notes = {"Concepts/Backpropagation.md": ref_bp}
    backend._unresolved_links = set()

    # Graph: Backpropagation.md has 2 outgoing edges, 1 incoming
    g = MagicMock()
    g.__contains__ = lambda self, x: x == "Concepts/Backpropagation.md"
    g.successors.return_value = ["Hub/Concepts.md", "Concepts/Gradient.md"]
    g.predecessors.return_value = ["Hub/Concepts.md"]
    g.out_degree.return_value = 2
    g.in_degree.return_value = 1
    backend._graph = g

    snap: GraphSnapshot = backend.graph_snapshot([ref_bp])

    # Key must be canonical path: "Concepts/Backpropagation" (no .md)
    assert "Concepts/Backpropagation" in snap.link_counts, (
        f"Expected path-canonical key 'Concepts/Backpropagation' in link_counts, "
        f"got: {list(snap.link_counts.keys())}"
    )
    assert snap.link_counts["Concepts/Backpropagation"] == 2
    assert "Concepts/Backpropagation" in snap.backlink_counts
    assert snap.backlink_counts["Concepts/Backpropagation"] == 1

    # Old name-based key must NOT be present
    assert "Backpropagation" not in snap.link_counts, (
        "Incremental snapshot must not use bare name as key (path-keyed only)"
    )


# ---------------------------------------------------------------------------
# B/Cell.md duplicate test
# ---------------------------------------------------------------------------

def test_duplicate_basename_distinct_keys_cli(tmp_path):
    """Two notes with the same basename in different folders produce distinct snapshot keys."""
    from silica.driver.cli_backend import ObsidianCLIBackend

    backend = _mock_cli_backend(tmp_path)

    ref_a = _make_ref("A/Cell.md")
    ref_b = _make_ref("B/Cell.md")
    backend._notes = {
        "A/Cell.md": ref_a,
        "B/Cell.md": ref_b,
    }

    g = MagicMock()
    def contains(path):
        return path in ("A/Cell.md", "B/Cell.md")
    g.__contains__ = lambda self, x: contains(x)

    def out_degree(p):
        return 1 if p == "A/Cell.md" else 0
    def in_degree(p):
        return 0 if p == "A/Cell.md" else 1
    g.out_degree.side_effect = out_degree
    g.in_degree.side_effect = in_degree
    g.successors.return_value = []
    g.predecessors.return_value = []
    backend._graph = g

    snap = backend.graph_snapshot([ref_a, ref_b])

    assert "A/Cell" in snap.link_counts, (
        f"Expected 'A/Cell' in link_counts, got: {list(snap.link_counts.keys())}"
    )
    assert "B/Cell" in snap.link_counts, (
        f"Expected 'B/Cell' in link_counts, got: {list(snap.link_counts.keys())}"
    )
    assert snap.link_counts["A/Cell"] == 1
    assert snap.link_counts["B/Cell"] == 0


# ---------------------------------------------------------------------------
# C1.3 — Unresolved link detected in incremental snapshot
# ---------------------------------------------------------------------------

def test_unresolved_link_detected_percettrone(tmp_path):
    """Unresolved links from the synthetic vault are captured in incremental snapshot."""
    from silica.driver.cli_backend import ObsidianCLIBackend

    backend = _mock_cli_backend(tmp_path)

    ref_p = _make_ref("Concepts/Perceptron.md")
    backend._notes = {"Concepts/Perceptron.md": ref_p}
    backend._unresolved_links = {("Concepts/Perceptron.md", "MissingNote")}

    g = MagicMock()
    g.__contains__ = lambda self, x: x == "Concepts/Perceptron.md"
    g.out_degree.return_value = 1  # resolved links
    g.in_degree.return_value = 0
    g.successors.return_value = ["Hub/Concepts.md"]
    g.predecessors.return_value = []
    backend._graph = g

    snap = backend.graph_snapshot([ref_p])

    assert any(
        lnk.target == "MissingNote" or "MissingNote" in lnk.target
        for lnk in snap.unresolved
    ), f"Expected unresolved 'MissingNote', got: {snap.unresolved}"


# ---------------------------------------------------------------------------
# C1.2 — Parity: incremental snapshot keys match full snapshot keys
# ---------------------------------------------------------------------------

def test_parity_incremental_snapshot_with_duplicates(tmp_path):
    """Incremental and full snapshots for the same notes produce the same key names."""
    from silica.driver.cli_backend import ObsidianCLIBackend
    import networkx as nx

    backend = _mock_cli_backend(tmp_path)

    ref_a = _make_ref("A/Cell.md")
    ref_b = _make_ref("B/Cell.md")
    backend._notes = {
        "A/Cell.md": ref_a,
        "B/Cell.md": ref_b,
    }
    backend._unresolved_links = set()

    # Build a real graph so both code paths read the same structure
    g = nx.DiGraph()
    g.add_node("A/Cell.md", ref=ref_a)
    g.add_node("B/Cell.md", ref=ref_b)
    g.add_edge("A/Cell.md", "B/Cell.md")
    backend._graph = g
    backend._graph_ready = True

    full_snap = backend.graph_snapshot(None)
    incr_snap = backend.graph_snapshot([ref_a, ref_b])

    # Keys present in incr must exist in full and match
    for key in incr_snap.link_counts:
        assert key in full_snap.link_counts, (
            f"Incremental key '{key}' not found in full snapshot"
        )
        assert incr_snap.link_counts[key] == full_snap.link_counts[key], (
            f"Count mismatch for '{key}': incr={incr_snap.link_counts[key]} "
            f"full={full_snap.link_counts[key]}"
        )


# ---------------------------------------------------------------------------
# C1.3 — Graph gate verdict is unchanged after the path-key refactor
# ---------------------------------------------------------------------------

def test_graph_gate_verdict_unchanged_post_refactor(tmp_path):
    """check_graph_regression with path-keyed pre/post graphs gives the same verdict."""
    from silica.kernel.graph_diff import check_graph_regression

    ref_a = _make_ref("notes/Alpha.md")
    ref_b = _make_ref("notes/Beta.md")

    pre = GraphSnapshot(
        orphans=[ref_b],
        unresolved=[],
        link_counts={"notes/Alpha": 1, "notes/Beta": 0},
        backlink_counts={"notes/Alpha": 0, "notes/Beta": 1},
    )
    post = GraphSnapshot(
        orphans=[ref_b],
        unresolved=[],
        link_counts={"notes/Alpha": 1, "notes/Beta": 0},
        backlink_counts={"notes/Alpha": 0, "notes/Beta": 1},
    )

    success, errors = check_graph_regression(pre, post, created_paths=[])
    assert success, f"Regression check should pass for identical snapshots. Errors: {errors}"


# ---------------------------------------------------------------------------
# C1.1 — Bulk read from resolvedLinks (Spike S1 — implemented)
# ---------------------------------------------------------------------------

def test_cli_graph_reads_resolved_links(tmp_path):
    """_load_graph_from_obsidian() reads resolvedLinks in bulk via a single CDP eval."""
    from silica.driver.cli_backend import ObsidianCLIBackend
    import json

    backend = ObsidianCLIBackend.__new__(ObsidianCLIBackend)
    backend.vault_path = tmp_path
    backend._vault_name = ""

    # Seed two notes so the graph-population logic has paths to match
    ref_a = _make_ref("Concepts/Alpha.md")
    ref_b = _make_ref("Concepts/Beta.md")
    backend._graph = nx.DiGraph()
    backend._graph.add_node("Concepts/Alpha.md", ref=ref_a)
    backend._graph.add_node("Concepts/Beta.md", ref=ref_b)
    backend._notes = {
        "Concepts/Alpha.md": ref_a,
        "Concepts/Beta.md": ref_b,
    }
    backend._unresolved_links = set()
    backend._mention_index = {}

    # Mock the CLI eval to return metadataCache-shaped JSON
    fake_metadata = {
        "resolved": {
            "Concepts/Alpha.md": {"Concepts/Beta.md": 1},
        },
        "unresolved": {
            "Concepts/Alpha.md": {"GhostNote": 1},
        },
    }
    with patch.object(backend, "_run_cli", return_value=json.dumps(fake_metadata)):
        result = backend._load_graph_from_obsidian()

    assert result is not None, "_load_graph_from_obsidian must return a non-None graph"
    # Resolved edge was added
    assert backend._graph.has_edge("Concepts/Alpha.md", "Concepts/Beta.md"), (
        "Expected resolved edge Alpha→Beta in graph"
    )
    # Unresolved link was recorded
    assert ("Concepts/Alpha.md", "GhostNote") in backend._unresolved_links, (
        "Expected unresolved link Alpha→GhostNote"
    )
