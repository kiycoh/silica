# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""reading_path — the BFS core behind /path (wikilinks + cooccur union)."""
from __future__ import annotations

import networkx as nx

from silica.kernel.mindmap import reading_path


class _Store:
    """CooccurStore stand-in: {min-endpoint (no .md): {other: score}}."""

    def __init__(self, edges: dict[str, dict[str, float]]):
        self._e = edges

    def note_edges_for(self, path: str) -> dict[str, float]:
        key = path.removesuffix(".md")
        out = dict(self._e.get(key, {}))
        for lo, nbrs in self._e.items():
            if lo != key and key in nbrs:
                out[lo] = nbrs[key]
        return out


def _graph(*edges: tuple[str, str]) -> nx.Graph:
    g = nx.Graph()
    g.add_edges_from(edges)
    return g


def test_wikilink_only_path():
    g = _graph(("a.md", "b.md"), ("b.md", "c.md"))
    got = reading_path("a.md", "c.md", graph=g, cooccur_store=_Store({}))
    assert got == [("a.md", "start"), ("b.md", "wikilink"), ("c.md", "wikilink")]


def test_cooccur_bridges_disconnected_wikilinks():
    g = _graph(("a.md", "b.md"))
    got = reading_path("a.md", "c.md", graph=g, cooccur_store=_Store({"b": {"c": 0.5}}))
    assert got == [("a.md", "start"), ("b.md", "wikilink"), ("c.md", "cooccur")]


def test_cooccur_reverse_direction_edge_is_walkable():
    # Edge stored with c as the min endpoint must still be found from b.
    g = _graph(("a.md", "b.md"))
    got = reading_path("a.md", "c.md", graph=g, cooccur_store=_Store({"c": {"b": 0.5}}))
    assert got == [("a.md", "start"), ("b.md", "wikilink"), ("c.md", "cooccur")]


def test_shortest_wins_and_wikilink_labels_shared_edges():
    # Direct wikilink a—c beats the two-hop route; a—c also has a cooccur edge,
    # and the wikilink leg must win the label.
    g = _graph(("a.md", "b.md"), ("b.md", "c.md"), ("a.md", "c.md"))
    got = reading_path("a.md", "c.md", graph=g, cooccur_store=_Store({"a": {"c": 0.9}}))
    assert got == [("a.md", "start"), ("c.md", "wikilink")]


def test_disconnected_returns_none():
    g = _graph(("a.md", "b.md"), ("x.md", "y.md"))
    assert reading_path("a.md", "y.md", graph=g, cooccur_store=_Store({})) is None
