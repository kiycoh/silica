"""Layout invariants for kernel/mindmap — deterministic radial wedge placement."""
from __future__ import annotations

import math

import networkx as nx
import pytest

from silica.kernel.mindmap import (
    BOX_H,
    BOX_W,
    MapMaterials,
    build_mapview,
)


def _materials(latent=()):
    """root — a,b (hop1); a — c, b — d (hop2). Two communities.  Plus latent."""
    g = nx.Graph()
    g.add_edges_from([
        ("root.md", "a.md"), ("root.md", "b.md"),
        ("a.md", "c.md"), ("b.md", "d.md"),
    ])
    return MapMaterials(
        graph=g,
        titles={p: p[:-3].upper() for p in g.nodes},
        community_of={"root.md": 0, "a.md": 0, "c.md": 0, "b.md": 1, "d.md": 1},
        latent=list(latent),
    )


def _boxes_overlap(a, b) -> bool:
    # Equal axis-aligned boxes centred at a, b overlap iff both axes are within a box.
    return abs(a.x - b.x) < BOX_W and abs(a.y - b.y) < BOX_H


def test_root_at_origin():
    mv = build_mapview("root.md", _materials())
    root = next(n for n in mv.nodes if n.id == "root.md")
    assert (root.x, root.y) == (0.0, 0.0)
    assert root.hop == 0


def test_no_box_overlap():
    mv = build_mapview("root.md", _materials([("e.md", "E", 0.9), ("f.md", "F", 0.8)]))
    ns = mv.nodes
    for i in range(len(ns)):
        for j in range(i + 1, len(ns)):
            assert not _boxes_overlap(ns[i], ns[j]), f"{ns[i].id} overlaps {ns[j].id}"


def test_communities_occupy_contiguous_wedges():
    # Each community must own a contiguous angular arc — no interleaving.
    mv = build_mapview("root.md", _materials())
    non_root = [n for n in mv.nodes if n.hop > 0]
    ordered = sorted(non_root, key=lambda n: math.atan2(n.y, n.x) % (2 * math.pi))
    runs = [n.community for n in ordered]
    # Number of contiguous community-runs equals the number of distinct communities.
    changes = sum(1 for k in range(1, len(runs)) if runs[k] != runs[k - 1])
    assert changes == len(set(runs)) - 1


def test_deterministic():
    m = _materials([("e.md", "E", 0.9)])
    a = build_mapview("root.md", m)
    b = build_mapview("root.md", _materials([("e.md", "E", 0.9)]))
    assert [(n.id, n.x, n.y) for n in a.nodes] == [(n.id, n.x, n.y) for n in b.nodes]


def test_root_argument_normalized_to_md():
    mv = build_mapview("root", _materials())  # no .md suffix
    assert mv.root == "root.md"
    assert any(n.id == "root.md" and n.hop == 0 for n in mv.nodes)
