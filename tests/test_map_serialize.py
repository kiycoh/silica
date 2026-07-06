"""Serializer + bounding invariants for kernel/mindmap."""
from __future__ import annotations

import json

import networkx as nx

from silica.kernel.mindmap import (
    BOX_W,
    MapMaterials,
    _resolve_in,
    build_mapview,
    mapview_to_canvas,
    mapview_to_gui,
    render_map_svg,
)

_MUTED = "#5a6372"


def _materials(latent=(), n_wiki=2):
    g = nx.Graph()
    for i in range(n_wiki):
        g.add_edge("root.md", f"n{i}.md")
    return MapMaterials(
        graph=g,
        titles={p: p[:-3] for p in g.nodes},
        community_of={p: 0 for p in g.nodes},
        latent=list(latent),
    )


def test_canvas_is_valid_and_conformant():
    mv = build_mapview("root.md", _materials([("lat.md", "Lat", 0.9)]))
    canvas = mapview_to_canvas(mv)
    # Round-trips as JSON.
    assert json.loads(json.dumps(canvas)) == canvas

    ids = {n["id"] for n in canvas["nodes"]}
    for node in canvas["nodes"]:
        assert node["type"] == "file"
        assert node["file"].endswith(".md")
    for edge in canvas["edges"]:
        assert edge["fromNode"] in ids
        assert edge["toNode"] in ids


def test_latent_edges_muted_with_tilde_label():
    mv = build_mapview("root.md", _materials([("lat.md", "Lat", 0.9)]))
    canvas = mapview_to_canvas(mv)
    kinds = {(e.src, e.dst): e.kind for e in mv.edges}
    for edge in canvas["edges"]:
        kind = kinds.get((edge["fromNode"], edge["toNode"])) or kinds.get(
            (edge["toNode"], edge["fromNode"])
        )
        if kind == "latent":
            assert edge["label"] == "≈"
            assert edge["color"] == _MUTED
        else:
            assert edge["label"] == ""
            assert edge["color"] != _MUTED


def test_gui_payload_same_positions_as_canvas():
    mv = build_mapview("root.md", _materials([("lat.md", "Lat", 0.9)]))
    canvas = {n["id"]: n for n in mapview_to_canvas(mv)["nodes"]}
    gui = {n["id"]: n for n in mapview_to_gui(mv)["nodes"]}
    assert canvas.keys() == gui.keys()
    half = round(BOX_W / 2)
    for nid, cn in canvas.items():
        # canvas stores top-left; gui stores centre — same map, offset by half-box.
        assert gui[nid]["x"] == cn["x"] + half


def test_cap_respected():
    mv = build_mapview("root.md", _materials(n_wiki=20), max_nodes=5)
    assert len(mv.nodes) <= 5


def test_both_legs_abstain_yields_wikilink_tree_only():
    mv = build_mapview("root.md", _materials(latent=[], n_wiki=3))
    assert mv.edges
    assert all(e.kind == "wikilink" for e in mv.edges)
    assert {n.id for n in mv.nodes} == {"root.md", "n0.md", "n1.md", "n2.md"}


def test_resolve_accepts_title_and_path_in_subfolder():
    # The GUI bug: a note in a subfolder given by its TITLE must resolve to the
    # full graph key (path with .md), not stay unmatched.
    paths = ["AI/Storia dell'IA moderna.md", "concetti/grafo.md"]
    titles = {
        "AI/Storia dell'IA moderna.md": "Storia dell'IA moderna",
        "concetti/grafo.md": "Grafo",
    }
    assert _resolve_in("Storia dell'IA moderna", paths, titles) == "AI/Storia dell'IA moderna.md"
    assert _resolve_in("AI/Storia dell'IA moderna", paths, titles) == "AI/Storia dell'IA moderna.md"
    assert _resolve_in("AI/Storia dell'IA moderna.md", paths, titles) == "AI/Storia dell'IA moderna.md"
    assert _resolve_in("nope", paths, titles) is None


def test_svg_renders_and_dashes_latent():
    mv = build_mapview("root.md", _materials([("lat.md", "Lat", 0.9)]))
    svg = render_map_svg(mv)
    assert "<svg" in svg and "</svg>" in svg
    assert "stroke-dasharray" in svg  # the latent edge is dashed
