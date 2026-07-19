"""Tests for export_graph() in silica/ui/web/graph_view.py.

The viewer moved out of the kernel and now vendors its JS bundle instead of
fetching it from a CDN. These lock the new behavior:

1. export_graph reads the vendored bundle and inlines it — the emitted HTML is
   self-contained (no CDN <script src=), i.e. it opens offline.
2. A missing vendored asset is a loud RuntimeError (packaging bug), never a
   silent fall back to the CDN <script src=.
3. Importing the viewer does NOT drag in FastAPI, so the core /graph command
   still works on a base install without the [gui] extra.
"""
from __future__ import annotations

import subprocess
import sys

import pytest


def test_export_graph_unifies_wikilinks_and_similar(monkeypatch, tmp_path):
    """One build carries BOTH the wikilink edges and the embedding k-NN overlay;
    communities are Louvain on the WIKILINKS, and the SIMILAR filter row renders."""
    import silica.kernel.graph_export as ge
    import silica.ui.web.graph_view as gv

    monkeypatch.setattr(gv, "_vendored_lib_js", lambda: "/*JS*/")
    monkeypatch.setattr(ge, "build_graph_data", lambda folder="": (
        [{"id": "a.md", "label": "a", "type": "note", "group": -1,
          "color": {"background": "#4d5575"}, "path": "a.md", "size": 16},
         {"id": "b.md", "label": "b", "type": "note", "group": -1,
          "color": {"background": "#4d5575"}, "path": "b.md", "size": 16}],
        [{"id": "e0", "from": "a.md", "to": "b.md", "type": "EXTRACTED",
          "color": {"color": "#8f8f8f"}}],
    ))
    seen = {}

    def fake_detect(nodes, edges, edge_type="EXTRACTED"):
        # Louvain must see only the wikilink layer, never the k-NN overlay.
        seen["edge_types"] = {e.get("type") for e in edges}
        return []

    monkeypatch.setattr(ge, "detect_communities", fake_detect)
    monkeypatch.setattr(ge, "knn_edges", lambda nodes, k=6: [
        {"id": "s0", "from": "a.md", "to": "b.md", "type": "SIMILAR",
         "color": {"color": "#00a5e1", "opacity": 0.35}, "width": 2.0, "score": 0.9},
    ])

    out = tmp_path / "g.html"
    res = gv.export_graph(output_path=str(out))
    html = out.read_text(encoding="utf-8")

    assert "SIMILAR" in html and "EXTRACTED" in html   # both layers in one build
    assert 'id="cb-similar"' in html                    # semantic-overlay filter row
    assert seen["edge_types"] == {"EXTRACTED"}          # k-NN kept out of Louvain
    assert res["edges"] == 1 and res["similar"] == 1


def test_export_graph_inlines_vendored_bundle_no_cdn(monkeypatch, tmp_path):
    import silica.kernel.graph_export as ge
    import silica.ui.web.graph_view as gv

    sentinel = "/*VENDORED_SENTINEL_12345*/"
    monkeypatch.setattr(gv, "_vendored_lib_js", lambda: sentinel)
    monkeypatch.setattr(ge, "build_graph_data", lambda folder="": (
        [{"id": "a", "label": "a", "type": "note", "group": -1,
          "color": {"background": "#4d5575"}, "path": "a", "size": 16}],
        [],
    ))
    monkeypatch.setattr(ge, "detect_communities", lambda nodes, edges: [])
    monkeypatch.setattr(ge, "knn_edges", lambda nodes, k=6: [])

    out = tmp_path / "g.html"
    res = gv.export_graph(output_path=str(out))

    html = out.read_text(encoding="utf-8")
    assert sentinel in html                 # vendored bundle inlined
    assert "<script src=" not in html       # no CDN fallback tag
    assert "cdn.jsdelivr" not in html       # genuinely offline
    assert res["success"] is True


def test_export_graph_has_density_forces_panel(monkeypatch, tmp_path):
    """Density-aware layout: the emitted HTML carries the auto-scaling baseline
    (FORCE_SCALE from avg degree) and the live Forces sliders that multiply it.
    Same template serves both views, so one export covers links and semantic."""
    import silica.kernel.graph_export as ge
    import silica.ui.web.graph_view as gv

    monkeypatch.setattr(gv, "_vendored_lib_js", lambda: "/*JS*/")
    monkeypatch.setattr(ge, "build_graph_data", lambda folder="": (
        [{"id": "a.md", "label": "a", "type": "note", "group": -1,
          "color": {"background": "#4d5575"}, "path": "a.md", "size": 16}],
        [],
    ))
    monkeypatch.setattr(ge, "detect_communities", lambda nodes, edges: [])
    monkeypatch.setattr(ge, "knn_edges", lambda nodes, k=6: [])

    out = tmp_path / "g.html"
    gv.export_graph(output_path=str(out))
    html = out.read_text(encoding="utf-8")

    assert "FORCE_SCALE" in html            # auto-scaling baseline present
    for sid in ("sl-repel", "sl-dist", "sl-center"):
        assert f'id="{sid}"' in html        # the three live sliders
    assert "d3ReheatSimulation" in html     # sliders reheat, never rebuild
    assert "silica-graph-forces" in html    # slider persistence key


def test_export_graph_raises_when_vendored_asset_missing(monkeypatch, tmp_path):
    """A missing vendored asset is loud — never a silent CDN fallback."""
    import silica.ui.web.graph_view as gv

    def boom() -> str:
        raise RuntimeError("graph_export: vendored 3d-force-graph.min.js is missing")

    monkeypatch.setattr(gv, "_vendored_lib_js", boom)
    with pytest.raises(RuntimeError, match="3d-force-graph"):
        gv.export_graph(output_path=str(tmp_path / "g.html"))


def test_graph_view_import_is_gui_free():
    """Importing the viewer must not require the optional [gui] extra (FastAPI).

    Runs in a clean interpreter so it is immune to other tests having already
    imported fastapi in the shared pytest process.
    """
    code = (
        "import silica.ui.web.graph_view, sys; "
        "assert 'fastapi' not in sys.modules, 'graph_view pulled in fastapi'; "
        "print('ok')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
