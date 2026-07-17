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


def test_export_graph_concepts_mode_inlines_bipartite(monkeypatch, tmp_path):
    """F4: mode="concepts" merges the Concept-set bipartite expansion into the
    exported dataset; default mode stays bit-identical (no concept nodes)."""
    import silica.kernel.graph_export as ge
    import silica.ui.web.graph_view as gv

    monkeypatch.setattr(gv, "_vendored_lib_js", lambda: "/*JS*/")
    monkeypatch.setattr(ge, "build_graph_data", lambda folder="": (
        [{"id": "a.md", "label": "a", "type": "note", "group": -1,
          "color": {"background": "#4d5575"}, "path": "a.md", "size": 16}],
        [],
    ))
    monkeypatch.setattr(ge, "detect_communities", lambda nodes, edges: [])
    monkeypatch.setattr(ge, "build_bipartite_data", lambda nodes, store, **kw: (
        [{"id": "concept:ml", "label": "ml", "type": "concept", "df": 2,
          "group": -1, "size": 10}],
        [{"from": "a.md", "to": "concept:ml", "type": "CONCEPT"}],
    ))

    out = tmp_path / "g.html"
    res = gv.export_graph(output_path=str(out), mode="concepts")
    html = out.read_text(encoding="utf-8")
    assert "concept:ml" in html
    assert "CONCEPT" in html
    assert res["concepts"] == 1

    out2 = tmp_path / "g2.html"
    gv.export_graph(output_path=str(out2))
    assert "concept:ml" not in out2.read_text(encoding="utf-8")


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
    Same template serves both views, so one export covers links and concepts."""
    import silica.kernel.graph_export as ge
    import silica.ui.web.graph_view as gv

    monkeypatch.setattr(gv, "_vendored_lib_js", lambda: "/*JS*/")
    monkeypatch.setattr(ge, "build_graph_data", lambda folder="": (
        [{"id": "a.md", "label": "a", "type": "note", "group": -1,
          "color": {"background": "#4d5575"}, "path": "a.md", "size": 16}],
        [],
    ))
    monkeypatch.setattr(ge, "detect_communities", lambda nodes, edges: [])

    out = tmp_path / "g.html"
    gv.export_graph(output_path=str(out))
    html = out.read_text(encoding="utf-8")

    assert "FORCE_SCALE" in html            # auto-scaling baseline present
    for sid in ("sl-repel", "sl-dist", "sl-center"):
        assert f'id="{sid}"' in html        # the three live sliders
    assert "d3ReheatSimulation" in html     # sliders reheat, never rebuild
    assert "silica-graph-forces-" in html   # per-view persistence key


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
