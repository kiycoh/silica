"""Tests for render_html() in silica/kernel/graph_export.py.

Exercises the 3d-force-graph renderer output without any network access.
render_html() accepts lib_js as a string parameter, so we can pass a dummy
string or "" to avoid CDN fetches entirely.
"""
from __future__ import annotations

import pytest

from silica.ui.web.graph_view import render_html


# ---------------------------------------------------------------------------
# Minimal fixtures
# ---------------------------------------------------------------------------

def _node(nid: str, label: str = "", group: int = -1, node_type: str = "note") -> dict:
    return {
        "id": nid,
        "label": label or nid,
        "type": node_type,
        "group": group,
        "color": {"background": "#2d4a6e", "border": "#4a9eff"},
        "path": nid,
        "size": 16,
    }


def _edge(eid: str, src: str, dst: str, etype: str = "EXTRACTED") -> dict:
    color = "#4a9eff" if etype == "EXTRACTED" else "#ffaa33"
    return {
        "id": eid,
        "from": src,
        "to": dst,
        "type": etype,
        "color": {"color": color, "opacity": 0.6},
        "width": 1.5,
    }


@pytest.fixture()
def small_graph():
    nodes = [_node("A"), _node("B"), _node("C", node_type="ghost")]
    edges = [
        _edge("e0", "A", "B", "EXTRACTED"),
        _edge("e1", "A", "C", "AMBIGUOUS"),
    ]
    return nodes, edges


# ---------------------------------------------------------------------------
# 1. Output contains ForceGraph3D(
# ---------------------------------------------------------------------------

class TestForceGraph3DPresent:
    def test_contains_forcegraph3d_constructor(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "ForceGraph3D(" in html

    def test_forcegraph3d_present_with_empty_lib_js(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="")
        assert "ForceGraph3D(" in html

    def test_uses_constructor_form_not_legacy_curried(self, small_graph):
        # 3d-force-graph >= 1.x uses `new ForceGraph3D(element)`. The legacy
        # curried `ForceGraph3D()(element)` form throws at runtime in 1.80.0,
        # leaving the graph area blank. Lock the constructor form in.
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "new ForceGraph3D(" in html
        assert "ForceGraph3D()(" not in html


# ---------------------------------------------------------------------------
# 2. linkSource("from") and linkTarget("to") are present
# ---------------------------------------------------------------------------

class TestLinkSourceTarget:
    def test_link_source_from(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert '.linkSource("from")' in html

    def test_link_target_to(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert '.linkTarget("to")' in html


# ---------------------------------------------------------------------------
# 3. Inline bundle vs CDN fallback
# ---------------------------------------------------------------------------

class TestLibJsInlining:
    def test_inline_when_lib_js_provided(self, small_graph):
        """When lib_js is non-empty, bundle is inlined as <script>…</script>."""
        nodes, edges = small_graph
        bundle = "/* 3d-force-graph bundle */"
        html = render_html(nodes, edges, lib_js=bundle)
        assert f"<script>{bundle}</script>" in html

    def test_no_cdn_src_when_lib_js_provided(self, small_graph):
        """When lib_js is non-empty, no <script src= CDN tag should appear."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="/* bundle */")
        assert "<script src=" not in html

    def test_cdn_fallback_when_no_lib_js(self, small_graph):
        """When lib_js is empty, a <script src= CDN tag should appear."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="")
        assert "<script src=" in html

    def test_cdn_url_points_to_3d_force_graph(self, small_graph):
        """CDN fallback should reference 3d-force-graph, not vis-network."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="")
        assert "3d-force-graph" in html


# ---------------------------------------------------------------------------
# 4. No vis.Network or new vis.DataSet in output
# ---------------------------------------------------------------------------

class TestNoVisReferences:
    def test_no_vis_network(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "vis.Network" not in html

    def test_no_vis_dataset(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "new vis.DataSet" not in html

    def test_no_vis_network_cdn_path(self):
        """Even with empty lib_js (CDN mode), no vis-network CDN URL appears."""
        html = render_html([], [], lib_js="")
        assert "vis-network" not in html


# ---------------------------------------------------------------------------
# Additional sanity checks
# ---------------------------------------------------------------------------

class TestRenderSanity:
    def test_title_appears_in_output(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, title="My Test Graph", lib_js="// x")
        assert "My Test Graph" in html

    def test_graph_data_json_embedded(self, small_graph):
        """RAW_NODES and RAW_EDGES constants should appear in the output."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "RAW_NODES" in html
        assert "RAW_EDGES" in html

    def test_outdeg_indeg_precompute_present(self, small_graph):
        """Degree precompute block should be present."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "outDeg[e.from]" in html
        assert "inDeg[e.to]" in html

    def test_fit_button_uses_graph_zoom(self, small_graph):
        """Fit graph button should call Graph.zoomToFit, not network.fit."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "Graph.zoomToFit(400)" in html
        assert "network.fit(" not in html

    def test_node_visibility_accessor(self, small_graph):
        """nodeVisibility accessor should be wired up."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert ".nodeVisibility(" in html

    def test_link_visibility_accessor(self, small_graph):
        """linkVisibility accessor should be wired up."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert ".linkVisibility(" in html

    def test_visibility_refresh_trick_present(self, small_graph):
        """applyFilters() should use the re-pass trick to force a visibility refresh."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "Graph.nodeVisibility(Graph.nodeVisibility())" in html

    def test_on_node_click_used(self, small_graph):
        """Drawer open should use onNodeClick, not network.on click."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "onNodeClick" in html
        assert 'network.on("click"' not in html

    def test_on_background_click_closes_drawer_and_clears_focus(self, small_graph):
        """Background tap closes the drawer AND reverts the focus/dim state.

        (A camera-orbit drag never reaches here: the vendored bundle sets
        clickAfterDrag=false, so onBackgroundClick fires only on a clean tap.)
        """
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "onBackgroundClick(" in html
        assert "closeDrawer()" in html
        assert "clearFocus()" in html

    def test_empty_graph_renders(self):
        """render_html with empty node/edge lists should not raise."""
        html = render_html([], [], lib_js="// x")
        assert "<!DOCTYPE html>" in html


# ---------------------------------------------------------------------------
# Search → results list → fly-to-focus (findability for the searching user)
# ---------------------------------------------------------------------------

class TestSearchResultsFlyTo:
    """Typing a query should produce a clickable ranked list, and choosing a
    result should fly the camera to that node and select it — not just dim the
    rest of the cloud."""

    def test_results_container_present(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert 'id="search-results"' in html

    def test_onsearch_renders_results(self, small_graph):
        """onSearch should populate the results list, not only set a filter."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "renderResults(" in html

    def test_scorer_searches_beyond_label(self, small_graph):
        """Ranking should consider path and tags, not just the label."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "function scoreNode(" in html
        assert ".path" in html

    def test_focus_node_uses_camera_position(self, small_graph):
        """Choosing a result flies the camera via the 3d-force-graph API."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "function focusNode(" in html
        assert ".cameraPosition(" in html

    def test_select_node_shared_between_click_and_result(self, small_graph):
        """Node-click and result-click should both route through selectNode."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "function selectNode(" in html

    def test_enter_focuses_top_result(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "onSearchKey(" in html

    def test_embedded_node_click_posts_open_note_to_parent(self, small_graph):
        """When embedded in the web-UI iframe, a node click hands off to the
        parent's note drawer instead of opening the internal metadata drawer."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "window.parent !== window" in html
        assert "postMessage" in html
        assert "silica-open-note" in html


class TestFocusDim:
    def test_neighbors_map_precomputed(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "neighbors" in html

    def test_apply_and_clear_focus_present(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "function applyFocus(" in html
        assert "function clearFocus(" in html

    def test_node_color_has_dim_branch(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "_dim" in html
        assert "#1c1c1c" in html

    def test_choose_node_applies_focus(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "applyFocus(node.id)" in html

    def test_clear_focus_zooms_to_fit(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "Graph.zoomToFit(600)" in html

    def test_direct_node_click_dims_without_camera_fly(self, small_graph):
        """Clicking a node in the 3D view itself dims non-neighbours like tree/search
        picks, but must NOT call focusNode — the user is already looking at the spot."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "Graph.onNodeClick(node => {" in html
        assert "selectNode(node); applyFocus(node.id);" in html

    def test_parent_can_sync_focus_by_path(self, small_graph):
        """The embedding page (note-panel navigation) can tell the graph which
        note is open elsewhere so it mirrors the dim state, without moving the camera."""
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "silica-focus-path" in html
        assert "NODE_BY_ID[e.data.path]" in html


class TestFileTree:
    def test_file_tree_container_present(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert 'id="file-tree"' in html

    def test_files_section_title_present(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert ">Files<" in html

    def test_tree_leaf_for_real_note(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        # node "A" is a real note (id == path == "A")
        assert 'class="tree-note" data-id="A"' in html

    def test_node_by_id_map_built(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "NODE_BY_ID" in html

    def test_choose_node_function_present(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "function chooseNode(" in html

    def test_tree_click_delegated(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert 'getElementById("file-tree").addEventListener("click"' in html

    def test_choose_result_routes_through_choose_node(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "chooseNode(" in html


# ---------------------------------------------------------------------------
# 5. Perf knobs for big vaults — keep WebGL geometry count low
# ---------------------------------------------------------------------------

class TestBigVaultPerfKnobs:
    """1200-node vaults lag because 3d-force-graph defaults turn every edge into
    a cylinder + arrow-cone mesh and never stop the layout. Lock the cheap path.
    """

    def test_links_are_zero_width_gl_lines(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert ".linkWidth(0)" in html

    def test_no_directional_arrow_cones(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert "linkDirectionalArrowLength" not in html

    def test_finite_cooldown(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert ".cooldownTicks(" in html

    def test_low_node_resolution(self, small_graph):
        nodes, edges = small_graph
        html = render_html(nodes, edges, lib_js="// dummy")
        assert ".nodeResolution(" in html
