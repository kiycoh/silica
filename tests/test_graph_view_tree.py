"""Unit tests for render_tree() — the vault file tree emitter."""
from __future__ import annotations

from silica.ui.web.graph_view import render_tree


def _node(nid: str, node_type: str = "note") -> dict:
    return {"id": nid, "label": nid.split("/")[-1], "type": node_type, "path": nid}


def test_nested_folders_become_details():
    html = render_tree([_node("a/b/c.md"), _node("a/d.md"), _node("e.md")])
    assert '<div id="file-tree">' in html
    assert "<summary>a</summary>" in html
    assert "<summary>b</summary>" in html
    # top-level folder is open, nested folder is not
    assert "<details open><summary>a</summary>" in html
    assert "<details><summary>b</summary>" in html


def test_notes_render_as_tree_note_divs_with_data_id():
    html = render_tree([_node("a/c.md")])
    assert '<div class="tree-note" data-id="a/c.md">c.md</div>' in html


def test_ghost_nodes_excluded():
    html = render_tree([_node("real.md"), _node("Ghost", node_type="ghost")])
    assert "real.md" in html
    assert "Ghost" not in html


def test_empty_path_excluded():
    n = {"id": "x", "label": "x", "type": "note", "path": ""}
    html = render_tree([n])
    assert "tree-note" not in html


def test_folders_sorted_before_notes_within_a_level():
    # at root: folder "z" must appear before note "a.md"
    html = render_tree([_node("a.md"), _node("z/inner.md")])
    assert html.index("<summary>z</summary>") < html.index("a.md")


def test_notes_sorted_case_insensitively():
    html = render_tree([_node("Beta.md"), _node("alpha.md")])
    assert html.index("alpha.md") < html.index("Beta.md")


def test_names_html_escaped():
    html = render_tree([_node("a & b.md")])
    assert "a &amp; b.md" in html


def test_empty_input_does_not_crash_and_has_no_leaves():
    html = render_tree([])
    assert html == '<div id="file-tree"></div>'


def test_all_ghost_input_has_no_leaves():
    html = render_tree([_node("g", node_type="ghost")])
    assert "tree-note" not in html
