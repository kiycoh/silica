# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Concept co-occurrence heatmap (kernel/heatmap.py): concept x concept matrix
over the cooccur store, seriated by Louvain community so topics read as
diagonal blocks and cross-community hot cells surface bridges. Same df window
as the F4 concepts view (min_df / df_cap over Concept sets) so both surfaces
describe the same concept subset."""
from __future__ import annotations

from silica.kernel.cooccurrence import CooccurStore


def _store(tmp_path):
    s = CooccurStore(path=tmp_path / "cooccur.json", lang="english")

    def contrib(stems, edges=()):
        return {"nodes": {k: {"label": k, "count": 1} for k in stems},
                "edges": [list(e) for e in edges]}

    # Two tight blocks (a1-a2, b1-b2) + one weak bridge (a1-b1).
    # hub appears in all 5 notes -> df_cap drops it; solo df=1 -> min_df drops it.
    s.upsert_note("n1.md", contrib(["a1", "a2", "hub"], [("a1", "a2", 3.0)]))
    s.upsert_note("n2.md", contrib(["a1", "a2", "hub"], [("a1", "a2", 2.0)]))
    s.upsert_note("n3.md", contrib(["b1", "b2", "hub"], [("b1", "b2", 4.0)]))
    s.upsert_note("n4.md", contrib(["b1", "b2", "hub", "solo"],
                                   [("b1", "b2", 1.0), ("b1", "solo", 1.0)]))
    s.upsert_note("n5.md", contrib(["a1", "b1", "hub"], [("a1", "b1", 0.5)]))
    return s


def test_build_applies_df_window_and_groups_communities_contiguously(tmp_path):
    from silica.kernel.heatmap import build_heatmap

    view = build_heatmap(_store(tmp_path))
    assert set(view.stems) == {"a1", "a2", "b1", "b2"}  # hub + solo dropped
    # Community blocks are contiguous runs (seriation IS the design).
    seen = []
    for c in view.community:
        if not seen or seen[-1] != c:
            seen.append(c)
    assert len(seen) == len(set(view.community))
    # Within a block: df descending (a1 df=3 before a2 df=2).
    assert view.stems.index("a1") < view.stems.index("a2")
    assert view.stems.index("b1") < view.stems.index("b2")


def test_build_matrix_is_symmetric_aggregated_weights(tmp_path):
    from silica.kernel.heatmap import build_heatmap

    view = build_heatmap(_store(tmp_path))
    i, j = view.stems.index("a1"), view.stems.index("a2")
    assert view.matrix[i][j] == 5.0  # 3.0 + 2.0 across notes
    assert view.matrix[j][i] == 5.0
    k = view.stems.index("b2")
    assert view.matrix[j][k] == 0.0  # a2-b2 never co-occur
    assert all(view.matrix[i][i] == 0.0 for i in range(len(view.stems)))


def test_build_empty_store_degrades_to_empty_view(tmp_path):
    from silica.kernel.heatmap import build_heatmap

    view = build_heatmap(CooccurStore(path=tmp_path / "c.json"))
    assert view.stems == [] and view.matrix == []


def test_render_is_selfcontained_html_with_cells_and_bridge_color(tmp_path):
    from silica.kernel.heatmap import build_heatmap, render_heatmap_svg

    html = render_heatmap_svg(build_heatmap(_store(tmp_path)))
    assert html.startswith("<!DOCTYPE html>") and "<svg" in html
    assert "<script src" not in html and "<link " not in html  # offline
    for label in ("a1", "a2", "b1", "b2"):
        assert label in html
    # The a1-b1 bridge crosses communities -> amber cell, distinct from blocks.
    assert "#c9a227" in html
    # Tooltip carries the pair + weight.
    assert "a1 × a2 — 5" in html


def test_render_empty_view_shows_message(tmp_path):
    from silica.kernel.heatmap import build_heatmap, render_heatmap_svg

    html = render_heatmap_svg(build_heatmap(CooccurStore(path=tmp_path / "c.json")))
    assert "no co-occurrence" in html.lower()


def test_build_focus_bypasses_df_window_with_top_neighbors(tmp_path):
    from silica.kernel.heatmap import build_heatmap

    view = build_heatmap(_store(tmp_path), focus="b1")
    assert view.focus == "b1"
    # b1's co-occurrence neighbors by weight: b2 (5.0), solo (1.0), a1 (0.5).
    # solo (df=1) and any hub would be dropped by the df window — focus skips it.
    assert set(view.stems) == {"b1", "b2", "solo", "a1"}


def test_build_focus_caps_neighbors_at_top_n(tmp_path):
    from silica.kernel.heatmap import build_heatmap

    view = build_heatmap(_store(tmp_path), focus="b1", top_n=2)
    assert set(view.stems) == {"b1", "b2"}  # strongest neighbor only


def test_build_focus_resolves_label_case_insensitively(tmp_path):
    from silica.kernel.heatmap import build_heatmap

    view = build_heatmap(_store(tmp_path), focus="B1")
    assert view.focus == "b1"


def test_build_focus_fuzzy_falls_back_to_substring_matches(tmp_path):
    from silica.kernel.heatmap import build_heatmap

    # "b" is not a stem/label; b1 and b2 contain it -> matched + their neighbors.
    view = build_heatmap(_store(tmp_path), focus="b")
    assert view.focus is None
    assert view.hits == frozenset({"b1", "b2"})
    assert {"b1", "b2", "solo"} <= set(view.stems)  # solo rides in as b1's neighbor
    assert view.note and "not found" in view.note and "related" in view.note


def test_build_focus_no_match_degrades_to_default_view_with_note(tmp_path):
    from silica.kernel.heatmap import build_heatmap

    view = build_heatmap(_store(tmp_path), focus="zzz")
    # Never a dead end: the default df-window view, annotated.
    assert set(view.stems) == {"a1", "a2", "b1", "b2"}
    assert view.note and "zzz" in view.note and "not found" in view.note


def test_render_overlays_note_translucently(tmp_path):
    from silica.kernel.heatmap import build_heatmap, render_heatmap_svg

    html = render_heatmap_svg(build_heatmap(_store(tmp_path), focus="zzz"))
    assert 'id="note"' in html and "zzz" in html


def test_render_carries_focus_search_form_with_cap_input(tmp_path):
    from silica.kernel.heatmap import build_heatmap, render_heatmap_svg

    html = render_heatmap_svg(build_heatmap(_store(tmp_path), focus="b1", top_n=17))
    assert '<form' in html and 'name="q"' in html  # GET form -> /heatmap?q=…
    assert 'name="n"' in html and 'value="17"' in html  # cap rides along


def test_build_focus_neighbor_outside_concept_sets_does_not_crash(tmp_path):
    from silica.kernel.heatmap import build_heatmap

    s = CooccurStore(path=tmp_path / "c.json", lang="english")
    # 32 stems: "weak" has the lowest count -> falls out of the top-30 Concept
    # set (df source), yet its edge lands it in adjacency (neighbor source).
    # Real stores hit this constantly; the synthetic ones above never do.
    nodes = {f"w{i:02d}": {"label": f"w{i:02d}", "count": 5} for i in range(30)}
    nodes["hub"] = {"label": "hub", "count": 9}
    nodes["weak"] = {"label": "weak", "count": 1}
    s.upsert_note("big.md", {"nodes": nodes, "edges": [["hub", "weak", 2.0]]})

    view = build_heatmap(s, focus="hub")
    assert "weak" in view.stems  # selected by edge weight; df=0 must not crash


def test_build_note_selects_note_concepts_plus_out_of_note_neighbors(tmp_path):
    from silica.kernel.heatmap import build_heatmap

    # n4's concepts with edges: b1, b2, solo (hub has none). a1 rides in as
    # b1's strongest out-of-note neighbor — the one thing this view adds
    # beyond the note body itself.
    view = build_heatmap(_store(tmp_path), note="n4.md")
    assert set(view.stems) == {"b1", "b2", "solo", "a1"}
    assert view.hits == frozenset({"b1", "b2", "solo"})  # own concepts highlighted


def test_build_note_caps_at_top_n_note_concepts_first(tmp_path):
    from silica.kernel.heatmap import build_heatmap

    # Own concepts df-desc fill the budget before any neighbor rides in.
    view = build_heatmap(_store(tmp_path), note="n4.md", top_n=2)
    assert set(view.stems) == {"b1", "b2"}


def test_build_note_resolves_title_without_extension(tmp_path):
    from silica.kernel.heatmap import build_heatmap

    view = build_heatmap(_store(tmp_path), note="N4")
    assert view.hits == frozenset({"b1", "b2", "solo"})


def test_build_note_unknown_degrades_to_default_view_with_note(tmp_path):
    from silica.kernel.heatmap import build_heatmap

    view = build_heatmap(_store(tmp_path), note="ghost.md")
    assert set(view.stems) == {"a1", "a2", "b1", "b2"}
    assert view.note and "ghost" in view.note and "not found" in view.note


def test_build_min_pct_zeroes_cells_below_share_of_max(tmp_path):
    from silica.kernel.heatmap import build_heatmap

    # max weight 5.0; at 50% the 0.5 a1-b1 bridge dies, strong pairs survive.
    view = build_heatmap(_store(tmp_path), min_pct=50)
    assert set(view.stems) == {"a1", "a2", "b1", "b2"}
    i, j = view.stems.index("a1"), view.stems.index("b1")
    assert view.matrix[i][j] == 0.0
    assert view.matrix[i][view.stems.index("a2")] == 5.0


def test_build_min_pct_drops_stems_left_without_cells(tmp_path):
    from silica.kernel.heatmap import build_heatmap

    # b1's neighbors: b2 (5.0), solo (1.0), a1 (0.5); at 50% only b2 survives.
    view = build_heatmap(_store(tmp_path), focus="b1", min_pct=50)
    assert set(view.stems) == {"b1", "b2"}


def test_render_form_carries_min_pct_input(tmp_path):
    from silica.kernel.heatmap import build_heatmap, render_heatmap_svg

    html = render_heatmap_svg(build_heatmap(_store(tmp_path), min_pct=30))
    assert 'name="p"' in html and 'value="30"' in html


def test_build_min_pct_in_focus_mode_references_the_focus_row(tmp_path):
    from silica.kernel.heatmap import build_heatmap

    s = CooccurStore(path=tmp_path / "c.json", lang="english")
    # A neighbor pair (c1-c2) dwarfs every focus cell: with a global-max
    # threshold, raising p would blank exactly what was searched for.
    s.upsert_note("n.md", {
        "nodes": {k: {"label": k, "count": 1} for k in ("f", "c1", "c2")},
        "edges": [["c1", "c2", 10.0], ["f", "c1", 2.0], ["f", "c2", 1.0]]})

    view = build_heatmap(s, focus="f", min_pct=40)  # thr = 40% of 2.0, not of 10.0
    i = view.stems.index("f")
    assert view.matrix[i][view.stems.index("c1")] == 2.0
    assert view.matrix[i][view.stems.index("c2")] == 1.0
