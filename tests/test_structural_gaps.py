# SPDX-License-Identifier: AGPL-3.0-or-later
"""Structural-gap detection + discourse-shape / betweenness plumbing."""
from silica.kernel.graph_export import structural_gaps
from silica.kernel.graph_report.compute import compute_report


def _clique(prefix: str, n: int, group: int):
    nodes = [
        {"id": f"{prefix}{i}", "label": f"{prefix}{i}", "type": "note", "group": group}
        for i in range(n)
    ]
    edges = [
        {"type": "EXTRACTED", "from": f"{prefix}{i}", "to": f"{prefix}{j}"}
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return nodes, edges


def test_two_disconnected_areas_surface_as_a_gap():
    na, ea = _clique("a", 4, 0)
    nb, eb = _clique("b", 4, 1)
    gaps = structural_gaps(na + nb, ea + eb)
    assert gaps, "two disconnected clusters must produce a gap"
    ca, cb, _, _, inter, score, density = gaps[0]
    assert (ca, cb) == (0, 1)
    assert inter == 0
    assert score == 16.0  # 4*4 / (1+0)
    assert density == 1.0  # 1 - 0/(4*4): a full structural hole


def test_one_link_between_shrinks_the_gap():
    na, ea = _clique("a", 4, 0)
    nb, eb = _clique("b", 4, 1)
    linked = [{"type": "EXTRACTED", "from": "a0", "to": "b0"}]
    gaps = structural_gaps(na + nb, ea + eb + linked)
    _, _, _, _, inter, score, density = gaps[0]
    assert inter == 1
    assert score == 8.0  # 4*4 / (1+1)
    assert density == 0.9375  # 1 - 1/(4*4): bounded, unlike score


def test_hub_is_the_highest_degree_member():
    na, ea = _clique("a", 3, 0)
    na.append({"id": "a_x", "label": "a_x", "type": "note", "group": 0})
    ea.append({"type": "EXTRACTED", "from": "a0", "to": "a_x"})  # a0 now highest degree
    nb, eb = _clique("b", 3, 1)
    gaps = structural_gaps(na + nb, ea + eb)
    _, _, hub_a, _, _, _, _ = gaps[0]
    assert hub_a == "a0"


def test_singleton_clusters_are_ignored():
    na, ea = _clique("a", 3, 0)
    solo = [{"id": "s0", "label": "s0", "type": "note", "group": 2}]
    gaps = structural_gaps(na + solo, ea, min_size=2)
    assert gaps == []  # only one cluster >= min_size -> no pair to compare


def test_compute_report_wires_gaps_discourse_and_betweenness():
    na, ea = _clique("a", 4, 0)
    nb, eb = _clique("b", 4, 1)
    # Louvain (inside compute_report) reassigns groups; two 4-cliques -> 2 areas.
    r = compute_report(_nodes_edges_override=(na + nb, ea + eb), analytics=True, top_k=10)
    assert r.structural_gaps, "analytics report must carry structural gaps"
    assert r.discourse_state == "Diversified"  # two balanced, well-connected areas
    assert all(isinstance(n.betweenness, float) for n in r.god_nodes)
