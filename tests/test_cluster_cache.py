"""Scaling Fix E — cache vault clusters to skip Louvain on small graph drift.

build_vault_graph_ctx runs Louvain (~3.1s at 10k notes) every run for the
per-note cluster/hub context. Clusters drift slowly, so the ctx is cached keyed
by a graph signature (node/edge counts) and reused while drift stays < ~2% —
Louvain recomputes only when the graph has grown enough to matter.
"""
from __future__ import annotations

import silica.kernel.graph_export as ge
import silica.kernel.graph_report as gr
import silica.router.states.setup as setup
from silica.kernel import paths
from silica.kernel.graph_report.models import ClusterStat, VaultReport


def _report():
    return VaultReport(
        generated_at="t", scope="", totals={},
        god_nodes=[], bridges=[], orphans=[], dangling=[],
        clusters=[ClusterStat(cluster_id=0, size=2, hub="a", members=["a", "b"], cohesion=0.0)],
        pagerank_map={"a": 0.0, "b": 0.0, "z": 0.0},  # z is isolated
    )


def _patch(monkeypatch, tmp_path, nodes, edges, counter):
    monkeypatch.setattr(paths, "index_dir", lambda: tmp_path)
    monkeypatch.setattr(ge, "build_graph_data", lambda folder="": (list(nodes), list(edges)))

    def fake_report(**kwargs):
        counter[0] += 1
        return _report()

    monkeypatch.setattr(gr, "compute_report", fake_report)


def test_cache_hit_skips_louvain(tmp_path, monkeypatch):
    nodes = [{"id": "a", "type": "note"}, {"id": "b", "type": "note"}, {"id": "z", "type": "note"}]
    edges = [{"from": "a", "to": "b", "type": "EXTRACTED"}]
    n = [0]
    _patch(monkeypatch, tmp_path, nodes, edges, n)

    ctx1 = setup.build_vault_graph_ctx(None)
    assert n[0] == 1                     # miss → Louvain ran
    assert ctx1["a"]["cluster_id"] == 0
    assert ctx1["z"]["cluster_id"] == -1  # isolated node enumerated

    ctx2 = setup.build_vault_graph_ctx(None)
    assert n[0] == 1                     # hit → compute_report NOT called again
    assert ctx2 == ctx1


def test_cache_recomputes_on_large_drift(tmp_path, monkeypatch):
    nodes = [{"id": "a", "type": "note"}, {"id": "b", "type": "note"}]
    edges = [{"from": "a", "to": "b", "type": "EXTRACTED"}]
    n = [0]
    _patch(monkeypatch, tmp_path, nodes, edges, n)

    setup.build_vault_graph_ctx(None)
    assert n[0] == 1

    # graph grows far beyond tolerance → recompute
    big_nodes = [{"id": f"n{i}", "type": "note"} for i in range(500)]
    big_edges = [{"from": f"n{i}", "to": f"n{i+1}", "type": "EXTRACTED"} for i in range(499)]
    monkeypatch.setattr(ge, "build_graph_data", lambda folder="": (big_nodes, big_edges))

    setup.build_vault_graph_ctx(None)
    assert n[0] == 2                     # drift > tol → Louvain re-ran


def test_small_drift_still_reuses_cache(tmp_path, monkeypatch):
    nodes = [{"id": f"n{i}", "type": "note"} for i in range(1000)]
    edges = [{"from": f"n{i}", "to": f"n{i+1}", "type": "EXTRACTED"} for i in range(999)]
    n = [0]
    _patch(monkeypatch, tmp_path, nodes, edges, n)

    setup.build_vault_graph_ctx(None)
    assert n[0] == 1

    # +5 nodes (< 2% of 1000) → still within tolerance → reuse
    nodes2 = nodes + [{"id": f"x{i}", "type": "note"} for i in range(5)]
    monkeypatch.setattr(ge, "build_graph_data", lambda folder="": (nodes2, list(edges)))
    setup.build_vault_graph_ctx(None)
    assert n[0] == 1                     # reused
