"""Fix 5 — the vault report splits a cheap structural core from expensive analytics.

Ingest only needs Louvain clusters + orphans + degree (cluster routing, orphan
repair). PageRank, cross-cluster bridges, god-nodes and per-cluster cohesion are
read only by the on-demand /graph and /report commands. `compute_report` defaults
to the structural core (`analytics=False`); the commands opt into the full report.
"""
from __future__ import annotations

from silica.kernel.graph_report import compute_report


def _graph():
    """Two clusters (A,B,C / D,E), a cross-cluster bridge C->D, an orphan F."""
    nodes = [
        {"id": "A", "label": "Alpha", "group": 0, "type": "note"},
        {"id": "B", "label": "Beta", "group": 0, "type": "note"},
        {"id": "C", "label": "Gamma", "group": 0, "type": "note"},
        {"id": "D", "label": "Delta", "group": 1, "type": "note"},
        {"id": "E", "label": "Epsilon", "group": 1, "type": "note"},
        {"id": "F", "label": "Phi", "group": -1, "type": "note"},
    ]
    edges = [
        {"id": "e0", "from": "A", "to": "B", "type": "EXTRACTED"},
        {"id": "e1", "from": "B", "to": "C", "type": "EXTRACTED"},
        {"id": "e2", "from": "A", "to": "C", "type": "EXTRACTED"},
        {"id": "e3", "from": "D", "to": "E", "type": "EXTRACTED"},
        {"id": "e4", "from": "C", "to": "D", "type": "EXTRACTED"},
    ]
    return nodes, edges


def test_default_skips_analytics_keeps_structural():
    nodes, edges = _graph()
    r = compute_report(_nodes_edges_override=(nodes, edges))  # analytics=False default

    # Analytics dropped (not computed):
    assert r.god_nodes == []
    assert r.bridges == []
    assert all(v == 0.0 for v in r.pagerank_map.values())  # no nx.pagerank run
    assert all(c.cohesion == 0.0 for c in r.clusters)      # no per-cluster edge scan

    # Structural core kept:
    assert r.clusters                       # Louvain clusters
    assert "F" in r.orphans                 # orphan detection (in-degree 0)
    assert any(c.hub for c in r.clusters)   # hubs (degree-ranked)


def test_cluster_and_orphan_parity_across_flag():
    """The two ingest consumers must get identical cluster/orphan data either way."""
    nodes, edges = _graph()
    cheap = compute_report(_nodes_edges_override=(nodes, edges))
    full = compute_report(_nodes_edges_override=(nodes, edges), analytics=True)

    assert cheap.orphans == full.orphans
    key = lambda cs: (cs.cluster_id, cs.hub, tuple(cs.members))
    assert sorted(map(key, cheap.clusters)) == sorted(map(key, full.clusters))


def test_analytics_true_restores_full_report():
    nodes, edges = _graph()
    r = compute_report(_nodes_edges_override=(nodes, edges), analytics=True)
    assert r.god_nodes                                # god-nodes computed
    assert r.bridges                                  # cross-cluster bridges
    assert any(c.cohesion > 0.0 for c in r.clusters)  # per-cluster cohesion computed
    # PageRank actually runs (scipy is a declared dependency). Guards against the
    # silent-swallow regression where a missing backend left pagerank_map all-zero.
    assert any(v > 0.0 for v in r.pagerank_map.values())


def test_triage_note_reads_are_analytics_only(monkeypatch):
    """Triage (lean/reformat) reads EVERY note — its output is read only by the
    on-demand /graph,/report path (analyst_plan + render), never by ingest. So the
    structural core must not pay the per-note read; it is gated under analytics.
    """
    import silica.driver as drv

    calls: list[str] = []

    class FakeDriver:
        def read_note(self, nid):
            calls.append(nid)
            raise FileNotFoundError  # triage swallows per-note errors; we count reads

    monkeypatch.setattr(drv, "DRIVER", FakeDriver())
    nodes, edges = _graph()

    compute_report(_nodes_edges_override=(nodes, edges))  # analytics=False
    assert calls == [], "ingest path must not read note bodies for triage"

    compute_report(_nodes_edges_override=(nodes, edges), analytics=True)
    assert calls, "analytics path still runs triage"


def test_vault_graph_ctx_has_no_pagerank_field(monkeypatch):
    """The deleted dead field: per-note ctx carries only cluster_id/hub/is_hub."""
    import silica.kernel.graph_report as gr
    from silica.kernel.graph_report.models import ClusterStat, VaultReport
    from silica.router.states.setup import build_vault_graph_ctx

    fake = VaultReport(
        generated_at="t", scope="", totals={},
        god_nodes=[], bridges=[], orphans=[], dangling=[],
        clusters=[ClusterStat(cluster_id=0, size=2, hub="A", members=["A", "B"], cohesion=0.0)],
        pagerank_map={"A": 0.0, "B": 0.0, "Z": 0.0},  # Z is an isolated node
    )
    monkeypatch.setattr(gr, "compute_report", lambda *a, **k: fake)
    # build_vault_graph_ctx first snapshots the graph for the cluster-cache key
    # (Scaling E); keep this hermetic (don't touch the real vault).
    import silica.kernel.graph_export as ge
    monkeypatch.setattr(ge, "build_graph_data", lambda folder="": (
        [{"id": "A", "type": "note"}, {"id": "B", "type": "note"}, {"id": "Z", "type": "note"}],
        [{"from": "A", "to": "B", "type": "EXTRACTED"}],
    ))

    ctx = build_vault_graph_ctx(None)
    assert ctx, "ctx should be populated"
    for entry in ctx.values():
        assert set(entry) == {"cluster_id", "hub", "is_hub"}  # no 'pagerank'
    assert "Z" in ctx  # isolated node still enumerated


def test_attention_ranks_idle_and_weakly_linked_first():
    """learn-anything's time-decay surfacing: score = (days_idle+1)/(1+degree).
    An old leaf outranks a fresh hub; a note with no mtime abstains.
    """
    import time

    nodes = [
        {"id": "hub.md", "label": "Hub", "group": 0, "type": "note"},
        {"id": "old.md", "label": "Old", "group": 0, "type": "note"},
        {"id": "fresh.md", "label": "Fresh", "group": 0, "type": "note"},
        {"id": "nomt.md", "label": "NoMtime", "group": 0, "type": "note"},
    ]
    edges = [  # all three leaves point at the hub → hub degree 3, leaves degree 1
        {"id": "e0", "from": "old.md", "to": "hub.md", "type": "EXTRACTED"},
        {"id": "e1", "from": "fresh.md", "to": "hub.md", "type": "EXTRACTED"},
        {"id": "e2", "from": "nomt.md", "to": "hub.md", "type": "EXTRACTED"},
    ]
    now = time.time()
    mtimes = {
        "hub.md": now,                 # fresh, deg 3 → 1/4  = 0.25
        "old.md": now - 100 * 86400,   # 100d idle, deg 1 → 101/2 = 50.5
        "fresh.md": now,               # fresh, deg 1 → 1/2  = 0.5
        # nomt.md absent → abstains
    }
    r = compute_report(
        _nodes_edges_override=(nodes, edges), analytics=True, _mtimes_override=mtimes
    )

    paths = [a.path for a in r.attention_candidates]
    assert paths[0] == "old.md"            # most neglected floats up
    assert paths.index("old.md") < paths.index("hub.md")  # idle beats fresh hub
    assert "nomt.md" not in paths          # no recency signal → abstains
    assert r.totals["attention_candidates"] == 3


def test_attention_is_analytics_only():
    """Cheap structural core (ingest) never computes attention."""
    nodes = [
        {"id": "a.md", "label": "A", "group": 0, "type": "note"},
        {"id": "b.md", "label": "B", "group": 0, "type": "note"},
    ]
    edges = [{"id": "e0", "from": "a.md", "to": "b.md", "type": "EXTRACTED"}]
    r = compute_report(
        _nodes_edges_override=(nodes, edges), _mtimes_override={"a.md": 0.0, "b.md": 0.0}
    )  # analytics=False
    assert r.attention_candidates == []
