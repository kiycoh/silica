"""Core deterministic computation of the VaultReport.

Builds degree/PageRank/Louvain/bridge/orphan/dangling stats from the
driver's wikilink graph, then attaches the optional PROPOSED signal
sections computed by embed_signals and cooccur_delta.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from silica.kernel.graph_report.cooccur_delta import _compute_cooccur_delta
from silica.kernel.graph_report.embed_signals import (
    _compute_duplicate_pairs,
    _compute_missing_links,
)
from silica.kernel.graph_report.models import (
    BridgeStat,
    ClusterStat,
    NodeStat,
    VaultReport,
)

logger = logging.getLogger(__name__)


def compute_report(
    folder: str = "",
    *,
    top_k: int = 10,
    with_embeddings: bool = False,
    with_cooccurrence: bool = False,
    _nodes_edges_override: tuple[list[dict], list[dict]] | None = None,
    _cooccur_store_override: Any | None = None,
) -> VaultReport:
    """Build a VaultReport from the driver's wikilink graph.

    Uses build_graph_data + detect_communities from graph_export, then
    computes degree, PageRank, Louvain clusters, bridges, orphans, and
    dangling links from the resolved (EXTRACTED) edge set only.

    Pass _nodes_edges_override for testing without a live driver.
    """
    import networkx as nx
    from silica.kernel.graph_export import build_graph_data, detect_communities

    if _nodes_edges_override is not None:
        nodes, edges = _nodes_edges_override
        detect_communities(nodes, edges)
    else:
        try:
            nodes, edges = build_graph_data(folder=folder)
            detect_communities(nodes, edges)
        except Exception as exc:
            logger.warning("graph_report: build_graph_data failed (%s) — returning empty report", exc)
            return _empty_report(folder)

    # Split real nodes from ghost nodes
    real_nodes = [n for n in nodes if n.get("type") != "ghost"]
    real_ids: set[str] = {n["id"] for n in real_nodes}

    # Build undirected graph from EXTRACTED edges only (authoritative)
    G_und = nx.Graph()
    G_und.add_nodes_from(real_ids)
    for e in edges:
        if e.get("type") == "EXTRACTED" and e["from"] in real_ids and e["to"] in real_ids:
            G_und.add_edge(e["from"], e["to"])

    # Build directed graph for in/out-degree
    G_dir = nx.DiGraph()
    G_dir.add_nodes_from(real_ids)
    for e in edges:
        if e.get("type") == "EXTRACTED" and e["from"] in real_ids and e["to"] in real_ids:
            G_dir.add_edge(e["from"], e["to"])

    # Degree maps
    out_deg: dict[str, int] = dict(G_dir.out_degree())
    in_deg: dict[str, int] = dict(G_dir.in_degree())
    deg: dict[str, int] = {n: out_deg.get(n, 0) + in_deg.get(n, 0) for n in real_ids}

    # Triage for stylistic refinement and enrichment
    lean_notes: list[str] = []
    reformat_notes: list[str] = []
    try:
        from silica.kernel import ofm, frontmatter
        from silica.driver import DRIVER

        for nid in real_ids:
            try:
                nc = DRIVER.read_note(nid)
                if not nc.content:
                    continue
                data, _, body = frontmatter.split(nc.content)
                is_empty = len(body.strip()) == 0
                is_lean = ofm.is_lean(body)
                if is_empty or is_lean:
                    lean_notes.append(nid)
                elif data is None or frontmatter.lint_tags(data):
                    reformat_notes.append(nid)
            except Exception:
                pass
    except Exception as exc:
        logger.warning("graph_report: triage failed — %s", exc)

    # PageRank (deterministic)
    try:
        pr: dict[str, float] = nx.pagerank(G_und, max_iter=200) if G_und.number_of_edges() > 0 else {}
    except Exception:
        pr = {}

    # Cluster map from detect_communities output
    cluster_map: dict[str, int] = {n["id"]: n.get("group", -1) for n in real_nodes}

    # ------------------------------------------------------------------
    # God nodes (top-k by degree, tiebreak pagerank desc, then id asc)
    # ------------------------------------------------------------------
    sorted_nodes = sorted(
        real_ids,
        key=lambda n: (-deg.get(n, 0), -pr.get(n, 0.0), n),
    )
    god_nodes: list[NodeStat] = []
    node_label: dict[str, str] = {n["id"]: n.get("label", n["id"]) for n in real_nodes}
    for nid in sorted_nodes[:top_k]:
        god_nodes.append(NodeStat(
            id=nid,
            label=node_label.get(nid, nid),
            cluster=cluster_map.get(nid, -1),
            out_degree=out_deg.get(nid, 0),
            in_degree=in_deg.get(nid, 0),
            degree=deg.get(nid, 0),
            pagerank=round(pr.get(nid, 0.0), 5),
        ))

    # ------------------------------------------------------------------
    # Surprising cross-cluster bridges (top-k)
    # ------------------------------------------------------------------
    bridges: list[BridgeStat] = []
    seen_bridge: set[tuple[str, str]] = set()
    for u, v in G_und.edges():
        cu, cv = cluster_map.get(u, -1), cluster_map.get(v, -1)
        if cu < 0 or cv < 0 or cu == cv:
            continue
        shared = len(list(nx.common_neighbors(G_und, u, v)))
        weight = (deg.get(u, 0) + deg.get(v, 0)) / (1 + shared)
        key = (min(u, v), max(u, v))
        if key not in seen_bridge:
            seen_bridge.add(key)
            bridges.append(BridgeStat(
                source=u, target=v,
                source_cluster=cu, target_cluster=cv,
                weight=round(weight, 4),
            ))
    bridges.sort(key=lambda b: (-b.weight, b.source, b.target))
    bridges = bridges[:top_k]

    # ------------------------------------------------------------------
    # Clusters
    # ------------------------------------------------------------------
    cluster_members: dict[int, list[str]] = {}
    for nid in real_ids:
        cid = cluster_map.get(nid, -1)
        if cid >= 0:
            cluster_members.setdefault(cid, []).append(nid)

    clusters: list[ClusterStat] = []
    for cid, members in sorted(cluster_members.items()):
        size = len(members)
        hub_node = max(members, key=lambda n: (deg.get(n, 0), n)) if members else None
        # Cohesion: intra-cluster edges / possible pairs
        possible = size * (size - 1) / 2 if size >= 2 else 0
        intra = 0
        if possible > 0:
            member_set = set(members)
            for u, v in G_und.edges():
                if u in member_set and v in member_set:
                    intra += 1
        cohesion = round(intra / possible, 4) if possible > 0 else 0.0
        clusters.append(ClusterStat(
            cluster_id=cid,
            size=size,
            hub=hub_node,
            members=sorted(members),
            cohesion=cohesion,
        ))

    # ------------------------------------------------------------------
    # Orphans (in-degree == 0, scoped to folder)
    # ------------------------------------------------------------------
    orphans: list[str] = sorted(
        nid for nid in real_ids if in_deg.get(nid, 0) == 0
    )

    # ------------------------------------------------------------------
    # Dangling (unresolved wikilinks aggregated by target name)
    # ------------------------------------------------------------------
    ghost_refs: dict[str, int] = {}
    for e in edges:
        if e.get("type") == "AMBIGUOUS":
            tgt_id: str = e.get("to", "")
            # tgt_id is "__unresolved__<name>" from graph_export
            name = tgt_id.removeprefix("__unresolved__") if tgt_id.startswith("__unresolved__") else tgt_id
            ghost_refs[name] = ghost_refs.get(name, 0) + 1

    dangling: list[dict] = sorted(
        [{"target": t, "refs": c} for t, c in ghost_refs.items()],
        key=lambda d: (-d["refs"], d["target"]),
    )

    # ------------------------------------------------------------------
    # Totals
    # ------------------------------------------------------------------
    n_links = sum(1 for e in edges if e.get("type") == "EXTRACTED")
    n_unresolved = sum(1 for e in edges if e.get("type") == "AMBIGUOUS")

    # Initialize report shell to allow recursive calculation of totals if needed
    report = VaultReport(
        generated_at=_now(),
        scope=folder,
        totals={}, # Placeholder
        god_nodes=god_nodes,
        bridges=bridges,
        orphans=orphans,
        dangling=dangling,
        clusters=clusters,
        pagerank_map={nid: round(pr.get(nid, 0.0), 5) for nid in real_ids},
        lean_notes=lean_notes,
        reformat_notes=reformat_notes,
    )

    if with_embeddings:
        report.missing_links = _compute_missing_links(report, G_und, tau=0.82, k=top_k)
        report.duplicate_pairs, report.confirmed_duplicate_pairs = _compute_duplicate_pairs(report)

    if with_cooccurrence:
        autolinks, stale, hubs = _compute_cooccur_delta(
            report, G_und, node_label,
            cooccur_store=_cooccur_store_override, k=top_k,
        )
        report.autolink_candidates = autolinks
        report.stale_links = stale
        report.missing_hubs = hubs

    totals = {
        "notes": len(real_ids),
        "links": n_links,
        "dangling_links": len(dangling),
        "missing_links": len(report.missing_links),
        "duplicate_pairs": len(report.duplicate_pairs),
        "confirmed_duplicates": len(report.confirmed_duplicate_pairs),
        "autolink_candidates": len(report.autolink_candidates),
        "stale_links": len(report.stale_links),
        "missing_hubs": len(report.missing_hubs),
        "lean_notes": len(lean_notes),
        "reformat_notes": len(reformat_notes),
        "orphans": len(orphans),
        "clusters": len(clusters),
    }
    report.totals = totals

    return report


def _empty_report(scope: str = "") -> VaultReport:
    return VaultReport(
        generated_at=_now(),
        scope=scope,
        totals={"notes": 0, "links": 0, "unresolved": 0, "orphans": 0, "clusters": 0},
        god_nodes=[],
        bridges=[],
        orphans=[],
        dangling=[],
        clusters=[],
        missing_links=[],
        duplicate_pairs=[],
        lean_notes=[],
        reformat_notes=[],
        pagerank_map={},
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
