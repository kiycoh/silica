# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Core deterministic computation of the VaultReport.

Builds degree/PageRank/Louvain/bridge/orphan/dangling stats from the
driver's wikilink graph, then attaches the optional PROPOSED signal
sections computed by embed_signals and cooccur_delta.
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from silica.kernel.graph_report.cooccur_delta import _compute_cooccur_delta
from silica.kernel.graph_report.embed_signals import (
    _compute_duplicate_pairs,
    _compute_missing_links,
)
from silica.kernel.graph_report.models import (
    AttentionCandidate,
    BridgeStat,
    ClusterStat,
    ContestedNote,
    NodeStat,
    SourceDrift,
    StructuralGap,
    VaultReport,
)

logger = logging.getLogger(__name__)


def compute_report(
    folder: str = "",
    *,
    top_k: int = 10,
    analytics: bool = False,
    with_embeddings: bool = False,
    with_cooccurrence: bool = False,
    _nodes_edges_override: tuple[list[dict], list[dict]] | None = None,
    _cooccur_store_override: Any | None = None,
    _mtimes_override: dict[str, float] | None = None,
) -> VaultReport:
    """Build a VaultReport from the driver's wikilink graph.

    Uses build_graph_data + detect_communities from graph_export, then
    computes degree, Louvain clusters, orphans, and dangling links from the
    resolved (EXTRACTED) edge set only — the cheap *structural core* nucleate
    reads (cluster routing + orphan repair).

    `analytics=True` additionally computes the expensive read-only signals that
    only the on-demand /graph and /report commands consume: PageRank, god-nodes,
    cross-cluster bridges, and per-cluster cohesion. Nucleate leaves it False to
    skip the 200-iteration PageRank and the bridge/cohesion edge traversals.

    Pass _nodes_edges_override for testing without a live driver.
    """
    import networkx as nx
    from silica.kernel.graph_export import (
        build_graph_data,
        detect_communities,
        structural_gaps,
    )

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

    # Triage for stylistic refinement and enrichment — analytics-only. It reads
    # EVERY note body (the dominant report cost on a large vault) and its output
    # (lean_notes/reformat_notes) is consumed only by build_task_plan + render on
    # the /graph,/report path. Nucleate never reads it, so the structural core skips
    # the per-note read entirely.
    lean_notes: list[str] = []
    reformat_notes: list[str] = []
    contested: list[ContestedNote] = []
    if analytics:
        try:
            from silica.kernel import ofm, frontmatter
            from silica.driver import DRIVER

            for nid in real_ids:
                try:
                    nc = DRIVER.read_note(nid)
                    if not nc.content:
                        continue
                    data, _, body = frontmatter.split(nc.content)
                    if data and data.get("contested"):
                        contested.append(
                            ContestedNote(path=nid, refs=list(data.get("contradictions") or []))
                        )
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

    # Source drift (spec-hermes-coherence §3) — analytics-only for parity with
    # the other on-demand /report signals above, though the read itself is
    # cheap (one <vault>/provenance.json parse, no per-note driver reads).
    source_drift: list[SourceDrift] = []
    if analytics:
        try:
            from silica.kernel.provenance import drifted_notes

            # Provenance notes are recorded WITHOUT the `.md` extension
            # (RunManifestEntry.path strips it), but graph node ids (real_ids)
            # carry `.md` (driver index keys) — strip at the seam before
            # intersecting, per codebase convention.
            real_stems = {i.removesuffix(".md") for i in real_ids}
            source_drift = [
                SourceDrift(note=note, source=source)
                for note, source in drifted_notes()
                if note in real_stems
            ]
        except Exception as exc:
            logger.warning("graph_report: source drift check failed — %s", exc)

    # PageRank — analytics-only (200-iteration power method); the structural
    # core leaves it empty so god-node tiebreaks and pagerank_map are all-zero.
    pr: dict[str, float] = {}
    if analytics:
        try:
            pr = nx.pagerank(G_und, max_iter=200) if G_und.number_of_edges() > 0 else {}
        except Exception:
            pr = {}

    # Betweenness — analytics-only (O(V·E), the most expensive metric). Sampled
    # at k<=400 pivots so it stays bounded on big vaults; k==n (small vaults) is
    # exact. seed fixed for deterministic output. Distinct from degree: it flags
    # bottleneck nodes whose removal fragments the discourse.
    # ponytail: k-sampled approximation; drop k= for exact if a vault is small
    # and you need the last digit.
    bet: dict[str, float] = {}
    if analytics and G_und.number_of_edges() > 0:
        try:
            bet = nx.betweenness_centrality(
                G_und, k=min(G_und.number_of_nodes(), 400), seed=42, normalized=True
            )
        except Exception:
            bet = {}

    # ------------------------------------------------------------------
    # Attention candidates — analytics-only. Spaced-repetition
    # surfacing, embedder-free: a note untouched for long AND weakly linked
    # floats up.  score = (days_idle + 1) / (1 + degree)  — pure ranking, no
    # weights, no config.  degree stands in for a per-note "confidence" (a
    # well-integrated note is trusted); adding a real confidence field would be
    # new source of truth to maintain — out of charter (derived data only).
    #
    # ponytail: mtime = "last touch by ANYONE", not "last human review" — a bulk
    # op (AI:true stamps, autolink) resets it and the list starts blind, then
    # repopulates over time.  Upgrade path if it misranks in practice: git
    # last-commit-per-file (SILICA_GIT_COMMIT vaults) or a last_reviewed stamp.
    attention: list[AttentionCandidate] = []
    if analytics:
        mtimes = _mtimes_override
        if mtimes is None:
            mtimes = {}
            from silica.driver import DRIVER

            mtime_of = getattr(DRIVER, "mtime_of", None)
            if mtime_of is not None:
                for nid in real_ids:
                    try:
                        ts = mtime_of(nid)
                    except Exception:
                        ts = None
                    if ts is not None:
                        mtimes[nid] = ts
        if mtimes:
            now_ts = datetime.now(timezone.utc).timestamp()
            for nid in real_ids:
                ts = mtimes.get(nid)
                if ts is None:
                    continue  # abstain: no recency signal for this note
                days_idle = max(0, int((now_ts - ts) // 86400))
                d = deg.get(nid, 0)
                attention.append(AttentionCandidate(
                    path=nid, days_idle=days_idle, degree=d,
                    score=round((days_idle + 1) / (1 + d), 3),
                ))
            attention.sort(key=lambda a: (-a.score, a.path))
            attention = attention[:top_k]

    # Cluster map from detect_communities output
    cluster_map: dict[str, int] = {n["id"]: n.get("group", -1) for n in real_nodes}
    node_label: dict[str, str] = {n["id"]: n.get("label", n["id"]) for n in real_nodes}

    # ------------------------------------------------------------------
    # God nodes + cross-cluster bridges — analytics-only (read by /graph,
    # /report; nucleate never touches them). Skipped for the structural core.
    # ------------------------------------------------------------------
    god_nodes: list[NodeStat] = []
    bridges: list[BridgeStat] = []
    if analytics:
        sorted_nodes = sorted(
            real_ids,
            key=lambda n: (-deg.get(n, 0), -pr.get(n, 0.0), n),
        )
        for nid in sorted_nodes[:top_k]:
            god_nodes.append(NodeStat(
                id=nid,
                label=node_label.get(nid, nid),
                cluster=cluster_map.get(nid, -1),
                out_degree=out_deg.get(nid, 0),
                in_degree=in_deg.get(nid, 0),
                degree=deg.get(nid, 0),
                pagerank=round(pr.get(nid, 0.0), 5),
                betweenness=round(bet.get(nid, 0.0), 4),
            ))

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
        # Cohesion (intra-cluster edges / possible pairs) — analytics-only: an
        # edge scan per cluster. The structural core leaves it 0.0.
        cohesion = 0.0
        possible = size * (size - 1) / 2 if size >= 2 else 0
        if analytics and possible > 0:
            member_set = set(members)
            intra = sum(1 for u, v in G_und.edges() if u in member_set and v in member_set)
            cohesion = round(intra / possible, 4)
        clusters.append(ClusterStat(
            cluster_id=cid,
            size=size,
            hub=hub_node,
            members=sorted(members),
            cohesion=cohesion,
        ))

    # Structural gaps + discourse shape — analytics-only, mirror of the bridge
    # signal (areas that SHOULD connect but don't) plus a one-word topology read.
    structural_gaps_list: list[StructuralGap] = []
    discourse_state = ""
    if analytics:
        structural_gaps_list = [
            StructuralGap(
                cluster_a=ca, cluster_b=cb, hub_a=ha, hub_b=hb,
                inter_edges=ie, gap_score=score,
            )
            for ca, cb, ha, hb, ie, score in structural_gaps(nodes, edges, top_k=top_k)
        ]
        discourse_state = _discourse_state(G_und, clusters)

    # ------------------------------------------------------------------
    # Orphans (in-degree == 0, scoped to folder)
    # ------------------------------------------------------------------
    orphans: list[str] = sorted(
        nid for nid in real_ids if in_deg.get(nid, 0) == 0
    )

    # ------------------------------------------------------------------
    # Dangling (unresolved wikilinks aggregated by target name)
    # ------------------------------------------------------------------
    # edge target is "__unresolved__<name>" from graph_export
    ghost_refs = Counter(
        e.get("to", "").removeprefix("__unresolved__")
        for e in edges
        if e.get("type") == "AMBIGUOUS"
    )

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
        betweenness_map={nid: round(bet.get(nid, 0.0), 4) for nid in real_ids},
        attention_candidates=attention,
        lean_notes=lean_notes,
        reformat_notes=reformat_notes,
        contested=contested,
        source_drift=source_drift,
        structural_gaps=structural_gaps_list,
        discourse_state=discourse_state,
    )

    if with_embeddings:
        report.missing_links = _compute_missing_links(report, G_und, tau=0.82, k=top_k)
        report.duplicate_pairs, report.confirmed_duplicate_pairs = _compute_duplicate_pairs(report)

    if with_cooccurrence:
        autolinks, stale, hubs, deficits = _compute_cooccur_delta(
            report, G_und, node_label,
            cooccur_store=_cooccur_store_override, k=top_k,
        )
        report.autolink_candidates = autolinks
        report.stale_links = stale
        report.missing_hubs = hubs
        report.integration_deficits = deficits

    if analytics:
        try:
            from silica.config import CONFIG as _CFG
            from silica.kernel.graph_report.code_signals import _compute_code_signals
            vault_path = getattr(_CFG, "vault_path", "") or ""
            if vault_path:
                wl = {(min(u, v), max(u, v)) for u, v in G_und.edges()}
                cov, import_autolinks = _compute_code_signals(vault_path, wl)
                report.code_coverage = cov
                if import_autolinks:
                    report.autolink_candidates = list(report.autolink_candidates) + import_autolinks
        except Exception as exc:
            logger.warning("graph_report: code signals skipped — %s", exc)

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
        "integration_deficits": len(report.integration_deficits),
        "attention_candidates": len(attention),
        "lean_notes": len(lean_notes),
        "reformat_notes": len(reformat_notes),
        "contested": len(contested),
        "source_drift": len(source_drift),
        "orphans": len(orphans),
        "clusters": len(clusters),
        "structural_gaps": len(structural_gaps_list),
        "code_files_documented": (report.code_coverage.documented if report.code_coverage else 0),
        "code_files_total": (report.code_coverage.total if report.code_coverage else 0),
    }
    report.totals = totals

    return report


def _discourse_state(G_und, clusters: list[ClusterStat]) -> str:
    """Report-side wrapper: measure giant-component share on G_und, then apply
    the shared discourse_shape rule (single source, also used by the graph HUD)."""
    import networkx as nx

    from silica.kernel.graph_export import discourse_shape

    giant = max((len(c) for c in nx.connected_components(G_und)), default=0)
    return discourse_shape(G_und.number_of_nodes(), giant, [c.size for c in clusters])


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
