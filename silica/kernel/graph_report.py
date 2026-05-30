"""L1 Graph Report — deterministic structural audit of the vault.

Builds a VaultReport from the driver's wikilink graph using only networkx
and the existing graph_export helpers. No LLM calls, no network access.

Principle: "embeddings PROPOSE, graph DISPOSES" — the report is authoritative
over vault structure; missing_links (embeddings) are clearly separated and
labelled as proposed candidates.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NodeStat:
    id: str           # vault-relative path (no .md)
    label: str        # display name
    cluster: int      # node["group"], -1 if none
    out_degree: int
    in_degree: int
    degree: int       # out+in
    pagerank: float   # rounded to 5 decimal places


@dataclass
class BridgeStat:
    source: str
    target: str
    source_cluster: int
    target_cluster: int
    weight: float     # surprise score: (deg(u)+deg(v)) / (1 + shared_neighbors)


@dataclass
class ClusterStat:
    cluster_id: int
    size: int
    hub: str | None        # highest-degree node in cluster
    members: list[str]     # capped at 25 in markdown, full in JSON
    cohesion: float        # intra-cluster edges / C(size,2)


@dataclass
class MissingLink:          # PROPOSED — not authoritative
    source: str
    target: str
    cosine: float


@dataclass
class VaultReport:
    generated_at: str
    scope: str
    totals: dict[str, int]
    god_nodes: list[NodeStat]
    bridges: list[BridgeStat]
    orphans: list[str]
    dangling: list[dict]   # [{"target": str, "refs": int}]
    clusters: list[ClusterStat]
    missing_links: list[MissingLink] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_report(
    folder: str = "",
    *,
    top_k: int = 10,
    with_embeddings: bool = False,
    _nodes_edges_override: tuple[list[dict], list[dict]] | None = None,
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
    totals = {
        "notes": len(real_ids),
        "links": n_links,
        "unresolved": n_unresolved,
        "orphans": len(orphans),
        "clusters": len(clusters),
    }

    report = VaultReport(
        generated_at=_now(),
        scope=folder,
        totals=totals,
        god_nodes=god_nodes,
        bridges=bridges,
        orphans=orphans,
        dangling=dangling,
        clusters=clusters,
    )

    # ------------------------------------------------------------------
    # Optional: missing links via embeddings
    # ------------------------------------------------------------------
    if with_embeddings:
        report.missing_links = _compute_missing_links(report, G_und, tau=0.82, k=top_k)

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
    )


def _compute_missing_links(
    report: VaultReport,
    G_und: Any,
    *,
    tau: float = 0.82,
    k: int = 10,
) -> list[MissingLink]:
    """Propose missing links via embedding similarity (PROPOSED — not authoritative)."""
    try:
        from silica.agent.providers import get_embedder
        from silica.config import CONFIG
        from silica.kernel.embed import EmbedStore
        import networkx as nx

        store = EmbedStore()
        if len(store) == 0:
            return []
        embedder = get_embedder(CONFIG)
    except Exception as exc:
        logger.debug("graph_report: embeddings unavailable (%s)", exc)
        return []

    god_paths = [n.id for n in report.god_nodes]
    results: list[MissingLink] = []
    seen: set[tuple[str, str]] = set()

    for source in god_paths:
        vec = store.get_vec(source)
        if vec is None:
            # try with .md stripped from path
            vec = store.get_vec(source.removesuffix(".md"))
        if vec is None:
            continue
        try:
            candidates = store.cosine_top_k(vec, k=k, exclude={source})
        except Exception:
            continue

        for cand in candidates:
            tgt = cand["path"]
            score = cand.get("score", 0.0)
            if score < tau:
                break  # results are sorted desc
            if tgt not in G_und or source not in G_und:
                continue
            # Skip if already adjacent or within 2 hops
            try:
                path_len = nx.shortest_path_length(G_und, source, tgt)
                if path_len <= 2:
                    continue
            except Exception:
                pass  # no path → candidate is valid
            key = (min(source, tgt), max(source, tgt))
            if key not in seen:
                seen.add(key)
                results.append(MissingLink(source=source, target=tgt, cosine=round(score, 4)))

    results.sort(key=lambda m: (-m.cosine, m.source, m.target))
    return results[:k]


# ---------------------------------------------------------------------------
# Output functions
# ---------------------------------------------------------------------------

_MEMBERS_CAP = 25  # max members shown per cluster in markdown


def to_markdown(report: VaultReport, title: str = "Silica Vault Report") -> str:
    """Render a VaultReport as OFM-friendly markdown."""
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append(f"_Generated: {report.generated_at}_")
    if report.scope:
        lines.append(f"_Scope: `{report.scope}`_")
    lines.append("")

    # Totals
    lines.append("## Totals")
    lines.append("| Metric | Count |")
    lines.append("|---|---|")
    for k, v in report.totals.items():
        lines.append(f"| {k.capitalize()} | {v} |")
    lines.append("")

    # God nodes
    lines.append("## God Nodes (High-Degree Hubs)")
    if report.god_nodes:
        lines.append("| Note | Cluster | Degree | In | Out | PageRank |")
        lines.append("|---|---|---|---|---|---|")
        for n in report.god_nodes:
            lines.append(f"| [[{n.label}]] | {n.cluster} | {n.degree} | {n.in_degree} | {n.out_degree} | {n.pagerank} |")
    else:
        lines.append("_No connected notes found._")
    lines.append("")

    # Surprising bridges
    lines.append("## Surprising Cross-Cluster Connections")
    if report.bridges:
        lines.append("| Source | Target | Clusters | Surprise |")
        lines.append("|---|---|---|---|")
        for b in report.bridges:
            src_label = b.source.rsplit("/", 1)[-1].removesuffix(".md")
            tgt_label = b.target.rsplit("/", 1)[-1].removesuffix(".md")
            lines.append(f"| [[{src_label}]] | [[{tgt_label}]] | {b.source_cluster}↔{b.target_cluster} | {b.weight} |")
    else:
        lines.append("_No cross-cluster bridges found._")
    lines.append("")

    # Clusters
    lines.append("## Clusters")
    if report.clusters:
        for c in report.clusters:
            hub_label = c.hub.rsplit("/", 1)[-1].removesuffix(".md") if c.hub else "—"
            lines.append(f"### Cluster {c.cluster_id} (size={c.size}, cohesion={c.cohesion})")
            lines.append(f"**Hub:** [[{hub_label}]]")
            members_shown = c.members[:_MEMBERS_CAP]
            member_links = ", ".join(
                f"[[{m.rsplit('/', 1)[-1].removesuffix('.md')}]]" for m in members_shown
            )
            if len(c.members) > _MEMBERS_CAP:
                member_links += f" … (+{len(c.members) - _MEMBERS_CAP} more)"
            lines.append(f"**Members:** {member_links}")
            lines.append("")
    else:
        lines.append("_No clusters detected (vault has no resolved wikilinks)._")
        lines.append("")

    # Orphans
    lines.append("## Orphans (No Incoming Links)")
    if report.orphans:
        for o in report.orphans:
            label = o.rsplit("/", 1)[-1].removesuffix(".md")
            lines.append(f"- [[{label}]]")
    else:
        lines.append("_No orphans._")
    lines.append("")

    # Dangling links
    lines.append("## Dangling Links (Unresolved Wikilinks)")
    if report.dangling:
        lines.append("| Target | References |")
        lines.append("|---|---|")
        for d in report.dangling:
            lines.append(f"| `{d['target']}` | {d['refs']} |")
    else:
        lines.append("_No unresolved wikilinks._")
    lines.append("")

    # Missing links (proposed)
    if report.missing_links:
        lines.append("## Proposed Missing Links _(embedding candidates — not authoritative)_")
        lines.append("| Source | Target | Cosine |")
        lines.append("|---|---|---|")
        for ml in report.missing_links:
            src_label = ml.source.rsplit("/", 1)[-1].removesuffix(".md")
            tgt_label = ml.target.rsplit("/", 1)[-1].removesuffix(".md")
            lines.append(f"| [[{src_label}]] | [[{tgt_label}]] | {ml.cosine} |")
        lines.append("")

    return "\n".join(lines)


def to_facts(report: VaultReport) -> dict:
    """Compact, stable subset for TaskLedger.facts (write-once, digest-friendly)."""
    return {
        "scope": report.scope,
        "totals": dict(report.totals),
        "god_nodes": [n.id for n in report.god_nodes],
        "top_bridges": [[b.source, b.target] for b in report.bridges[:5]],
        "orphan_count": report.totals.get("orphans", 0),
        "dangling_top": report.dangling[:5],
    }


def to_digest(report: VaultReport, *, max_items: int = 8) -> str:
    """Compact summary targeting < 500 tokens."""
    lines: list[str] = []
    t = report.totals
    lines.append(
        f"VAULT AUDIT  scope={report.scope or 'all'}  "
        f"notes={t.get('notes',0)}  links={t.get('links',0)}  "
        f"clusters={t.get('clusters',0)}  orphans={t.get('orphans',0)}  "
        f"unresolved={t.get('unresolved',0)}"
    )
    lines.append("─" * 36)

    if report.god_nodes:
        hubs = ", ".join(
            f"{n.label}(deg={n.degree})"
            for n in report.god_nodes[:max_items]
        )
        lines.append(f"TOP HUBS  {hubs}")

    if report.bridges:
        shown = report.bridges[:max_items]
        blist = ", ".join(
            f"{b.source.rsplit('/',1)[-1].removesuffix('.md')}↔{b.target.rsplit('/',1)[-1].removesuffix('.md')}(w={b.weight})"
            for b in shown
        )
        lines.append(f"BRIDGES  {blist}")

    if report.orphans:
        orp = ", ".join(
            o.rsplit("/", 1)[-1].removesuffix(".md")
            for o in report.orphans[:max_items]
        )
        extra = f" (+{len(report.orphans)-max_items} more)" if len(report.orphans) > max_items else ""
        lines.append(f"ORPHANS  {orp}{extra}")

    if report.dangling:
        dang = ", ".join(
            f"{d['target']}(×{d['refs']})"
            for d in report.dangling[:max_items]
        )
        lines.append(f"DANGLING  {dang}")

    if report.clusters:
        clist = ", ".join(
            f"C{c.cluster_id}(n={c.size},hub={c.hub.rsplit('/',1)[-1].removesuffix('.md') if c.hub else '-'})"
            for c in report.clusters[:max_items]
        )
        lines.append(f"CLUSTERS  {clist}")

    if report.missing_links:
        ml = ", ".join(
            f"{m.source.rsplit('/',1)[-1].removesuffix('.md')}→{m.target.rsplit('/',1)[-1].removesuffix('.md')}(cos={m.cosine})"
            for m in report.missing_links[:max_items]
        )
        lines.append(f"PROPOSED  {ml}")

    return "\n".join(lines)


def write_report(report: VaultReport, output_path: str) -> dict:
    """Write GRAPH_REPORT.md and report.json. Returns {path_md, path_json}."""
    import dataclasses

    out_md = Path(output_path)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(to_markdown(report), encoding="utf-8")

    out_json = out_md.with_suffix(".json")
    out_json.write_bytes(orjson.dumps(dataclasses.asdict(report), option=orjson.OPT_INDENT_2))

    logger.info(
        "graph_report: wrote %s and %s",
        out_md,
        out_json,
    )
    return {"path_md": str(out_md.resolve()), "path_json": str(out_json.resolve())}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
