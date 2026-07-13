# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""L1 Graph Data — deterministic, no LLM calls, no network.

Builds the vault's wikilink graph as node/edge lists (build_graph_data) and
detects topic communities via Louvain (detect_communities). Works with both CLI
and FS backends: triggers the driver index via graph_snapshot(), then reads
_graph / _unresolved_links / _notes directly to avoid O(N) subprocess calls on
the CLI backend.

Community detection via networkx.algorithms.community.louvain_communities
(built-in since networkx >= 3.0, already declared in pyproject.toml).
Degrades gracefully to no-community mode if unavailable.

The HTML viewer that renders this data (render_html / export_graph) lives in
silica.ui.web.graph_view — kept out of the kernel so this layer stays offline.
"""
from __future__ import annotations

import colorsys
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Community:
    id: int
    label: str
    color: str
    size: int

logger = logging.getLogger(__name__)


# Cluster colors: one distinct, vivid hue per community — the color encodes
# Louvain membership (real structure), so it must be unique per community and
# stable for a given id. Golden-angle hue rotation from brand cyan spreads hues
# evenly for any count; fixed high saturation + mid lightness keep them vivid and
# guarantee no color is ever black or white.
def _community_color(i: int) -> str:
    hue = (187.0 + i * 137.508) % 360.0          # 187° = brand cyan; 137.508° = golden angle
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, 0.56, 0.72)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


# Precomputed prefix for the legend default + tests; live code calls _community_color.
COMMUNITY_COLORS = [_community_color(i) for i in range(12)]

_EDGE_COLOR_EXTRACTED = "#8f8f8f"   # phosphor gray — resolved links
_EDGE_COLOR_AMBIGUOUS = "#ff2a2a"   # hazard red — unresolved (warning semantics)
_NODE_DEFAULT_COLOR = {"background": "#5c5c5c", "border": "#8f8f8f",
                       "highlight": {"background": "#8f8f8f", "border": "#eaeaea"}}
_NODE_GHOST_COLOR   = {"background": "#161616", "border": "#5c5c5c",
                       "highlight": {"background": "#262626", "border": "#8f8f8f"}}


def _infer_type(path: str) -> str:
    p = path.lower().replace("\\", "/")
    if "_inbox" in p or p.startswith("inbox/"):
        return "inbox"
    stem = Path(path).stem.lower()
    if "hub" in stem:
        return "hub"
    return "note"


# Silica's own generated files at the vault ROOT (log.md, GRAPH_REPORT.md) are
# tooling output, not knowledge notes. The driver indexes them like any note, so
# without this filter GRAPH_REPORT.md's hundreds of `[[...]]` would make it the
# top hub AND give every note it lists an incoming link — silently zeroing the
# orphan count on the next run. Matched by root-relative stem only, so a genuine
# note in a subfolder (e.g. "Concepts/log.md") stays in the graph.
_VAULT_ROOT_ARTIFACT_STEMS = frozenset({"log", "GRAPH_REPORT"})


def is_vault_artifact(note_id: str) -> bool:
    """True if `note_id` is a Silica-generated file at the vault root.

    Id form varies by caller (graph node ids carry `.md`; other callers may
    not), so this matches on the `.md`-stripped stem and requires no path
    separator — i.e. vault-root only.
    """
    stem = note_id.replace("\\", "/").removesuffix(".md")
    return "/" not in stem and stem in _VAULT_ROOT_ARTIFACT_STEMS


def build_graph_data(folder: str = "") -> tuple[list[dict], list[dict]]:
    """Build node and edge lists from the driver's internal nx.DiGraph.

    Calls driver.graph_snapshot() once to populate _graph, _notes, and
    _unresolved_links, then reads them directly. This avoids O(N) subprocess
    calls on the CLI backend.
    """
    from silica.driver import get_driver

    driver = get_driver()
    internal_notes, unresolved_links, internal_graph = driver.graph_data(folder=folder)

    def _in_scope(path: str) -> bool:
        if not folder:
            return True
        prefix = folder.rstrip("/") + "/"
        return path.startswith(prefix) or path == folder.rstrip("/")

    in_scope: set[str] = {
        p.replace("\\", "/") for p in internal_notes if _in_scope(p.replace("\\", "/"))
    }

    nodes: list[dict] = []
    for raw_path, ref in internal_notes.items():
        path = raw_path.replace("\\", "/")
        if path not in in_scope:
            continue
        if is_vault_artifact(path):   # keep Silica's own log.md/GRAPH_REPORT.md out of the graph
            continue
        nodes.append({
            "id":    path,
            "label": ref.name,
            "title": path,
            "type":  _infer_type(path),
            "group": -1,
            "color": dict(_NODE_DEFAULT_COLOR),
            "path":  path,
            "font":  {"color": "#e7ebf1", "size": 13},
            "size":  16,
        })

    node_ids: set[str] = {n["id"] for n in nodes}

    edges: list[dict] = []
    edge_set: set[tuple[str, str]] = set()
    edge_idx = 0

    for src_raw, tgt_raw in internal_graph.edges():
        src = src_raw.replace("\\", "/")
        tgt = tgt_raw.replace("\\", "/")
        if src not in node_ids or tgt not in node_ids:
            continue
        key = (src, tgt)
        if key in edge_set:
            continue
        edge_set.add(key)
        edges.append({
            "id":     f"e{edge_idx}",
            "from":   src,
            "to":     tgt,
            "type":   "EXTRACTED",
            "color":  {"color": _EDGE_COLOR_EXTRACTED, "opacity": 0.6},
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}},
            "width":  1.5,
        })
        edge_idx += 1

    ghost_nodes: dict[str, dict] = {}
    for src_raw, tgt_raw in unresolved_links:
        src = src_raw.replace("\\", "/")
        if src not in node_ids:
            continue
        tgt_name = tgt_raw.removesuffix(".md").rsplit("/", 1)[-1]
        ghost_id  = f"__unresolved__{tgt_name}"

        if ghost_id not in ghost_nodes:
            ghost_nodes[ghost_id] = {
                "id":           ghost_id,
                "label":        tgt_name,
                "title":        f"⚠ Unresolved: {tgt_name}",
                "type":         "ghost",
                "group":        -1,
                "color":        dict(_NODE_GHOST_COLOR),
                "path":         "",
                "font":         {"color": "#8a93a3", "size": 11},
                "size":         10,
                "borderWidth":  2,
                "borderDashes": True,
            }

        key = (src, ghost_id)
        if key not in edge_set:
            edge_set.add(key)
            edges.append({
                "id":     f"e{edge_idx}",
                "from":   src,
                "to":     ghost_id,
                "type":   "AMBIGUOUS",
                "color":  {"color": _EDGE_COLOR_AMBIGUOUS, "opacity": 0.4},
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
                "width":  1.0,
                "dashes": [4, 4],
            })
            edge_idx += 1

    nodes.extend(ghost_nodes.values())
    return nodes, edges


def detect_communities(nodes: list[dict], edges: list[dict]) -> list[Community]:
    """Louvain community detection on EXTRACTED edges, in-place.

    Assigns node["group"] (int) and node["color"]. Ghost nodes keep group == -1.
    Degrades gracefully if networkx < 3.0.

    Returns a list of Community objects with topic labels where available.
    """
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
    except (ImportError, AttributeError):
        logger.warning("graph_export: louvain_communities unavailable (networkx >= 3.0 required). Skipped.")
        return []

    real_ids = {n["id"] for n in nodes if n.get("type") != "ghost"}
    G = nx.Graph()
    G.add_nodes_from(real_ids)
    for e in edges:
        if e.get("type") == "EXTRACTED" and e["from"] in real_ids and e["to"] in real_ids:
            G.add_edge(e["from"], e["to"])

    if G.number_of_edges() == 0:
        logger.info("graph_export: no EXTRACTED edges — community detection skipped.")
        return []

    try:
        communities = louvain_communities(G, seed=42)
    except Exception as exc:
        logger.warning("graph_export: louvain_communities raised %s: %s", type(exc).__name__, exc)
        return []

    node_to_comm: dict[str, int] = {
        node_id: i
        for i, comm in enumerate(communities)
        for node_id in comm
    }

    for node in nodes:
        if node.get("type") == "ghost":
            continue
        comm_id = node_to_comm.get(node["id"], -1)
        node["group"] = comm_id
        if comm_id >= 0:
            color = _community_color(comm_id)
            node["color"] = {
                "background": color,
                "border":     color,
                "highlight":  {"background": color, "border": "#e7ebf1"},
            }

    # Fetch community labels from the co-occurrence index; degrade to {} on any failure.
    # Member ids carry '.md'; CooccurStore.note_nodes normalises via cooccur_key, so
    # no manual strip is needed here (single source of truth for the key).
    from silica.kernel.cooccurrence import get_cooccur_store
    try:
        labels = get_cooccur_store().community_labels([set(c) for c in communities])
    except Exception:
        labels = {}

    logger.info("graph_export: %d communities across %d nodes.", len(communities), len(real_ids))

    return [
        Community(
            id=i,
            label=labels.get(i, f"Cluster {i}"),
            color=_community_color(i),
            size=len(comm),
        )
        for i, comm in enumerate(communities)
    ]


def structural_gaps(
    nodes: list[dict], edges: list[dict], top_k: int = 10, min_size: int = 2
) -> list[tuple[int, int, str, str, int, float]]:
    """Community-pairs with the largest structural hole, score-descending.

    The mirror of the cross-cluster bridge signal: a bridge is a weak link that
    EXISTS between two areas; a structural gap is a pair of well-formed areas
    where the linking edges are ABSENT. Returns tuples
    (cluster_a, cluster_b, hub_a, hub_b, inter_edges, gap_score) with
    gap_score = size_a * size_b / (1 + inter_edges) — two large areas with no
    connecting edge rank highest.

    Reads node["group"] (set by detect_communities) and EXTRACTED edges, so it
    agrees node-for-node with the exported graph. Pure dict counting, no nx.
    ponytail: in a sparse vault this mostly ranks by size-product — that IS the
    signal (biggest disconnected areas). Upgrade to a modularity-gain score if
    it ever reads noisy.
    """
    group_of = {
        n["id"]: n["group"]
        for n in nodes
        if n.get("type") != "ghost" and n.get("group", -1) >= 0
    }
    if not group_of:
        return []

    sizes = Counter(group_of.values())
    deg: Counter = Counter()
    inter: Counter = Counter()  # (lo_cluster, hi_cluster) -> #EXTRACTED edges joining them
    for e in edges:
        if e.get("type") != "EXTRACTED":
            continue
        ga, gb = group_of.get(e.get("from")), group_of.get(e.get("to"))
        if ga is None or gb is None:
            continue
        deg[e["from"]] += 1
        deg[e["to"]] += 1
        if ga != gb:
            inter[(min(ga, gb), max(ga, gb))] += 1

    # Hub per cluster = highest-degree member (ties → larger id), matching
    # ClusterStat.hub so the report row and the overlay edge point at the same note.
    hub: dict[int, str] = {}
    for nid, g in group_of.items():
        if g not in hub or (deg[nid], nid) > (deg[hub[g]], hub[g]):
            hub[g] = nid

    big = sorted(g for g, s in sizes.items() if s >= min_size)
    gaps: list[tuple[int, int, str, str, int, float]] = []
    for i, ca in enumerate(big):
        for cb in big[i + 1:]:
            ie = inter.get((ca, cb), 0)
            score = sizes[ca] * sizes[cb] / (1 + ie)
            gaps.append((ca, cb, hub[ca], hub[cb], ie, round(score, 2)))
    gaps.sort(key=lambda t: (-t[5], t[0], t[1]))
    return gaps[:top_k]


def discourse_shape(n_nodes: int, giant: int, cluster_sizes: list[int]) -> str:
    """One-word topology diagnosis from primitives.

    The single source of the rule, shared by the /graph report Summary and the
    graph HUD badge so the two surfaces never disagree.

    Fragmented: the graph splits into pieces — the giant component holds under
                half the notes.
    Focused:    one area dominates — the largest linked cluster is over half the
                linked notes (a star around a few hubs).
    Diversified: several comparably-sized, well-connected areas.
    ponytail: two 0.5 thresholds on component-share and cluster-share. Good
    enough for a headline; tune the knobs if a real vault reads wrong.
    """
    if n_nodes <= 0:
        return ""
    if giant / n_nodes < 0.5:
        return "Fragmented"
    linked = sorted((s for s in cluster_sizes if s >= 2), reverse=True)
    if linked and linked[0] / sum(linked) > 0.5:
        return "Focused"
    return "Diversified"


# ---------------------------------------------------------------------------
# Vault cluster-context cache
#
# Per-note cluster membership ({path: {cluster_id, hub, is_hub}}) persisted
# under index_dir() so any layer can annotate a note with its graph community
# without re-running Louvain. Written by the router's payload phase
# (build_vault_graph_ctx) and by silica_vault_report; read best-effort by
# build_substrate and silica_related. Bounded staleness by design: a note
# added after the last write reads as "no cluster" until the next refresh.
# ---------------------------------------------------------------------------

def cluster_ctx_path() -> Path:
    # Function, not constant: resolves per current vault; tests monkeypatch it.
    from silica.kernel import paths

    return paths.index_dir() / "clusters_ctx.json"


def load_cluster_ctx() -> dict | None:
    """Full cache envelope {"sig": [nodes, edges], "ctx": {...}} or None."""
    import orjson

    p = cluster_ctx_path()
    if not p.exists():
        return None
    try:
        return orjson.loads(p.read_bytes())
    except Exception:
        return None


def save_cluster_ctx(sig: list[int], ctx: dict) -> None:
    import orjson

    try:
        p = cluster_ctx_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(orjson.dumps({"sig": sig, "ctx": ctx}))
    except Exception as e:
        logger.debug("graph_export: cluster ctx cache save skipped (%s)", e)


def cluster_ctx_map() -> dict[str, dict]:
    """Just the ctx map from the cached envelope; {} when cold or unreadable."""
    return (load_cluster_ctx() or {}).get("ctx") or {}


def ctx_from_report(report) -> dict[str, dict]:
    """Per-note cluster ctx from a VaultReport (duck-typed: importing
    graph_report here would be a cycle — its compute imports this module)."""
    ctx: dict[str, dict] = {}
    for cs in report.clusters:
        for member in cs.members:
            ctx[member] = {
                "cluster_id": cs.cluster_id,
                "hub": cs.hub,
                "is_hub": member == cs.hub,
            }
    # Isolated nodes too (cluster -1 = "no cluster" for consumers);
    # pagerank_map carries every real node id, zero-valued without analytics.
    for node_id in report.pagerank_map:
        if node_id not in ctx:
            ctx[node_id] = {"cluster_id": -1, "hub": None, "is_hub": False}
    return ctx


def graph_distances(source: str, *, folder: str = "") -> dict[str, int] | None:
    """BFS hop distances from `source` over the resolved wikilink graph.

    Returns {node_id (no .md): hops} for every note reachable from `source`,
    or None when the graph is unavailable or `source` is not in it. The
    structural complement of a semantic score: high similarity + absent/large
    distance = a missing link worth creating; distance 1 = already linked.
    ponytail: full snapshot + BFS per call; cache per run if it shows hot.
    """
    try:
        import networkx as nx

        nodes, edges = build_graph_data(folder=folder)
    except Exception:
        return None
    real = {n["id"] for n in nodes if n.get("type") != "ghost"}
    src = source if source in real else source + ".md"
    if src not in real:
        src = source.removesuffix(".md")
        if src not in real:
            return None
    G = nx.Graph()
    G.add_nodes_from(real)
    for e in edges:
        if e.get("type") == "EXTRACTED" and e["from"] in real and e["to"] in real:
            G.add_edge(e["from"], e["to"])
    dist = nx.single_source_shortest_path_length(G, src)
    return {p.removesuffix(".md"): d for p, d in dist.items()}


def cluster_hub_of(ctx: dict, path: str) -> str | None:
    """Short hub label of `path`'s cluster in a cached ctx map, or None.

    Ctx keys are driver graph node ids, whose .md suffix varies by backend;
    facade paths carry none — try both forms at the seam.
    """
    p = path.removesuffix(".md")
    gctx = ctx.get(p) or ctx.get(p + ".md")
    hub = (gctx or {}).get("hub")
    if not hub:
        return None
    return hub.rsplit("/", 1)[-1].removesuffix(".md")


def canvas_metrics(nodes: list[dict], edges: list[dict], k: int = 400) -> tuple[dict[str, float], int]:
    """One nx build over EXTRACTED edges → (betweenness per node, giant-component
    size), for node sizing and the discourse-shape badge on the exported graph.

    Kept separate from graph_report.compute_report (the report path) so the
    offline export doesn't drag in the full report machinery; the betweenness
    formula matches compute_report's so the two agree.
    ponytail: betweenness is O(V·E), sampled at k pivots to stay bounded on big
    vaults (k==n on small ones is exact). Empty/0 if networkx is unavailable.
    """
    try:
        import networkx as nx
    except ImportError:
        return {}, 0

    real = {n["id"] for n in nodes if n.get("type") != "ghost"}
    G = nx.Graph()
    G.add_nodes_from(real)
    for e in edges:
        if e.get("type") == "EXTRACTED" and e.get("from") in real and e.get("to") in real:
            G.add_edge(e["from"], e["to"])

    giant = max((len(c) for c in nx.connected_components(G)), default=0)
    bet: dict[str, float] = {}
    if G.number_of_edges() > 0:
        try:
            bet = nx.betweenness_centrality(
                G, k=min(G.number_of_nodes(), k), seed=42, normalized=True
            )
        except Exception:
            bet = {}
    return bet, giant
