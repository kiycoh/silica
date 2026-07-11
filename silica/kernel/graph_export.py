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
