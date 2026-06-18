"""L1 Graph Export — deterministic, no LLM calls.

Builds a self-contained 3d-force-graph HTML visualization from the vault's
wikilink graph. Works with both CLI and FS backends: triggers the driver index
via graph_snapshot(), then reads _graph / _unresolved_links / _notes directly
to avoid O(N) subprocess calls on the CLI backend.

Community detection via networkx.algorithms.community.louvain_communities
(built-in since networkx >= 3.0, already declared in pyproject.toml).
Degrades gracefully to no-community mode if unavailable.
"""
from __future__ import annotations

import html
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass
class Community:
    id: int
    label: str
    color: str
    size: int

logger = logging.getLogger(__name__)

_VIS_JS_URL  = "https://cdn.jsdelivr.net/npm/3d-force-graph@1.80.0/dist/3d-force-graph.min.js"


def _fetch(url: str) -> str:
    return httpx.get(url, timeout=30).raise_for_status().text


def _fetch_lib_js() -> str:
    """Fetch 3d-force-graph from CDN. Raises RuntimeError with a clear message on failure."""
    try:
        logger.info("graph_export: fetching 3d-force-graph from CDN…")
        js = _fetch(_VIS_JS_URL)
        logger.info("graph_export: 3d-force-graph fetched (%.0f KB).", len(js) / 1024)
        return js
    except Exception as exc:
        raise RuntimeError(
            f"graph_export: failed to fetch 3d-force-graph from CDN — {exc}\n"
            "Check your internet connection and try again."
        ) from exc

COMMUNITY_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F0B27A", "#82E0AA", "#F1948A", "#AED6F1", "#A9DFBF",
    "#F8C471", "#7FB3D3", "#A3E4D7", "#F9E79F", "#D7BDE2",
]
_EDGE_COLOR_EXTRACTED = "#4a9eff"
_EDGE_COLOR_AMBIGUOUS = "#ffaa33"
_NODE_DEFAULT_COLOR = {"background": "#2d4a6e", "border": "#4a9eff",
                       "highlight": {"background": "#3a5f8a", "border": "#7ec8ff"}}
_NODE_GHOST_COLOR   = {"background": "#3d2020", "border": "#ff6b6b",
                       "highlight": {"background": "#4d2a2a", "border": "#ff9999"}}


def _infer_type(path: str) -> str:
    p = path.lower().replace("\\", "/")
    if "_inbox" in p or p.startswith("inbox/"):
        return "inbox"
    stem = Path(path).stem.lower()
    if "hub" in stem:
        return "hub"
    return "note"


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
        nodes.append({
            "id":    path,
            "label": ref.name,
            "title": path,
            "type":  _infer_type(path),
            "group": -1,
            "color": dict(_NODE_DEFAULT_COLOR),
            "path":  path,
            "font":  {"color": "#e0e0e0", "size": 13},
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
                "font":         {"color": "#ff9999", "size": 11},
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
            color = COMMUNITY_COLORS[comm_id % len(COMMUNITY_COLORS)]
            node["color"] = {
                "background": color,
                "border":     color,
                "highlight":  {"background": color, "border": "#ffffff"},
            }

    # Fetch community labels from the co-occurrence index; degrade to {} on any failure.
    from silica.kernel.cooccurrence import CooccurStore
    try:
        labels = CooccurStore().community_labels(
            [{m.removesuffix(".md") for m in c} for c in communities]
        )
    except Exception:
        labels = {}

    logger.info("graph_export: %d communities across %d nodes.", len(communities), len(real_ids))

    return [
        Community(
            id=i,
            label=labels.get(i, f"Cluster {i}"),
            color=COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)],
            size=len(comm),
        )
        for i, comm in enumerate(communities)
    ]


def render_html(
    nodes: list[dict],
    edges: list[dict],
    communities: "list[Community]" = (),  # type: ignore[assignment]
    title: str = "Silica Knowledge Graph",
    lib_js: str = "",
) -> str:
    """Produce a fully self-contained 3d-force-graph HTML string.

    Pass lib_js to embed the bundle inline (truly offline-capable).
    If omitted, CDN link is used as a fallback.
    communities is a list of Community objects; legend is built from it.
    """
    nodes_json = json.dumps(nodes, ensure_ascii=False).replace("</", "<\\/")
    edges_json = json.dumps(edges, ensure_ascii=False).replace("</", "<\\/")

    n_notes      = sum(1 for n in nodes if n.get("type") != "ghost")
    n_ghost      = sum(1 for n in nodes if n.get("type") == "ghost")
    n_extracted  = sum(1 for e in edges if e.get("type") == "EXTRACTED")
    n_ambiguous  = sum(1 for e in edges if e.get("type") == "AMBIGUOUS")
    n_communities = len(communities)

    legend_items = "".join(
        f'<div class="legend-item" data-community="{c.id}" onclick="filterCommunity({c.id})">'
        f'<span class="dot" style="background:{c.color}"></span>{html.escape(c.label)} '
        f'<span style="color:#555;font-size:11px;margin-left:auto">{c.size}</span>'
        f'</div>\n'
        for c in communities
    )

    comm_labels_json = json.dumps(
        {c.id: c.label for c in communities}, ensure_ascii=False
    ).replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  {f'<script>{lib_js}</script>' if lib_js else '<script src="' + _VIS_JS_URL + '"></script>'}
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{display:flex;height:100vh;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
          background:#0f0f1a;color:#d4d4d4;overflow:hidden}}
    #sidebar{{width:240px;flex-shrink:0;background:#141427;border-right:1px solid #2a2a4a;
              display:flex;flex-direction:column;padding:14px 12px;gap:14px;overflow-y:auto}}
    #sidebar h1{{font-size:13px;color:#7ec8ff;font-weight:600;letter-spacing:.5px}}
    .stat-grid{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}
    .stat{{background:#1e1e38;border-radius:6px;padding:8px;text-align:center}}
    .stat .val{{font-size:20px;font-weight:700;color:#7ec8ff}}
    .stat .lbl{{font-size:10px;color:#888;margin-top:2px}}
    #search{{width:100%;padding:7px 10px;background:#1e1e38;border:1px solid #2a2a4a;
             border-radius:6px;color:#d4d4d4;font-size:13px;outline:none}}
    #search:focus{{border-color:#4a9eff}}
    .section-title{{font-size:11px;color:#666;text-transform:uppercase;letter-spacing:.8px}}
    .filter-row{{display:flex;align-items:center;gap:7px;font-size:12px;cursor:pointer;
                 padding:3px 0;user-select:none}}
    .filter-row input{{cursor:pointer;accent-color:#4a9eff}}
    .dot-edge{{width:24px;height:3px;border-radius:2px;flex-shrink:0}}
    #legend-box{{display:flex;flex-direction:column;gap:2px;max-height:200px;overflow-y:auto}}
    .legend-item{{display:flex;align-items:center;gap:6px;font-size:12px;cursor:pointer;
                  padding:3px 6px;border-radius:4px}}
    .legend-item:hover{{background:#1e1e38}}
    .legend-item.active{{background:#1e2a3a;outline:1px solid #4a9eff}}
    .dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
    .btn{{padding:7px 10px;background:#1e1e38;border:1px solid #2a2a4a;border-radius:6px;
           color:#aaa;font-size:12px;cursor:pointer;text-align:center}}
    .btn:hover{{border-color:#4a9eff;color:#7ec8ff}}
    #graph-wrap{{flex:1;position:relative}}
    #graph{{width:100%;height:100%}}
    #drawer{{width:260px;flex-shrink:0;background:#141427;border-left:1px solid #2a2a4a;
             padding:16px 14px;overflow-y:auto;display:none;flex-direction:column;gap:12px}}
    #drawer.open{{display:flex}}
    #drawer-title{{font-size:15px;font-weight:600;color:#7ec8ff;word-break:break-word}}
    #drawer-path{{font-size:11px;color:#555;word-break:break-all}}
    #drawer-meta{{font-size:12px;color:#aaa}}
    .drawer-section{{display:flex;flex-direction:column;gap:4px}}
    .drawer-label{{font-size:10px;color:#555;text-transform:uppercase;letter-spacing:.8px}}
    .drawer-val{{font-size:13px;color:#ccc}}
    .tag{{display:inline-block;padding:2px 7px;background:#1e1e38;border-radius:10px;
           font-size:11px;color:#4a9eff;margin:2px}}
    #close-drawer{{align-self:flex-end;cursor:pointer;color:#555;font-size:18px;line-height:1}}
    #close-drawer:hover{{color:#ccc}}
  </style>
</head>
<body>

<div id="sidebar">
  <h1>&#11041; {title}</h1>

  <div class="stat-grid">
    <div class="stat"><div class="val">{n_notes}</div><div class="lbl">Notes</div></div>
    <div class="stat"><div class="val">{n_extracted}</div><div class="lbl">Links</div></div>
    <div class="stat"><div class="val">{n_communities}</div><div class="lbl">Clusters</div></div>
    <div class="stat"><div class="val">{n_ghost}</div><div class="lbl">Unresolved</div></div>
  </div>

  <input id="search" type="text" placeholder="Search notes&#8230;" oninput="onSearch(this.value)">

  <div>
    <div class="section-title" style="margin-bottom:8px">Edge types</div>
    <label class="filter-row">
      <input type="checkbox" id="cb-extracted" checked onchange="updateEdgeFilter()">
      <div class="dot-edge" style="background:#4a9eff"></div>
      Resolved
      <span style="color:#555;font-size:11px;margin-left:auto">{n_extracted}</span>
    </label>
    <label class="filter-row" style="margin-top:4px">
      <input type="checkbox" id="cb-ambiguous" onchange="updateEdgeFilter()">
      <div class="dot-edge" style="background:#ffaa33"></div>
      Unresolved
      <span style="color:#555;font-size:11px;margin-left:auto">{n_ambiguous}</span>
    </label>
  </div>

  <div>
    <div class="section-title" style="margin-bottom:6px">Communities</div>
    <div id="legend-box">
{legend_items}      <div class="legend-item active" id="legend-all" onclick="filterCommunity(-2)">
        <span class="dot" style="background:#555"></span>Show all
      </div>
    </div>
  </div>

  <div class="btn" onclick="Graph.zoomToFit(400)">&#8862; Fit graph</div>
</div>

<div id="graph-wrap"><div id="graph"></div></div>

<div id="drawer">
  <span id="close-drawer" onclick="closeDrawer()">&#10005;</span>
  <div id="drawer-title">&#8212;</div>
  <div id="drawer-path"></div>
  <div id="drawer-meta"></div>
  <div class="drawer-section">
    <div class="drawer-label">Out-links</div>
    <div id="drawer-out" class="drawer-val">&#8212;</div>
  </div>
  <div class="drawer-section">
    <div class="drawer-label">Backlinks</div>
    <div id="drawer-in" class="drawer-val">&#8212;</div>
  </div>
  <div id="drawer-tags-section" class="drawer-section" style="display:none">
    <div class="drawer-label">Tags</div>
    <div id="drawer-tags"></div>
  </div>
</div>

<script>
const RAW_NODES = {nodes_json};
const RAW_EDGES = {edges_json};
const COMM_LABELS = {comm_labels_json};

const outDeg = {{}}, inDeg = {{}};
RAW_EDGES.forEach(e => {{
  outDeg[e.from] = (outDeg[e.from] || 0) + 1;
  inDeg[e.to]   = (inDeg[e.to]   || 0) + 1;
}});

let activeCommunity = -2;
let showExtracted = true;
let showAmbiguous = false;
let searchQuery = "";

const Graph = new ForceGraph3D(document.getElementById("graph"))
  .backgroundColor("#0f0f1a")
  .graphData({{ nodes: RAW_NODES, links: RAW_EDGES }})
  .linkSource("from").linkTarget("to")
  .nodeLabel("label").nodeVal("size")
  .nodeColor(n => (n.color && n.color.background) || "#888")
  .linkColor(l => (l.color && l.color.color) || "#4a9eff")
  .linkWidth(l => l.width || 1)
  .linkDirectionalArrowLength(l => l.type === "AMBIGUOUS" ? 2.5 : 3.5)
  .linkDirectionalArrowRelPos(1)
  .nodeVisibility(n => !n._hidden)
  .linkVisibility(l => !l._hidden);

function applyFilters() {{
  RAW_NODES.forEach(n => {{
    n._hidden = (activeCommunity !== -2 && n.group !== activeCommunity) ||
                (!!searchQuery && !n.label.toLowerCase().includes(searchQuery));
  }});
  RAW_EDGES.forEach(e => {{
    e._hidden = (e.type === "EXTRACTED" && !showExtracted) ||
                (e.type === "AMBIGUOUS" && !showAmbiguous);
  }});
  // Re-pass the current accessor to force a visibility refresh without resetting the physics layout
  Graph.nodeVisibility(Graph.nodeVisibility());
  Graph.linkVisibility(Graph.linkVisibility());
}}

function updateEdgeFilter() {{
  showExtracted = document.getElementById("cb-extracted").checked;
  showAmbiguous = document.getElementById("cb-ambiguous").checked;
  applyFilters();
}}

function filterCommunity(cid) {{
  activeCommunity = cid;
  document.querySelectorAll(".legend-item").forEach(el => el.classList.remove("active"));
  const el = cid === -2
    ? document.getElementById("legend-all")
    : document.querySelector(`[data-community="${{cid}}"]`);
  if (el) el.classList.add("active");
  applyFilters();
}}

function onSearch(q) {{
  searchQuery = q.trim().toLowerCase();
  applyFilters();
}}

Graph.onNodeClick(node => {{
  document.getElementById("drawer-title").textContent = node.label;
  document.getElementById("drawer-path").textContent  = node.path || "(ghost node)";
  const commText = (Number.isInteger(node.group) && node.group >= 0 && COMM_LABELS[node.group])
    ? ` · ${{COMM_LABELS[node.group]}}` : "";
  document.getElementById("drawer-meta").textContent = `${{node.type}}${{commText}}`;
  document.getElementById("drawer-out").textContent = outDeg[node.id] || 0;
  document.getElementById("drawer-in").textContent  = inDeg[node.id]  || 0;

  const tagsSection = document.getElementById("drawer-tags-section");
  const tags = node.tags || [];
  if (tags.length) {{
    document.getElementById("drawer-tags").innerHTML =
      tags.map(t => `<span class="tag">#${{t}}</span>`).join("");
    tagsSection.style.display = "flex";
  }} else {{
    tagsSection.style.display = "none";
  }}

  document.getElementById("drawer").classList.add("open");
}});

Graph.onBackgroundClick(closeDrawer);

function closeDrawer() {{
  document.getElementById("drawer").classList.remove("open");
}}

applyFilters();
</script>
</body>
</html>"""


def export_graph(
    output_path: str,
    folder: str = "",
    title: str = "Silica Knowledge Graph",
) -> dict:
    """Build and write the graph HTML to output_path.

    Returns dict with keys: success, path, nodes, edges, communities, unresolved.
    """
    nodes, edges = build_graph_data(folder=folder)
    communities = detect_communities(nodes, edges)
    lib_js = _fetch_lib_js()
    html_out = render_html(nodes, edges, communities, title=title, lib_js=lib_js)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_out, encoding="utf-8")

    n_notes       = sum(1 for n in nodes if n.get("type") != "ghost")
    n_ghost       = sum(1 for n in nodes if n.get("type") == "ghost")
    n_extracted   = sum(1 for e in edges if e.get("type") == "EXTRACTED")
    n_communities = len(communities)

    logger.info(
        "graph_export: wrote %s — %d notes, %d links, %d clusters, %d unresolved",
        out, n_notes, n_extracted, n_communities, n_ghost,
    )
    return {
        "success":     True,
        "path":        str(out.resolve()),
        "nodes":       n_notes,
        "edges":       n_extracted,
        "communities": n_communities,
        "unresolved":  n_ghost,
    }
