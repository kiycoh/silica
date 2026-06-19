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

import colorsys
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

_EDGE_COLOR_EXTRACTED = "#22d3ee"   # cyan — resolved links
_EDGE_COLOR_AMBIGUOUS = "#6366f1"   # indigo — unresolved (web/ uses indigo for ambiguous)
_NODE_DEFAULT_COLOR = {"background": "#4d5575", "border": "#22d3ee",
                       "highlight": {"background": "#5a6372", "border": "#e7ebf1"}}
_NODE_GHOST_COLOR   = {"background": "#151a23", "border": "#6366f1",
                       "highlight": {"background": "#1e2530", "border": "#8a93a3"}}


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
            color=_community_color(i),
            size=len(comm),
        )
        for i, comm in enumerate(communities)
    ]


def render_html(
    nodes: list[dict],
    edges: list[dict],
    communities: "list[Community]" = (),  # type: ignore[assignment]
    title: str = "Vault Graph",
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
        f'<span style="color:#5a6372;font-size:11px;margin-left:auto">{c.size}</span>'
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
    :root{{
      --void:#0B0D12;--slate:#10141B;--slate-2:#151A23;
      --line:#1E2530;--line-2:#2B3442;
      --frost:#E7EBF1;--ash:#8A93A3;--ash-dim:#5A6372;
      --cyan:#22D3EE;--indigo:#6366F1;--edge:#4D5575;
      --grad:linear-gradient(100deg,var(--cyan),var(--indigo));
      --mono:ui-monospace,"Cascadia Code","SF Mono",Menlo,Consolas,"DejaVu Sans Mono",monospace;
    }}
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{display:flex;height:100vh;font-family:var(--mono);font-weight:300;letter-spacing:-.01em;
          background:var(--void);color:var(--frost);overflow:hidden;-webkit-font-smoothing:antialiased}}
    #sidebar{{width:240px;flex-shrink:0;background:var(--slate);border-right:1px solid var(--line);
              display:flex;flex-direction:column;padding:16px 14px;gap:16px;overflow-y:auto;
              background-image:radial-gradient(circle at 1px 1px,rgba(34,211,238,.05) 1px,transparent 0);
              background-size:34px 34px}}
    #sidebar h1{{font-size:.82rem;font-weight:700;letter-spacing:.28em;text-transform:uppercase;
                 background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent}}
    .stat-grid{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}
    .stat{{background:var(--slate-2);border:1px solid var(--line);border-radius:3px;padding:9px;text-align:center}}
    .stat .val{{font-size:20px;font-weight:700;color:var(--cyan)}}
    .stat .lbl{{font-size:10px;color:var(--ash-dim);margin-top:2px;letter-spacing:.04em}}
    #search{{width:100%;padding:8px 10px;background:var(--slate-2);border:1px solid var(--line-2);
             border-radius:3px;color:var(--frost);font-family:var(--mono);font-size:13px;outline:none}}
    #search:focus{{border-color:var(--cyan)}}
    .section-title{{font-size:10px;color:var(--ash-dim);text-transform:uppercase;letter-spacing:.18em}}
    .filter-row{{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--ash);cursor:pointer;
                 padding:3px 0;user-select:none}}
    .filter-row input{{cursor:pointer;accent-color:var(--cyan)}}
    .dot-edge{{width:24px;height:3px;border-radius:2px;flex-shrink:0}}
    #legend-box{{display:flex;flex-direction:column;gap:2px;max-height:200px;overflow-y:auto}}
    .legend-item{{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--ash);cursor:pointer;
                  padding:3px 6px;border-radius:3px}}
    .legend-item:hover{{background:var(--slate-2);color:var(--frost)}}
    .legend-item.active{{background:var(--slate-2);outline:1px solid var(--cyan);color:var(--frost)}}
    .dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
    .btn{{padding:8px 10px;background:var(--slate-2);border:1px solid var(--line-2);border-radius:3px;
           color:var(--ash);font-family:var(--mono);font-size:12px;cursor:pointer;text-align:center}}
    .btn:hover{{border-color:var(--cyan);color:var(--cyan)}}
    #graph-wrap{{flex:1;position:relative}}
    #graph{{width:100%;height:100%}}
    #drawer{{width:260px;flex-shrink:0;background:var(--slate);border-left:1px solid var(--line);
             padding:18px 16px;overflow-y:auto;display:none;flex-direction:column;gap:12px}}
    #drawer.open{{display:flex}}
    #drawer-title{{font-size:15px;font-weight:600;color:var(--frost);word-break:break-word}}
    #drawer-path{{font-size:11px;color:var(--ash-dim);word-break:break-all}}
    #drawer-meta{{font-size:12px;color:var(--cyan)}}
    .drawer-section{{display:flex;flex-direction:column;gap:4px}}
    .drawer-label{{font-size:10px;color:var(--ash-dim);text-transform:uppercase;letter-spacing:.18em}}
    .drawer-val{{font-size:13px;color:var(--frost)}}
    .tag{{display:inline-block;padding:2px 7px;background:var(--slate-2);border:1px solid var(--line);
           border-radius:10px;font-size:11px;color:var(--cyan);margin:2px}}
    #close-drawer{{align-self:flex-end;cursor:pointer;color:var(--ash-dim);font-size:18px;line-height:1}}
    #close-drawer:hover{{color:var(--frost)}}
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
      <div class="dot-edge" style="background:#22d3ee"></div>
      Resolved
      <span style="color:#5a6372;font-size:11px;margin-left:auto">{n_extracted}</span>
    </label>
    <label class="filter-row" style="margin-top:4px">
      <input type="checkbox" id="cb-ambiguous" onchange="updateEdgeFilter()">
      <div class="dot-edge" style="background:#6366f1"></div>
      Unresolved
      <span style="color:#5a6372;font-size:11px;margin-left:auto">{n_ambiguous}</span>
    </label>
  </div>

  <div>
    <div class="section-title" style="margin-bottom:6px">Communities</div>
    <div id="legend-box">
{legend_items}      <div class="legend-item active" id="legend-all" onclick="filterCommunity(-2)">
        <span class="dot" style="background:#4d5575"></span>Show all
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

// --- Node color = its community color, flat -------------------------------
// One hue per community: every node in a community shares the exact color,
// hub or leaf. Degree is shown by size, never by washing the hue out.
function nodeColor(n) {{
  if (n.type === 'ghost') return '#4a5468';   // muted slate — dimmed, never black
  return (n.color && n.color.background) || '#5a6372';
}}

const Graph = new ForceGraph3D(document.getElementById("graph"))
  .backgroundColor("#0B0D12")
  .graphData({{ nodes: RAW_NODES, links: RAW_EDGES }})
  .linkSource("from").linkTarget("to")
  .nodeLabel("label").nodeVal("size")
  .nodeColor(nodeColor)
  .linkColor(l => (l.color && l.color.color) || "#22d3ee")
  // Perf on big vaults (1200+ notes): linkWidth>0 makes every edge a cylinder
  // mesh and arrows add a cone per edge — thousands of meshes. Width 0 ⇒ cheap
  // GL lines; no arrows; fewer sphere segments; finite cooldown so the sim
  // settles and stops reflowing instead of re-laying-out every frame.
  .linkWidth(0)
  .nodeResolution(6)
  .cooldownTicks(100)
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
    title: str = "Vault Graph",
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
