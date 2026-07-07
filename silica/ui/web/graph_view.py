"""Graph viewer — the 3d-force-graph HTML emitter for the vault wikilink graph.

Split out of `silica.kernel.graph_export` (which keeps the deterministic *data*
role: build_graph_data / detect_communities). This module owns only the viewer:
it turns nodes/edges/communities into a fully self-contained HTML file.

The JS bundle is *vendored* (silica/ui/web/static/3d-force-graph.min.js, pinned
to v1.80.0) and inlined into every emitted file — the artifact opens offline,
with no network at render time. `render_html` keeps an empty-`lib_js` CDN
fallback for direct/test callers, but `export_graph` (the production path) always
inlines the vendored bundle and raises loudly if the asset is missing.
"""
from __future__ import annotations

import html
import importlib.resources
import json
import logging
from pathlib import Path

from silica.kernel.graph_export import Community

logger = logging.getLogger(__name__)

_VIS_JS_URL = "https://cdn.jsdelivr.net/npm/3d-force-graph@1.80.0/dist/3d-force-graph.min.js"


def _vendored_lib_js() -> str:
    """Read the vendored 3d-force-graph bundle shipped under ui/web/static/.

    Raises a clear RuntimeError if the asset is absent (a packaging bug). We do
    NOT fall back to render_html's empty-lib_js CDN <script src>: that would
    silently reintroduce the network dependency this split removed and hide the
    bug. Keep the trust-boundary failure loud.
    """
    res = importlib.resources.files("silica.ui.web") / "static" / "3d-force-graph.min.js"
    if not res.is_file():
        raise RuntimeError(
            "graph_export: vendored 3d-force-graph.min.js is missing from "
            "silica/ui/web/static/ — packaging bug. Reinstall silica or re-vendor "
            "the asset (pinned v1.80.0)."
        )
    return res.read_text(encoding="utf-8")


def render_tree(nodes: list[dict]) -> str:
    """Build a collapsible <details> file tree from real note paths.

    Pure: nodes -> HTML. Folders become nested <details>/<summary> (native
    collapse, no JS); notes become <div class="tree-note" data-id=ID>NAME</div>.
    Ghost nodes (type == "ghost" or empty path) are unresolved links, not files,
    so they are skipped. Folders sort before notes at each level; both groups
    sort case-insensitively.
    """
    root: dict = {}
    for n in nodes:
        if n.get("type") == "ghost":
            continue
        path = n.get("path") or ""
        if not path:
            continue
        *folders, leaf = path.split("/")
        cur = root
        for f in folders:
            cur = cur.setdefault(f, {})
        cur.setdefault("__notes__", []).append((leaf, n.get("id", path)))

    def emit(tree: dict, depth: int) -> str:
        out = []
        for name in sorted((k for k in tree if k != "__notes__"), key=str.lower):
            attr = " open" if depth == 0 else ""
            out.append(f"<details{attr}><summary>{html.escape(name)}</summary>")
            out.append(emit(tree[name], depth + 1))
            out.append("</details>")
        for leaf, nid in sorted(tree.get("__notes__", []), key=lambda x: x[0].lower()):
            out.append(
                f'<div class="tree-note" data-id="{html.escape(nid, quote=True)}">'
                f"{html.escape(leaf)}</div>"
            )
        return "".join(out)

    return f'<div id="file-tree">{emit(root, 0)}</div>'


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
    #search-results{{display:none;flex-direction:column;gap:1px;max-height:260px;overflow-y:auto;
                     margin-top:6px;border:1px solid var(--line);border-radius:3px;background:var(--slate-2)}}
    #search-results.open{{display:flex}}
    #search-count{{font-size:10px;color:var(--ash-dim);letter-spacing:.04em;padding:6px 8px 2px}}
    .result-item{{display:flex;flex-direction:column;gap:1px;padding:6px 8px;cursor:pointer;border-left:2px solid transparent}}
    .result-item:hover,.result-item.sel{{background:var(--slate);border-left-color:var(--cyan)}}
    .result-name{{font-size:12px;color:var(--frost);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
    .result-sub{{font-size:10px;color:var(--ash-dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
    .result-sub em{{color:var(--cyan);font-style:normal}}
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

  <input id="search" type="text" placeholder="Search notes, paths, #tags&#8230;"
         oninput="onSearch(this.value)" onkeydown="onSearchKey(event)" autocomplete="off">
  <div id="search-results"></div>

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
    n._hidden = (activeCommunity !== -2 && n.group !== activeCommunity);
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

// --- Search → ranked results → fly-to-focus -------------------------------
// Search by what people actually remember: title first, then path, then
// #tags, then the cluster they were browsing. Choosing a result flies the
// camera to the node and selects it — the graph answers "where is it", not
// just "is it somewhere in this cloud".
let results = [], selIdx = -1;

function scoreNode(n, q) {{
  if (n.type === 'ghost') return 0;
  const label = (n.label || '').toLowerCase();
  if (label === q)            return 5;
  if (label.startsWith(q))    return 4;
  if (label.includes(q))      return 3;
  if ((n.path || '').toLowerCase().includes(q)) return 2;
  if ((n.tags || []).some(t => t.toLowerCase().includes(q))) return 2;
  const cl = COMM_LABELS[n.group];
  if (cl && cl.toLowerCase().includes(q)) return 1;
  return 0;
}}

function renderResults(q) {{
  const box = document.getElementById("search-results");
  if (!q) {{ box.className = ""; box.innerHTML = ""; results = []; selIdx = -1; return; }}
  results = RAW_NODES
    .map(n => [scoreNode(n, q), n])
    .filter(p => p[0] > 0)
    .sort((a, b) => b[0] - a[0] || a[1].label.localeCompare(b[1].label))
    .slice(0, 12)
    .map(p => p[1]);
  selIdx = results.length ? 0 : -1;

  const esc = s => String(s).replace(/[&<>]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]));
  const sub = n => {{
    const cl = COMM_LABELS[n.group];
    return cl ? '<em>' + esc(cl) + '</em>' : esc(n.path || n.type);
  }};
  box.innerHTML =
    '<div id="search-count">' + (results.length || 'no') +
      ' result' + (results.length === 1 ? '' : 's') + '</div>' +
    results.map((n, i) =>
      '<div class="result-item' + (i === selIdx ? ' sel' : '') +
        '" onclick="chooseResult(' + i + ')">' +
        '<span class="result-name">' + esc(n.label) + '</span>' +
        '<span class="result-sub">' + sub(n) + '</span>' +
      '</div>').join("");
  box.className = "open";
}}

function chooseResult(i) {{
  const n = results[i];
  if (!n) return;
  selIdx = i;
  selectNode(n);
  focusNode(n);
}}

function moveSel(d) {{
  if (!results.length) return;
  selIdx = (selIdx + d + results.length) % results.length;
  document.querySelectorAll("#search-results .result-item")
    .forEach((el, i) => el.classList.toggle("sel", i === selIdx));
}}

function onSearch(q) {{ renderResults(q.trim().toLowerCase()); }}

function onSearchKey(e) {{
  if (e.key === "Enter")          {{ e.preventDefault(); chooseResult(selIdx); }}
  else if (e.key === "ArrowDown") {{ e.preventDefault(); moveSel(1); }}
  else if (e.key === "ArrowUp")   {{ e.preventDefault(); moveSel(-1); }}
  else if (e.key === "Escape")    {{ document.getElementById("search").value = ""; renderResults(""); }}
}}

// Fly the camera to a node along its outward radial, looking at it. Coords
// (node.x/y/z) exist once the layout has run (cooldownTicks); before that they
// default to 0 and the camera simply recentres — harmless.
function focusNode(node) {{
  const r = Math.hypot(node.x || 0, node.y || 0, node.z || 0) || 1;
  const k = 1 + 90 / r;
  Graph.cameraPosition(
    {{ x: (node.x || 0) * k, y: (node.y || 0) * k, z: (node.z || 0) * k }},
    node, 900
  );
}}

function selectNode(node) {{
  // Embedded in the web-UI iframe: hand off to the parent's note drawer instead
  // of opening this internal metadata drawer (avoids two stacked drawers).
  if (window.parent !== window) {{
    window.parent.postMessage({{ type: "silica-open-note", path: node.path }}, "*");
    return;
  }}
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
}}

Graph.onNodeClick(selectNode);
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

    Reads the vendored JS first (fail fast on a packaging bug) and always inlines
    it, so the emitted file is self-contained/offline. Returns dict with keys:
    success, path, nodes, edges, communities, unresolved.
    """
    from silica.kernel.graph_export import build_graph_data, detect_communities

    lib_js = _vendored_lib_js()  # fail fast before the graph build
    nodes, edges = build_graph_data(folder=folder)
    communities = detect_communities(nodes, edges)
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
