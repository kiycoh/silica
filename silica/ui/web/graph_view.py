# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

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


def _vendored_font_face() -> str:
    """@font-face rule with the Lexend woff2 inlined as a data: URI, so the
    exported HTML stays fully self-contained (it is opened from file:// too).
    Cosmetic asset: if missing, degrade to the system-ui fallback, not a raise."""
    import base64

    res = importlib.resources.files("silica.ui.web") / "static" / "lexend-latin.woff2"
    if not res.is_file():
        return ""
    b64 = base64.b64encode(res.read_bytes()).decode("ascii")
    return (
        '@font-face{font-family:"Lexend";'
        f'src:url("data:font/woff2;base64,{b64}") format("woff2");'
        "font-weight:100 900;font-style:normal;font-display:swap}"
    )


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
    discourse: str = "",
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
    n_gaps       = sum(1 for e in edges if e.get("type") == "GAP")
    n_communities = len(communities)
    discourse_badge = (
        f'<div style="font-size:11px;color:#8f8f8f;letter-spacing:.04em;margin-bottom:6px">'
        f'discourse: <span style="color:#c9a227;font-weight:600">{html.escape(discourse)}</span></div>'
        if discourse else ""
    )

    legend_items = "".join(
        f'<div class="legend-item" data-community="{c.id}" data-size="{c.size}" onclick="filterCommunity({c.id})">'
        f'<span class="dot" style="background:{c.color}"></span>{html.escape(c.label)} '
        f'<span style="color:#5a6372;font-size:11px;margin-left:auto">{c.size}</span>'
        f'</div>\n'
        for c in communities
    )

    comm_labels_json = json.dumps(
        {c.id: c.label for c in communities}, ensure_ascii=False
    ).replace("</", "<\\/")

    tree_html = render_tree(nodes)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  {f'<script>{lib_js}</script>' if lib_js else '<script src="' + _VIS_JS_URL + '"></script>'}
  <style>
    /* Silicon-carbide iridescence: blue-black crystal substrate, cool-white
       foreground, one azure accent (exact pixel samples from the SiC macro).
       Gold flags caution. Community hues stay data-driven. Type: Lexend,
       matching the app shell (inlined data: URI — this file must stay
       self-contained for file:// use). */
    {_vendored_font_face()}
    :root{{
      --void:#0A0D14;--slate:#0F131C;--slate-2:#161B27;
      --line:#232A3A;--line-2:#38425A;
      --frost:#E8ECF5;--ash:#8B95AC;--ash-dim:#566076;
      --accent:#00A5E1;--gold:#C69700;
      --grad:linear-gradient(115deg,#00CEEA 0%,#0068B8 42%,#9D00B6 100%);
      --sans:"Lexend",system-ui,sans-serif;
    }}
    *{{box-sizing:border-box;margin:0;padding:0;border-radius:0}}
    html{{scrollbar-width:thin;scrollbar-color:var(--line-2) transparent}}
    ::-webkit-scrollbar{{width:8px;height:8px}}
    ::-webkit-scrollbar-track{{background:transparent}}
    ::-webkit-scrollbar-thumb{{background:var(--line-2);border:2px solid var(--void)}}
    ::-webkit-scrollbar-thumb:hover{{background:var(--ash-dim)}}
    body{{display:flex;height:100vh;font-family:var(--sans);font-weight:400;
          background:var(--void);color:var(--frost);overflow:hidden;-webkit-font-smoothing:antialiased}}
    /* CRT scanlines — decorative only, clicks pass through */
    body::after{{content:"";position:fixed;inset:0;z-index:99;pointer-events:none;
      background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,.07) 3px,rgba(0,0,0,.07) 4px)}}
    #sidebar{{width:240px;flex-shrink:0;background:var(--slate);border-right:1px solid var(--line);
              display:flex;flex-direction:column;padding:14px 12px;gap:14px;overflow-y:auto}}
    #sidebar h1{{font-size:.82rem;font-weight:800;letter-spacing:.28em;text-transform:uppercase;background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent}}
    body.embedded #sidebar{{display:none}}
    .stat-grid{{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);border:1px solid var(--line)}}
    .stat{{background:var(--slate-2);padding:9px;text-align:center}}
    .stat .val{{font-size:20px;font-weight:700;color:var(--frost)}}
    .stat .val.warn{{color:var(--gold)}}
    .stat .lbl{{font-size:10px;color:var(--ash-dim);margin-top:2px;letter-spacing:.08em;text-transform:uppercase}}
    #search{{width:100%;padding:8px 10px;background:var(--slate-2);border:1px solid var(--line-2);
             color:var(--frost);font-family:var(--sans);font-size:13px;outline:none}}
    #search:focus{{border-color:var(--frost)}}
    .section-title{{font-size:10px;color:var(--ash-dim);text-transform:uppercase;letter-spacing:.18em}}
    .section-title::before{{content:"// ";color:var(--accent)}}
    .filter-row{{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--ash);cursor:pointer;
                 padding:3px 0;user-select:none}}
    .filter-row input{{cursor:pointer;accent-color:var(--accent)}}
    .dot-edge{{width:24px;height:3px;flex-shrink:0}}
    #sort-communities:hover{{color:var(--frost)}}
    #legend-box{{display:flex;flex-direction:column;gap:2px;max-height:200px;overflow-y:auto}}
    .legend-item{{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--ash);cursor:pointer;
                  padding:3px 6px}}
    .legend-item:hover{{background:var(--slate-2);color:var(--frost)}}
    .legend-item.active{{background:var(--slate-2);outline:1px solid var(--frost);color:var(--frost)}}
    .dot{{width:9px;height:9px;flex-shrink:0}}
    .btn{{padding:8px 10px;background:var(--slate-2);border:1px solid var(--line-2);
           color:var(--ash);font-family:var(--sans);font-size:12px;cursor:pointer;text-align:center;
           text-transform:uppercase;letter-spacing:.06em}}
    .btn:hover{{border-color:var(--frost);color:var(--frost)}}
    /* min-width:0 + overflow:hidden: the WebGL canvas must never force the flex
       item wider than the viewport (it pushes the absolute HUD off-screen) */
    #graph-wrap{{flex:1;min-width:0;position:relative;overflow:hidden}}
    #graph{{width:100%;height:100%}}
    /* HUD — floating legend/filter panel anchored to the graph itself */
    #hud{{position:absolute;top:10px;right:10px;z-index:5;width:216px;max-height:calc(100% - 20px);
          display:flex;flex-direction:column;gap:12px;padding:12px;overflow-y:auto;
          background:rgba(10,13,20,.92);border:1px solid var(--line-2)}}
    #drawer{{width:260px;flex-shrink:0;background:var(--slate);border-left:1px solid var(--line);
             padding:18px 16px;overflow-y:auto;display:none;flex-direction:column;gap:12px}}
    #drawer.open{{display:flex}}
    #drawer-title{{font-size:15px;font-weight:600;color:var(--frost);word-break:break-word}}
    #drawer-path{{font-size:11px;color:var(--ash-dim);word-break:break-all}}
    #drawer-meta{{font-size:12px;color:var(--ash)}}
    .drawer-section{{display:flex;flex-direction:column;gap:4px}}
    .drawer-label{{font-size:10px;color:var(--ash-dim);text-transform:uppercase;letter-spacing:.18em}}
    .drawer-val{{font-size:13px;color:var(--frost)}}
    .tag{{display:inline-block;padding:2px 7px;background:var(--slate-2);border:1px solid var(--line);
           font-size:11px;color:var(--ash);margin:2px}}
    #close-drawer{{align-self:flex-end;cursor:pointer;color:var(--ash-dim);font-size:18px;line-height:1}}
    #close-drawer:hover{{color:var(--frost)}}
    #search-results{{display:none;flex-direction:column;gap:1px;max-height:260px;overflow-y:auto;
                     margin-top:6px;border:1px solid var(--line);background:var(--slate-2)}}
    #search-results.open{{display:flex}}
    #search-count{{font-size:10px;color:var(--ash-dim);letter-spacing:.04em;padding:6px 8px 2px}}
    .result-item{{display:flex;flex-direction:column;gap:1px;padding:6px 8px;cursor:pointer;border-left:2px solid transparent}}
    .result-item:hover,.result-item.sel{{background:var(--slate);border-left-color:var(--accent)}}
    .result-name{{font-size:12px;color:var(--frost);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
    .result-sub{{font-size:10px;color:var(--ash-dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
    .result-sub em{{color:var(--frost);font-style:normal}}
    #file-tree{{display:flex;flex-direction:column;max-height:260px;overflow-y:auto;font-size:12px}}
    #file-tree summary{{cursor:pointer;color:var(--ash);padding:2px 0;user-select:none;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
    #file-tree summary:hover{{color:var(--frost)}}
    #file-tree details details,#file-tree .tree-note{{margin-left:12px}}
    .tree-note{{color:var(--ash);cursor:pointer;padding:2px 6px;border-left:2px solid transparent;
               white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
    .tree-note:hover{{background:var(--slate-2);border-left-color:var(--accent);color:var(--frost)}}
    .force-row{{display:flex;justify-content:space-between;align-items:center;font-size:12px;
                color:var(--ash);margin-top:6px}}
    .force-row .fv{{color:var(--ash-dim);font-size:11px}}
    .force-slider{{width:100%;accent-color:var(--accent);cursor:pointer;margin-top:2px}}
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
    <div class="section-title" style="margin-bottom:6px">Files</div>
    {tree_html}
  </div>
</div>

<div id="graph-wrap">
  <div id="graph"></div>
  <div id="hud">
    <div>
      <div class="section-title" style="margin-bottom:8px">Edge types</div>
      <label class="filter-row">
        <input type="checkbox" id="cb-extracted" checked onchange="updateEdgeFilter()">
        <div class="dot-edge" style="background:#8f8f8f"></div>
        Resolved
        <span style="color:#5c5c5c;font-size:11px;margin-left:auto">{n_extracted}</span>
      </label>
      <label class="filter-row" style="margin-top:4px">
        <input type="checkbox" id="cb-ambiguous" onchange="updateEdgeFilter()">
        <div class="dot-edge" style="background:#ff2a2a"></div>
        Unresolved
        <span style="color:#5c5c5c;font-size:11px;margin-left:auto">{n_ambiguous}</span>
      </label>
      <label class="filter-row" style="margin-top:4px" title="Well-formed areas with no links between them — a bridge could go here">
        <input type="checkbox" id="cb-gaps" checked onchange="updateEdgeFilter()">
        <div class="dot-edge" style="background:#c9a227"></div>
        Structural gaps
        <span style="color:#5c5c5c;font-size:11px;margin-left:auto">{n_gaps}</span>
      </label>
    </div>

    <div>
      <div class="section-title" style="margin-bottom:6px;display:flex;align-items:center;justify-content:space-between">
        Communities
        <span id="sort-communities" style="color:#8f8f8f;cursor:pointer;font-size:11px;letter-spacing:0;text-transform:none"
              onclick="toggleCommunitySort()" title="sort by size">size &#8597;</span>
      </div>
      {discourse_badge}
      <div id="legend-box">
{legend_items}      <div class="legend-item active" id="legend-all" onclick="filterCommunity(-2)">
          <span class="dot" style="background:#5c5c5c"></span>Show all
        </div>
      </div>
    </div>

    <div>
      <div class="section-title" style="display:flex;align-items:center;justify-content:space-between">
        Forces
        <span style="color:#8f8f8f;cursor:pointer;font-size:11px;letter-spacing:0;text-transform:none"
              onclick="resetForces()" title="back to auto-scaled defaults">reset</span>
      </div>
      <div class="force-row">Repel<span class="fv" id="fv-repel">1.0&times;</span></div>
      <input type="range" class="force-slider" id="sl-repel" min="-0.7" max="0.7" step="0.01" value="0" oninput="onForceSlider()">
      <div class="force-row">Link distance<span class="fv" id="fv-dist">1.0&times;</span></div>
      <input type="range" class="force-slider" id="sl-dist" min="-0.7" max="0.7" step="0.01" value="0" oninput="onForceSlider()">
      <div class="force-row">Center<span class="fv" id="fv-center">1.00</span></div>
      <input type="range" class="force-slider" id="sl-center" min="0" max="1" step="0.05" value="1" oninput="onForceSlider()">
    </div>

    <div style="display:flex;gap:6px">
      <div class="btn" style="flex:1" onclick="Graph.zoomToFit(400)">&#8862; Fit graph</div>
      <div class="btn" title="rebuild from the vault (e.g. after editing notes outside silica)"
           onclick="location.reload()">&#8635;</div>
    </div>
  </div>
</div>

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
// Embedded in the web app's iframe: the app's own sidebar (stats/search/tree)
// replaces the internal one; only the graph + HUD legend remain.
if (window.parent !== window) document.body.classList.add("embedded");

const RAW_NODES = {nodes_json};
const RAW_EDGES = {edges_json};
const COMM_LABELS = {comm_labels_json};

const outDeg = {{}}, inDeg = {{}};
RAW_EDGES.forEach(e => {{
  outDeg[e.from] = (outDeg[e.from] || 0) + 1;
  inDeg[e.to]   = (inDeg[e.to]   || 0) + 1;
}});

const NODE_BY_ID = {{}};
RAW_NODES.forEach(n => {{ NODE_BY_ID[n.id] = n; }});

const neighbors = {{}};
RAW_EDGES.forEach(e => {{
  (neighbors[e.from] = neighbors[e.from] || new Set()).add(e.to);
  (neighbors[e.to]   = neighbors[e.to]   || new Set()).add(e.from);
}});

let focusId = null;

// Highlight a node and its 1-hop neighbours; dim everything else. Refresh via
// the accessor re-pass idiom (same trick applyFilters uses for visibility) so
// the physics layout is untouched.
function applyFocus(id) {{
  focusId = id;
  const nb = neighbors[id] || new Set();
  RAW_NODES.forEach(n => {{ n._dim = id != null && n.id !== id && !nb.has(n.id); }});
  RAW_EDGES.forEach(e => {{ e._dim = id != null && e.from !== id && e.to !== id; }});
  Graph.nodeColor(Graph.nodeColor());
  Graph.linkColor(Graph.linkColor());
}}

function clearFocus() {{
  focusId = null;
  RAW_NODES.forEach(n => {{ n._dim = false; }});
  RAW_EDGES.forEach(e => {{ e._dim = false; }});
  Graph.nodeColor(Graph.nodeColor());
  Graph.linkColor(Graph.linkColor());
  Graph.zoomToFit(600);
}}

let activeCommunity = -2;
let showExtracted = true;
let showAmbiguous = false;
let showGaps = true;

// --- Node color = its community color, flat -------------------------------
// One hue per community: every node in a community shares the exact color,
// hub or leaf. Degree is shown by size, never by washing the hue out.
function nodeColor(n) {{
  // ponytail: solid darken-to-background dim; switch to rgba() only if visual
  // verification shows 3d-force-graph honours per-node alpha.
  if (n._dim) return '#1c1c1c';
  if (n.type === 'ghost') return '#4a4a4a';   // muted gray — dimmed, never black
  if (n.type === 'concept') return '#8f7fa3'; // concepts view: neutral violet, no community
  return (n.color && n.color.background) || '#566076';
}}

// --- Density-aware forces ---------------------------------------------------
// The lib's d3 defaults (charge -60 in 3D, link distance 30) collapse dense
// graphs into a hairball: equilibrium spacing must grow with avg degree or
// neighborhoods overlap. sqrt keeps sparse graphs (k<=2) exactly as before
// (scale=1) and opens dense ones up to 4x. Sliders multiply on top of this
// baseline, so the auto-scaling stays authoritative as the vault grows.
const AVG_DEG = RAW_NODES.length ? 2 * RAW_EDGES.length / RAW_NODES.length : 0;
const FORCE_SCALE = Math.min(4, Math.max(1, Math.sqrt(AVG_DEG / 2)));
const BASE_CHARGE = -60 * FORCE_SCALE * FORCE_SCALE;
const BASE_DIST = 30 * FORCE_SCALE;
// Fixed 100 ticks never let a big graph unfold; scale settle time with size.
const COOLDOWN_TICKS = 100 + Math.min(200, Math.round(RAW_NODES.length / 10));

const Graph = new ForceGraph3D(document.getElementById("graph"))
  .backgroundColor("#0A0D14")
  .graphData({{ nodes: RAW_NODES, links: RAW_EDGES }})
  .linkSource("from").linkTarget("to")
  .nodeLabel("label").nodeVal("size")
  .nodeColor(nodeColor)
  .linkColor(l => l._dim ? '#141414' : ((l.color && l.color.color) || "#8f8f8f"))
  // Perf on big vaults (1200+ notes): linkWidth>0 makes every edge a cylinder
  // mesh and arrows add a cone per edge — thousands of meshes. Width 0 ⇒ cheap
  // GL lines; no arrows; fewer sphere segments; finite cooldown so the sim
  // settles and stops reflowing instead of re-laying-out every frame.
  .linkWidth(0)
  // Structural gaps have no dash in WebGL — mark them by motion instead: amber
  // particles stream along the absent bridge. Only for GAP links, and they stop
  // when the link is dimmed (node focus) so focus mode stays quiet.
  .linkDirectionalParticles(l => l.type === "GAP" && !l._dim ? 2 : 0)
  .linkDirectionalParticleColor(() => "{_EDGE_COLOR_GAP}")
  .linkDirectionalParticleWidth(2)
  .nodeResolution(6)
  .cooldownTicks(COOLDOWN_TICKS)
  .nodeVisibility(n => !n._hidden)
  .linkVisibility(l => !l._hidden);

// Slider multipliers persist per view — links and concepts have different
// densities, so a manual correction for one shouldn't leak into the other.
// The baseline is never persisted: it is recomputed from the current graph.
const FORCES_KEY = "silica-graph-forces-" +
  (RAW_NODES.some(n => n.type === "concept") ? "concepts" : "links");
let forceMul = {{ repel: 1, dist: 1, center: 1 }};
try {{
  Object.assign(forceMul, JSON.parse(localStorage.getItem(FORCES_KEY)) || {{}});
}} catch (e) {{ /* corrupt or blocked storage -> auto defaults */ }}

function applyForces(reheat) {{
  // distanceMax bounds both over-dispersion and per-tick cost on big graphs.
  Graph.d3Force("charge").strength(BASE_CHARGE * forceMul.repel)
    .distanceMax(600 * FORCE_SCALE);
  Graph.d3Force("link").distance(BASE_DIST * forceMul.dist);
  // Center capped at 1: d3 forceCenter shifts positions directly, >1 oscillates.
  Graph.d3Force("center").strength(Math.min(1, forceMul.center));
  if (reheat) Graph.d3ReheatSimulation();
}}

// Log-scale track for the multiplier sliders: x1 sits mid-track and the
// useful 0.2-1 range gets half the travel instead of a sliver.
const fromSlider = v => Math.pow(10, +v);
const toSlider = m => Math.log10(m);

function syncForceUI() {{
  document.getElementById("sl-repel").value = toSlider(forceMul.repel);
  document.getElementById("sl-dist").value = toSlider(forceMul.dist);
  document.getElementById("sl-center").value = forceMul.center;
  document.getElementById("fv-repel").textContent = forceMul.repel.toFixed(1) + "\\u00d7";
  document.getElementById("fv-dist").textContent = forceMul.dist.toFixed(1) + "\\u00d7";
  document.getElementById("fv-center").textContent = (+forceMul.center).toFixed(2);
}}

function onForceSlider() {{
  forceMul.repel = fromSlider(document.getElementById("sl-repel").value);
  forceMul.dist = fromSlider(document.getElementById("sl-dist").value);
  forceMul.center = +document.getElementById("sl-center").value;
  try {{ localStorage.setItem(FORCES_KEY, JSON.stringify(forceMul)); }} catch (e) {{}}
  syncForceUI();
  applyForces(true);
}}

function resetForces() {{
  forceMul = {{ repel: 1, dist: 1, center: 1 }};
  try {{ localStorage.removeItem(FORCES_KEY); }} catch (e) {{}}
  syncForceUI();
  applyForces(true);
}}

syncForceUI();
applyForces(false); // sim just started at full alpha, no reheat needed

function applyFilters() {{
  RAW_NODES.forEach(n => {{
    n._hidden = (activeCommunity !== -2 && n.group !== activeCommunity);
  }});
  RAW_EDGES.forEach(e => {{
    e._hidden = (e.type === "EXTRACTED" && !showExtracted) ||
                (e.type === "AMBIGUOUS" && !showAmbiguous) ||
                (e.type === "GAP" && !showGaps);
  }});
  // Re-pass the current accessor to force a visibility refresh without resetting the physics layout
  Graph.nodeVisibility(Graph.nodeVisibility());
  Graph.linkVisibility(Graph.linkVisibility());
}}

function updateEdgeFilter() {{
  showExtracted = document.getElementById("cb-extracted").checked;
  showAmbiguous = document.getElementById("cb-ambiguous").checked;
  showGaps = document.getElementById("cb-gaps").checked;
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
  if (cid !== -2) Graph.zoomToFit(400, 50, n => n.group === cid); // isolate: fit camera to the filtered set
}}

// --- Communities legend: sort by size, toggling ascending <-> descending ----
let communitySortAsc = true;
function toggleCommunitySort() {{
  const box = document.getElementById("legend-box");
  const allItem = document.getElementById("legend-all");
  const items = Array.from(box.querySelectorAll(".legend-item[data-community]"));
  items.sort((a, b) => (+a.dataset.size - +b.dataset.size) * (communitySortAsc ? 1 : -1));
  items.forEach(el => box.insertBefore(el, allItem));
  document.getElementById("sort-communities").textContent = communitySortAsc ? "size ↑" : "size ↓";
  communitySortAsc = !communitySortAsc;
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

// Shared selection path for tree clicks and search results: open the note view
// and fly the camera. Task 3 adds neighbour dimming here.
function chooseNode(node) {{
  if (!node) return;
  selectNode(node);
  focusNode(node);
  applyFocus(node.id);
}}

function chooseResult(i) {{
  const n = results[i];
  if (!n) return;
  selIdx = i;
  chooseNode(n);
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
  const k = 1 + 90 * 3 / r;
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
  const betwText = node.betweenness ? ` · betweenness ${{node.betweenness}}` : "";
  document.getElementById("drawer-meta").textContent = `${{node.type}}${{commText}}${{betwText}}`;
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

// Direct clicks in the 3D view get the same dim-non-neighbours treatment as
// tree/search picks, but skip focusNode's camera fly — the user is already
// looking at this spot, recentring would just be jarring.
Graph.onNodeClick(node => {{ selectNode(node); applyFocus(node.id); }});
Graph.onBackgroundClick(() => {{ closeDrawer(); clearFocus(); }});

// The embedding page (chat + note-panel) tells us which note is open
// elsewhere — e.g. a link followed inside the note panel itself — so the
// graph mirrors it. Dim only, no camera move (same reasoning as above).
window.addEventListener("message", e => {{
  if (e.data && e.data.type === "silica-focus-path") {{
    applyFocus(NODE_BY_ID[e.data.path] ? e.data.path : null);
  }}
}});

function closeDrawer() {{
  document.getElementById("drawer").classList.remove("open");
}}

document.getElementById("file-tree").addEventListener("click", e => {{
  const leaf = e.target.closest(".tree-note");
  if (leaf) chooseNode(NODE_BY_ID[leaf.dataset.id]);
}});

applyFilters();
</script>
</body>
</html>"""


_EDGE_COLOR_GAP = "#c9a227"  # dim amber — "a bridge could go here, and doesn't"


def _gap_edges(nodes: list[dict], edges: list[dict], top_k: int = 5) -> list[dict]:
    """Top structural gaps as overlay edges between two area hubs.

    Reads: 'these two well-formed areas should probably connect, and don't.'
    Reuses graph_export.structural_gaps so the overlay agrees with the /graph
    report's Structural Gaps section node-for-node. Only the keys 3d-force-graph
    actually honours: from/to (linkSource/linkTarget), color.color (linkColor),
    and type (visibility toggle + particle accessor). The lib draws these as
    amber directional-particle links — WebGL has no dashed line, so motion, not
    a dash pattern, is what sets a gap apart. score rides along for the title map.
    """
    from silica.kernel.graph_export import structural_gaps

    return [
        {
            "id":    f"gap{i}",
            "from":  hub_a,
            "to":    hub_b,
            "type":  "GAP",
            "color": {"color": _EDGE_COLOR_GAP},
            "score": score,
        }
        for i, (ca, cb, hub_a, hub_b, ie, score) in enumerate(
            structural_gaps(nodes, edges, top_k=top_k)
        )
    ]


def export_graph(
    output_path: str,
    folder: str = "",
    title: str = "Vault Graph",
    mode: str = "links",
) -> dict:
    """Build and write the graph HTML to output_path.

    Reads the vendored JS first (fail fast on a packaging bug) and always inlines
    it, so the emitted file is self-contained/offline. Returns dict with keys:
    success, path, nodes, edges, communities, unresolved.

    mode="concepts" (F4): merges the note->Concept-set bipartite expansion
    (kernel.graph_export.build_bipartite_data) into the dataset — the on-disk
    incidence IS a hypergraph (a note is a hyperedge over its concepts).
    Structural-gap particles stay off in this mode (different question).
    """
    from silica.kernel.graph_export import build_graph_data, detect_communities

    from silica.kernel.graph_export import canvas_metrics, discourse_shape

    lib_js = _vendored_lib_js()  # fail fast before the graph build
    nodes, edges = build_graph_data(folder=folder)
    communities = detect_communities(nodes, edges)

    n_concepts = 0
    if mode == "concepts":
        from silica.kernel.graph_export import bipartite_for_active_vault

        cnodes, cedges = bipartite_for_active_vault(nodes)
        n_concepts = len(cnodes)
        nodes = nodes + cnodes
        edges = edges + cedges

    # Betweenness → node size (bottleneck nodes swell) + discourse-shape badge,
    # from one shared nx build. Base size 16 stays for ordinary nodes.
    bet, giant = canvas_metrics(nodes, edges)
    if bet:
        for n in nodes:
            if n.get("type") not in ("ghost", "concept"):
                b = round(bet.get(n["id"], 0.0), 4)
                n["betweenness"] = b
                n["size"] = round(16 + 40 * b, 2)
    discourse = discourse_shape(
        sum(1 for n in nodes if n.get("type") not in ("ghost", "concept")),
        giant, [c.size for c in communities],
    )

    # Gap particles answer a linking question; the concepts view asks an
    # incidence one — keep them apart.
    gaps = [] if mode == "concepts" else _gap_edges(nodes, edges)
    html_out = render_html(
        nodes, edges + gaps, communities, title=title, lib_js=lib_js, discourse=discourse
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_out, encoding="utf-8")

    n_notes       = sum(1 for n in nodes if n.get("type") not in ("ghost", "concept"))
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
        "gaps":        len(gaps),
        "concepts":    n_concepts,
    }
