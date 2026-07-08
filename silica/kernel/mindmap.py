# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""L1 Mindmap — deterministic radial map rooted on one note (zero LLM).

Builds a `MapView` — a radial mind-map centred on a single note — from the vault's
wikilink graph plus the latent (embeddings + co-occurrence) relatedness leg. The
builder computes the 2D coordinates **server-side**, so the two surfaces that
materialise a MapView (an Obsidian `.canvas` file and the web GUI's static SVG)
show the *identical* map and cannot diverge.

Complementary to `graph_export` (which draws the flat whole-vault network): `/graph`
is the network, `/map <note>` is a rooted, radial tree.

Layout is deterministic (same input → same positions, no `random`, no physics) and
non-overlap is guaranteed *by construction*: every pair of node centres ends up at
euclidean distance ≥ hypot(W, H), which is a sufficient condition for two equal
axis-aligned boxes never to overlap.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

# Fixed node-box size in canvas units. The non-overlap guarantee is stated in
# terms of these: distance ≥ hypot(W, H) ⇒ the two boxes' AABBs are disjoint.
BOX_W = 260.0
BOX_H = 80.0
_DIAG = math.hypot(BOX_W, BOX_H)

# Muted slate for community-less nodes (group == -1) — never black/white.
_MUTED = "#5a6372"


@dataclass
class MapNode:
    id: str          # stable: vault-relative path WITH .md (e.g. "concetti/x.md")
    path: str        # same as id; the file the Canvas node points at
    title: str
    x: float
    y: float
    community: int   # global Louvain membership (reused from graph_export); -1 = none
    hop: int         # 0 = root, 1, 2
    subtitle: str | None = None


@dataclass
class MapEdge:
    src: str          # node id
    dst: str          # node id
    kind: str         # "wikilink" | "latent"
    weight: float


@dataclass
class MapView:
    root: str
    nodes: list[MapNode]
    edges: list[MapEdge]


@dataclass
class MapMaterials:
    """Everything build_mapview needs, injectable so tests need no live vault.

    `graph` is an undirected view of the wikilink graph (ids carry `.md`).
    `latent` is the already-normalised relatedness leg: (id_with_md, title, weight).
    """
    graph: object                         # nx.Graph-like: supports `in` and .neighbors()
    titles: dict[str, str]                # id -> display title
    community_of: dict[str, int]          # id -> global community (missing ⇒ -1)
    latent: list[tuple[str, str, float]] = field(default_factory=list)


def node_color(community: int) -> str:
    """Community colour, shared with /graph; muted slate for -1 (no community)."""
    if community < 0:
        return _MUTED
    from silica.kernel.graph_export import _community_color
    return _community_color(community)


# ---------------------------------------------------------------------------
# Neighbourhood selection + cap
# ---------------------------------------------------------------------------

def _with_md(path: str) -> str:
    return path if path.endswith(".md") else path + ".md"


def _stem_title(node_id: str) -> str:
    return node_id.rsplit("/", 1)[-1].removesuffix(".md")


def _bfs(root: str, graph: object, hops: int) -> dict[str, tuple[int, str | None]]:
    """BFS up to `hops` on the undirected wikilink graph.

    Returns id -> (hop, parent_id); root maps to (0, None). Neighbours are
    visited in sorted order so the result is deterministic.
    """
    seen: dict[str, tuple[int, str | None]] = {root: (0, None)}
    q: deque[str] = deque([root])
    while q:
        u = q.popleft()
        hop, _ = seen[u]
        if hop >= hops or u not in graph:
            continue
        for v in sorted(graph.neighbors(u)):
            if v not in seen:
                seen[v] = (hop + 1, u)
                q.append(v)
    return seen


def _select(
    root: str,
    materials: MapMaterials,
    *,
    max_nodes: int,
    hops: int,
) -> dict[str, tuple[int, str | None]]:
    """Pick the capped node set: root + wikilink BFS + latent, priority-ordered.

    Returns selected: id -> (hop, parent). Priority tiers (kept top `max_nodes`):
    root, wikilink hop-1, wikilink hop-2, latent — so wikilink hop-1 always
    outranks latent neighbours.
    """
    reached = _bfs(root, materials.graph, hops=hops)
    candidates: list[tuple[int, float, str]] = []  # (tier, -weight, id) sort key

    for nid, (hop, _parent) in reached.items():
        if nid == root:
            continue
        tier = 1 if hop == 1 else 2  # hop-1 outranks all deeper wikilink hops
        candidates.append((tier, -1.0, nid))

    for lid, _title, score in materials.latent:
        if lid == root or lid in reached:
            continue
        candidates.append((3, -float(score), lid))

    candidates.sort()
    kept = {root}
    for _tier, _negw, nid in candidates[: max(0, max_nodes - 1)]:
        kept.add(nid)

    selected: dict[str, tuple[int, str | None]] = {root: (0, None)}
    for nid in kept:
        if nid == root:
            continue
        if nid in reached:
            selected[nid] = reached[nid]
        else:
            selected[nid] = (1, root)  # latent neighbours hang off the root at hop 1
    return selected


# ---------------------------------------------------------------------------
# Radial wedge layout (deterministic; the only new algorithmic piece)
# ---------------------------------------------------------------------------

def _layout(nodes: list[MapNode], parent: dict[str, str | None]) -> None:
    """Place nodes radially, mutating each node's x/y. Root stays at (0, 0).

    360° is partitioned into one wedge per community present, width ∝ the
    community's node count. hop-1 sit on ring r1, hop-2 on ring r2, each spread
    across their community's wedge (hop-2 ordered by their parent's angle so
    children trail their parents — ponytail: the spec's "fan centred on parent"
    degrades to within-wedge ordering, which keeps every node inside its own
    community wedge). Radii are scaled so no two boxes overlap.
    """
    non_root = [n for n in nodes if n.hop > 0]
    if not non_root:
        return

    by_comm: dict[int, list[MapNode]] = defaultdict(list)
    for n in non_root:
        by_comm[n.community].append(n)

    total = len(non_root)
    wedge: dict[int, tuple[float, float]] = {}
    cursor = 0.0
    for c in sorted(by_comm):                       # sorted ⇒ deterministic
        width = 2 * math.pi * len(by_comm[c]) / total
        wedge[c] = (cursor, cursor + width)
        cursor += width

    # Two rings only: hop-1 inner, everything deeper on the outer ring.
    def _ring(hop: int) -> int:
        return 1 if hop == 1 else 2

    angle: dict[str, float] = {}
    slots: dict[int, list[float]] = {1: [], 2: []}  # slot widths per ring, for min-gap

    # Ring 1 first (hop-1), so ring-2 can be ordered by their parent's angle.
    for c in sorted(by_comm):
        start, end = wedge[c]
        ring1 = sorted((n for n in by_comm[c] if _ring(n.hop) == 1), key=lambda n: n.id)
        if ring1:
            slot = (end - start) / len(ring1)
            slots[1].append(slot)
            for i, n in enumerate(ring1):
                angle[n.id] = start + (i + 0.5) * slot

    for c in sorted(by_comm):
        start, end = wedge[c]
        center = (start + end) / 2
        ring2 = sorted(
            (n for n in by_comm[c] if _ring(n.hop) == 2),
            key=lambda n: (angle.get(parent.get(n.id) or "", center), n.id),
        )
        if ring2:
            slot = (end - start) / len(ring2)
            slots[2].append(slot)
            for i, n in enumerate(ring2):
                angle[n.id] = start + (i + 0.5) * slot

    def ring_radius(ring: int) -> float:
        count = sum(1 for n in non_root if _ring(n.hop) == ring)
        r = _DIAG                                   # root ↔ ring-1 spacing floor
        if count >= 2 and slots[ring]:
            min_gap = min(slots[ring])              # ≤ π when count ≥ 2 ⇒ sin > 0
            r = max(r, _DIAG / (2 * math.sin(min_gap / 2)))
        return r

    r1 = ring_radius(1)
    r2 = max(ring_radius(2), r1 + _DIAG)            # outer ring clears the inner by ≥ diag
    radius = {1: r1, 2: r2}

    for n in non_root:
        a = angle[n.id]
        r = radius[_ring(n.hop)]
        n.x = r * math.cos(a)
        n.y = r * math.sin(a)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_mapview(
    root: str,
    materials: MapMaterials,
    *,
    max_nodes: int = 35,
    hops: int = 2,
) -> MapView:
    """Build a MapView rooted on `root` from injected materials (pure)."""
    root = _with_md(root)
    selected = _select(root, materials, max_nodes=max_nodes, hops=hops)

    parent = {nid: p for nid, (_hop, p) in selected.items()}
    nodes: list[MapNode] = []
    for nid, (hop, _p) in selected.items():
        nodes.append(
            MapNode(
                id=nid,
                path=nid,
                title=materials.titles.get(nid, _stem_title(nid)),
                x=0.0,
                y=0.0,
                community=materials.community_of.get(nid, -1),
                hop=hop,
                subtitle=nid.rsplit("/", 1)[0] if "/" in nid else None,
            )
        )
    _layout(nodes, parent)

    ids = {n.id for n in nodes}
    edges: list[MapEdge] = []
    seen_pairs: set[tuple[str, str]] = set()

    # Wikilink edges: any graph edge among the selected nodes (tree + cross-branch).
    graph = materials.graph
    for u in ids:
        if u not in graph:
            continue
        for v in graph.neighbors(u):
            if v not in ids:
                continue
            key = (u, v) if u <= v else (v, u)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            edges.append(MapEdge(src=key[0], dst=key[1], kind="wikilink", weight=1.0))

    # Latent edges: root → each surviving latent neighbour not already wiki-linked.
    for lid, _title, score in materials.latent:
        lid = _with_md(lid)
        if lid not in ids or lid == root:
            continue
        key = (root, lid) if root <= lid else (lid, root)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        edges.append(MapEdge(src=root, dst=lid, kind="latent", weight=float(score)))

    return MapView(root=root, nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------

def _xy(n: MapNode) -> tuple[int, int]:
    """Rounded integer coordinates — shared by both serializers so they agree."""
    return round(n.x), round(n.y)


def _side(dx: float, dy: float) -> str:
    """Nearest box side for an edge leaving toward (dx, dy). Canvas y grows down."""
    if abs(dx) >= abs(dy):
        return "right" if dx >= 0 else "left"
    return "bottom" if dy >= 0 else "top"


# Wikilink edges keep a full colour; latent edges degrade to a muted colour +
# the "≈" label, because JSON Canvas has no dashed-edge style (only colour+label).
_CANVAS_WIKI_COLOR = "#22d3ee"
_CANVAS_LATENT_COLOR = _MUTED


def mapview_to_canvas(mv: MapView) -> dict:
    """Serialize a MapView to a JSON Canvas dict (jsoncanvas.org)."""
    nodes = []
    for n in mv.nodes:
        x, y = _xy(n)
        node: dict = {
            "id": n.id,
            "type": "file",
            "file": n.path,
            "x": x - round(BOX_W / 2),
            "y": y - round(BOX_H / 2),
            "width": round(BOX_W),
            "height": round(BOX_H),
        }
        if n.community >= 0:
            node["color"] = node_color(n.community)
        nodes.append(node)

    by_id = {n.id: n for n in mv.nodes}
    edges = []
    for i, e in enumerate(mv.edges):
        s, d = by_id[e.src], by_id[e.dst]
        latent = e.kind == "latent"
        edges.append({
            "id": f"e{i}",
            "fromNode": e.src,
            "toNode": e.dst,
            "fromSide": _side(d.x - s.x, d.y - s.y),
            "toSide": _side(s.x - d.x, s.y - d.y),
            "color": _CANVAS_LATENT_COLOR if latent else _CANVAS_WIKI_COLOR,
            "label": "≈" if latent else "",
        })
    return {"nodes": nodes, "edges": edges}


def mapview_to_gui(mv: MapView) -> dict:
    """Serialize a MapView to the GUI payload — the SAME positions as the canvas."""
    nodes = []
    for n in mv.nodes:
        x, y = _xy(n)
        nodes.append({
            "id": n.id,
            "title": n.title,
            "x": x,
            "y": y,
            "community": n.community,
            "hop": n.hop,
            "subtitle": n.subtitle,
            "color": node_color(n.community),
        })
    edges = [
        {"src": e.src, "dst": e.dst, "kind": e.kind, "weight": e.weight}
        for e in mv.edges
    ]
    return {"root": mv.root, "nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Static SVG render (GUI surface; native, zero deps, positions precomputed)
# ---------------------------------------------------------------------------

def _clip_to_box(ox: float, oy: float, hw: float, hh: float, dx: float, dy: float) -> tuple[float, float]:
    """Point where the ray from (ox,oy) toward (ox+dx,oy+dy) exits the axis-aligned
    box of half-extents (hw,hh) centred at (ox,oy). Used to trim edge endpoints to
    a card's border instead of its centre."""
    tx = hw / abs(dx) if dx else math.inf
    ty = hh / abs(dy) if dy else math.inf
    t = min(tx, ty)
    if not math.isfinite(t):
        return ox, oy
    return ox + t * dx, oy + t * dy


def render_map_svg(mv: MapView, title: str = "Mindmap") -> str:
    """Render a MapView as a self-contained, interactive SVG page.

    Consumes the precomputed positions (no force layout ⇒ cannot diverge from the
    canvas). Cards carry a community wash + accent bar; wikilink edges are solid
    curves with an arrowhead on true parent→child hops (same-ring wikilinks and
    all latent edges stay arrowless — they aren't "downstream" relationships).
    Pan/zoom/click-to-focus are plain SVG + vanilla JS (no new dependency),
    mirroring the dim-on-focus idiom already shipped for /graph.
    """
    import html

    by_id = {n.id: n for n in mv.nodes}
    pad = BOX_W
    xs = [n.x for n in mv.nodes] or [0.0]
    ys = [n.y for n in mv.nodes] or [0.0]
    min_x = min(xs) - pad
    min_y = min(ys) - pad
    vb_w = (max(xs) - min(xs)) + 2 * pad
    vb_h = (max(ys) - min(ys)) + 2 * pad

    # Every node at a given hop sits on the exact same radius (see _layout), so
    # any representative node gives the ring's true radius — a real structural
    # readout, not a decorative circle.
    ring_r = {n.hop: math.hypot(n.x, n.y) for n in mv.nodes if n.hop in (1, 2)}
    guide_svg = "".join(
        f'<circle class="ring-guide" cx="0" cy="0" r="{r:.1f}"/>'
        for r in sorted(set(ring_r.values()))
    )
    halo_r = min(70.0, ring_r.get(1, 110.0) * 0.55)

    edge_svg = []
    for e in mv.edges:
        s, d = by_id[e.src], by_id[e.dst]
        latent = e.kind == "latent"
        # Bow the line into a deterministic quadratic curve (perpendicular
        # offset from the midpoint) — an organic arc instead of a straight
        # ruler line, with no randomness so the render stays reproducible.
        parent, child = (s, d) if s.hop <= d.hop else (d, s)
        x1, y1, x2, y2 = parent.x, parent.y, child.x, child.y
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy) or 1.0
        bow = min(36.0, length * 0.16)
        cx, cy = mx - dy / length * bow, my + dx / length * bow
        # Trim both ends to the card's border along the curve's own tangent
        # (the direction toward the control point) so the line touches the
        # badge edge and never its interior.
        sx, sy = _clip_to_box(x1, y1, BOX_W / 2, BOX_H / 2, cx - x1, cy - y1)
        ex, ey = _clip_to_box(x2, y2, BOX_W / 2, BOX_H / 2, cx - x2, cy - y2)
        is_tree = parent.hop != child.hop
        cls = "edge latent" if latent else ("edge wiki" if is_tree else "edge wiki lateral")
        marker = ' marker-end="url(#arrow)"' if is_tree and not latent else ""
        edge_svg.append(
            f'<path class="{cls}" data-src="{html.escape(e.src, quote=True)}" '
            f'data-dst="{html.escape(e.dst, quote=True)}" '
            f'd="M {sx:.1f} {sy:.1f} Q {cx:.1f} {cy:.1f} {ex:.1f} {ey:.1f}"{marker}/>'
        )

    node_svg = []
    for n in mv.nodes:
        color = node_color(n.community)
        rx, ry = n.x - BOX_W / 2, n.y - BOX_H / 2
        root_cls = " root" if n.hop == 0 else ""
        title_esc = html.escape(n.title)
        sub_html = f'<div class="card-sub">{html.escape(n.subtitle)}</div>' if n.subtitle else ""
        node_svg.append(
            f'<g class="card{root_cls}" data-id="{html.escape(n.id, quote=True)}" '
            f'transform="translate({rx:.1f},{ry:.1f})">'
            f'<rect class="frame" width="{BOX_W}" height="{BOX_H}" rx="10" '
            f'fill="{color}" stroke="{color}"/>'
            f'<rect class="accent" width="4" height="{BOX_H}" rx="2" fill="{color}"/>'
            f'<foreignObject x="16" y="0" width="{BOX_W - 28}" height="{BOX_H}">'
            f'<div xmlns="http://www.w3.org/1999/xhtml" class="card-body">'
            f'<div class="card-title">{title_esc}</div>{sub_html}</div>'
            f'</foreignObject></g>'
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root{{
    --void:#0B0D12;--slate-2:#151A23;--line:#1E2530;--line-2:#2B3442;
    --frost:#E7EBF1;--ash:#8A93A3;--ash-dim:#5A6372;--cyan:#22D3EE;
    --mono:ui-monospace,"Cascadia Code","SF Mono",Menlo,Consolas,"DejaVu Sans Mono",monospace;
  }}
  *{{box-sizing:border-box}}
  html,body{{margin:0;height:100%;background:var(--void);overflow:hidden;
             -webkit-user-select:none;user-select:none}}
  svg{{width:100%;height:100vh;display:block;cursor:grab;touch-action:none}}
  svg.panning{{cursor:grabbing}}
  .ring-guide{{fill:none;stroke:var(--line-2);stroke-width:1;stroke-dasharray:2 6;opacity:.6}}
  .root-halo{{animation:halo 2.8s ease-in-out infinite}}
  @media (prefers-reduced-motion:reduce){{.root-halo{{animation:none}}}}
  @keyframes halo{{0%,100%{{opacity:.5;transform:scale(1)}}50%{{opacity:.18;transform:scale(1.12)}}}}
  .edge{{fill:none;stroke:var(--cyan);stroke-width:1.6;stroke-opacity:.55;
         transition:opacity .15s ease,stroke-opacity .15s ease}}
  .edge.lateral{{stroke-width:1.1;stroke-opacity:.28}}
  .edge.latent{{stroke:var(--ash-dim);stroke-dasharray:6 6;stroke-opacity:.55}}
  .edge.dim{{opacity:.1}}
  .card{{cursor:pointer;transition:opacity .15s ease}}
  .card.dim{{opacity:.3}}
  .card .frame{{fill-opacity:.12;stroke-opacity:.6;stroke-width:1;
                filter:drop-shadow(0 2px 6px rgba(0,0,0,.55))}}
  .card:hover .frame{{fill-opacity:.2;stroke-opacity:1}}
  .card.root .frame{{stroke:var(--cyan);stroke-opacity:1;stroke-width:2.5}}
  .card-body{{font-family:var(--mono);height:100%;display:flex;flex-direction:column;
              justify-content:center;gap:3px;pointer-events:none;overflow:hidden}}
  .card-title{{font-size:12.5px;font-weight:600;line-height:1.25;color:var(--frost);
               display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}}
  .card-sub{{font-size:10px;color:var(--ash-dim);letter-spacing:.03em;
             white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
  #hud{{position:fixed;top:14px;right:14px;display:flex;flex-direction:column;
        align-items:flex-end;gap:8px;font-family:var(--mono);z-index:2}}
  #fit-btn{{padding:7px 10px;background:var(--slate-2);border:1px solid var(--line-2);
            color:var(--ash);font-family:var(--mono);font-size:12px;cursor:pointer;border-radius:3px}}
  #fit-btn:hover{{border-color:var(--cyan);color:var(--cyan)}}
  #legend{{display:flex;flex-direction:column;gap:5px;background:var(--slate-2);
           border:1px solid var(--line);border-radius:3px;padding:8px 10px;
           font-size:11px;color:var(--ash)}}
  .legend-row{{display:flex;align-items:center;gap:7px}}
  .swatch{{width:20px;height:0;border-top:2px solid var(--cyan)}}
  .swatch.dashed{{border-top-style:dashed;border-color:var(--ash-dim)}}
</style>
</head>
<body>
<svg id="stage" viewBox="{min_x:.0f} {min_y:.0f} {vb_w:.0f} {vb_h:.0f}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--cyan)" fill-opacity=".8"/>
    </marker>
    <radialGradient id="halo-grad" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#22D3EE" stop-opacity=".4"/>
      <stop offset="100%" stop-color="#22D3EE" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <g id="scene">
    {guide_svg}
    <circle class="root-halo" cx="0" cy="0" r="{halo_r:.0f}" fill="url(#halo-grad)"/>
    <g id="edges">{"".join(edge_svg)}</g>
    <g id="nodes">{"".join(node_svg)}</g>
  </g>
</svg>
<div id="hud">
  <button id="fit-btn" type="button">⊹ Fit map</button>
  <div id="legend">
    <div class="legend-row"><span class="swatch"></span>wikilink</div>
    <div class="legend-row"><span class="swatch dashed"></span>related (≈)</div>
  </div>
</div>
<script>
const stage = document.getElementById("stage");
const scene = document.getElementById("scene");
let tx = 0, ty = 0, scale = 1;
function applyTransform() {{
  scene.setAttribute("transform", "translate(" + tx + "," + ty + ") scale(" + scale + ")");
}}
function toBase(evt) {{
  const pt = stage.createSVGPoint();
  pt.x = evt.clientX; pt.y = evt.clientY;
  return pt.matrixTransform(stage.getScreenCTM().inverse());
}}

// Pan: drag the background. Zoom: wheel, anchored on the cursor so the point
// under it stays put (tx/ty live in the SVG's own root coordinate space, which
// getScreenCTM() reports independently of the inner group's own transform).
let dragging = false, lastX = 0, lastY = 0;
stage.addEventListener("pointerdown", (e) => {{
  if (e.target.closest(".card")) return;
  dragging = true; lastX = e.clientX; lastY = e.clientY;
  stage.classList.add("panning");
  stage.setPointerCapture(e.pointerId);
}});
stage.addEventListener("pointermove", (e) => {{
  if (!dragging) return;
  const k = {vb_w:.1f} / stage.clientWidth;
  tx += (e.clientX - lastX) * k; ty += (e.clientY - lastY) * k;
  lastX = e.clientX; lastY = e.clientY;
  applyTransform();
}});
function endDrag() {{ dragging = false; stage.classList.remove("panning"); }}
stage.addEventListener("pointerup", endDrag);
stage.addEventListener("pointerleave", endDrag);

stage.addEventListener("wheel", (e) => {{
  e.preventDefault();
  const p = toBase(e);
  const wx = (p.x - tx) / scale, wy = (p.y - ty) / scale;
  scale = Math.min(4, Math.max(0.3, scale * (e.deltaY > 0 ? 0.9 : 1.1)));
  tx = p.x - wx * scale; ty = p.y - wy * scale;
  applyTransform();
}}, {{ passive: false }});

document.getElementById("fit-btn").addEventListener("click", () => {{
  tx = 0; ty = 0; scale = 1; applyTransform();
}});

// --- click-to-focus: dim everything except the clicked card + its 1-hop
// edges (same idiom as /graph); background click clears. Embedded in the app
// iframe, a click also hands the note off to the parent's note panel.
const neighbors = {{}};
document.querySelectorAll(".edge").forEach((el) => {{
  const a = el.dataset.src, b = el.dataset.dst;
  (neighbors[a] = neighbors[a] || new Set()).add(b);
  (neighbors[b] = neighbors[b] || new Set()).add(a);
}});

function focusNode(id) {{
  document.querySelectorAll(".card").forEach((el) => {{
    const nb = neighbors[id] || new Set();
    el.classList.toggle("dim", id != null && el.dataset.id !== id && !nb.has(el.dataset.id));
  }});
  document.querySelectorAll(".edge").forEach((el) => {{
    el.classList.toggle("dim", id != null && el.dataset.src !== id && el.dataset.dst !== id);
  }});
}}

document.querySelectorAll(".card").forEach((el) => {{
  el.addEventListener("click", () => {{
    focusNode(el.dataset.id);
    if (window.parent !== window) {{
      window.parent.postMessage({{ type: "silica-open-note", path: el.dataset.id }}, "*");
    }}
  }});
}});
stage.addEventListener("click", (e) => {{ if (!e.target.closest(".card")) focusNode(null); }});

const knownIds = new Set(Array.from(document.querySelectorAll(".card")).map((el) => el.dataset.id));
window.addEventListener("message", (e) => {{
  if (e.data && e.data.type === "silica-focus-path") {{
    focusNode(knownIds.has(e.data.path) ? e.data.path : null);
  }}
}});
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# Live-vault material gathering (IO; the tool/endpoint call this)
# ---------------------------------------------------------------------------

def _resolve_in(note: str, note_paths: list[str], titles: dict[str, str]) -> str | None:
    """Resolve `note` (a path OR a title) to a graph key. Pure; sorted ⇒ stable.

    Graph keys are full vault-relative paths WITH .md; a user (or the GUI input)
    may give a bare title or a path with/without .md. Try exact path first, then
    match by basename/title case-insensitively.
    """
    keys = sorted(p.replace("\\", "/") for p in note_paths)
    key_set = set(keys)
    cand = note.replace("\\", "/").strip()
    for c in (cand, cand + ".md", cand.removesuffix(".md")):
        if c in key_set:
            return c
    target = cand.removesuffix(".md").rsplit("/", 1)[-1].lower()
    for path in keys:
        stem = path.removesuffix(".md").rsplit("/", 1)[-1].lower()
        if stem == target or (titles.get(path, "") or "").lower() == target:
            return path
    return None


def note_resolver():
    """One driver read → a pure closure: ref (path or title) -> graph key | None.

    Reuse when resolving many refs per render (e.g. linkifying a message): the
    driver graph is read once, the returned callable does no further IO.
    """
    from silica.driver import get_driver

    notes, _unresolved, _g = get_driver().graph_data("")
    titles = {p.replace("\\", "/"): ref.name for p, ref in notes.items()}
    paths = list(notes)
    return lambda ref: _resolve_in(ref, paths, titles)


def resolve_note_path(note: str) -> str | None:
    """Resolve a note path or title to its vault-relative graph key (with .md)."""
    return note_resolver()(note)


def gather_materials(root: str, *, latent_k: int = 10) -> MapMaterials:
    """Collect wikilink graph, titles, global communities, and the latent leg."""
    from silica.driver import get_driver
    from silica.kernel.graph_export import build_graph_data, detect_communities
    from silica.kernel.relatedness import related_notes

    driver = get_driver()
    notes, _unresolved, g = driver.graph_data("")
    titles = {p.replace("\\", "/"): ref.name for p, ref in notes.items()}

    nodes, edges = build_graph_data("")
    detect_communities(nodes, edges)  # assigns node["group"] in place (global, seed=42)
    community_of = {n["id"]: n.get("group", -1) for n in nodes}

    latent: list[tuple[str, str, float]] = []
    try:
        embed_store, cooccur_store = _load_stores()
        for r in related_notes(
            _with_md(root), embed_store=embed_store, cooccur_store=cooccur_store, k=latent_k
        ):
            latent.append((_with_md(r.path), r.name, r.score))
    except Exception:
        latent = []

    undirected = g.to_undirected(as_view=True) if hasattr(g, "to_undirected") else g
    return MapMaterials(
        graph=undirected, titles=titles, community_of=community_of, latent=latent
    )


def _load_stores():
    """(embed_store, cooccur_store), each None when empty/unavailable ⇒ leg abstains."""
    from silica.config import CONFIG

    embed_store = None
    try:
        from silica.kernel.embed import get_store
        es = get_store()
        embed_store = es if len(es) > 0 else None
    except Exception:
        embed_store = None

    cooccur_store = None
    try:
        from silica.kernel.cooccurrence import get_cooccur_store
        cs = get_cooccur_store(lang=CONFIG.cooccurrence_lang)
        cooccur_store = cs if len(cs) > 0 else None
    except Exception:
        cooccur_store = None

    return embed_store, cooccur_store
