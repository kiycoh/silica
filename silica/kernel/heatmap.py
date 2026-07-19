# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Concept co-occurrence heatmap — a concept x concept matrix over the cooccur
store, rendered as a self-contained SVG page (mindmap idioms: server-side
layout, zero deps, offline).

Seriation is the design: rows/columns are grouped by Louvain community over
the concept graph, so topics read as diagonal blocks and hot cells OUTSIDE a
block are cross-topic bridges — the one thing a matrix shows that the force
graph cannot. Concept selection uses a df window (min_df / df_cap over
CORRELATE Concept sets) to drop degenerate and hub concepts.
"""
from __future__ import annotations

import html
import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_BRIDGE_COLOR = "#c9a227"  # gold — cross-community cells, same hue as /graph's discourse badge


@dataclass(frozen=True)
class HeatmapView:
    stems: list[str]        # render order (community blocks, df desc within)
    labels: list[str]       # parallel display surfaces
    df: list[int]           # parallel note-frequency
    community: list[int]    # parallel block id (contiguous runs), -1 = none
    matrix: list[list[float]]  # symmetric co-occurrence weights, diag 0.0
    focus: str | None = None   # exact-resolved focus stem
    hits: frozenset[str] = frozenset()  # highlighted stems (fuzzy matches, or a note's own concepts)
    note: str | None = None    # overlay message ("X not found — …")
    cap: int = 40              # the top_n this view was built with (form echo)
    min_pct: int = 0           # weight floor as % of matrix max (form echo)


def build_heatmap(store, *, top_n: int = 40, min_df: int = 2,
                  df_cap: int | None = None, focus: str | None = None,
                  min_pct: int = 0, note: str | None = None) -> HeatmapView:
    """Pure read over the store. df window: min_df drops degenerate single-note
    concepts, df_cap (default 5% of notes, floor 3) drops hub concepts.

    `focus` asks a different question — "what does THIS word correlate
    with" — so it bypasses the df window entirely: the resolved concept plus
    its top_n-1 strongest co-occurrence neighbors, hubs and rare stems
    included.

    `note` asks a third: "how do THIS note's concepts correlate vault-wide" —
    the note's own Concept set (highlighted) plus its strongest out-of-note
    neighbors, which is what the matrix adds beyond the note body itself."""
    from silica.kernel.correlate import topk_set

    df: dict[str, int] = {}
    paths = store.paths()
    for p in paths:
        for stem in topk_set(store.note_nodes(p)):
            df[stem] = df.get(stem, 0) + 1
    adj = store.adjacency()

    focus_stem = None
    hits: frozenset[str] = frozenset()
    msg = None
    kept = None
    if focus is not None:
        q = focus.strip().lower()
        focus_stem = _resolve_focus(store, adj, q)
        if focus_stem is not None:
            neigh = sorted(adj.get(focus_stem, {}).items(), key=lambda kv: (-kv[1], kv[0]))
            kept = [focus_stem] + [s for s, _w in neigh[: top_n - 1]]
        else:
            # Fuzzy fallback: substring over stems and labels, best-connected
            # first — never a dead end. Remaining budget goes to the matches'
            # strongest co-occurrents so the correlations are actually visible.
            matched = sorted(
                (s for s in adj if q and (q in s or q in store.node_label(s).lower())),
                key=lambda s: (-sum(adj[s].values()), s))[:top_n]
            if matched:
                hits = frozenset(matched)
                msg = (f'concept "{focus.strip()}" not found — showing '
                       f'{len(matched)} related concept{"s" if len(matched) > 1 else ""}')
                kept = _pad_with_neighbors(matched, adj, top_n)
            else:
                msg = f'concept "{focus.strip()}" not found'
    elif note is not None:
        path = _resolve_note(store, note)
        own = [] if path is None else sorted(
            (s for s in topk_set(store.note_nodes(path)) if s in adj),
            key=lambda s: (-df.get(s, 0), s))[:top_n]
        if own:
            hits = frozenset(own)
            kept = _pad_with_neighbors(own, adj, top_n)
        else:
            msg = (f'note "{note.strip()}" not found' if path is None
                   else f'"{note.strip()}" has no co-occurring concepts')
    if kept is None:  # no focus/note, or one that matched nothing -> default view
        if df_cap is None:
            df_cap = max(3, math.ceil(0.05 * len(paths)))
        kept = [s for s, d in df.items() if min_df <= d <= df_cap]
        kept = sorted(kept, key=lambda s: (-df[s], s))[:top_n]
    if not kept:
        return HeatmapView([], [], [], [], [], note=msg, cap=top_n)

    comm = _communities(kept, adj)
    # Blocks: bigger community first, min-stem tie-break; df desc within.
    blocks: dict[int, list[str]] = {}
    for s in kept:
        blocks.setdefault(comm[s], []).append(s)
    ordered_blocks = sorted(blocks.values(), key=lambda ss: (-len(ss), min(ss)))
    stems: list[str] = []
    community: list[int] = []
    for i, block in enumerate(ordered_blocks):
        # df.get: focus/fuzzy neighbors come from adjacency (ALL stems), df only
        # covers Concept sets (top-30/note) — a weight-selected neighbor can
        # legitimately have no df entry.
        stems.extend(sorted(block, key=lambda s: (-df.get(s, 0), s)))
        community.extend([i if comm[block[0]] != -1 else -1] * len(block))

    matrix = [[float(adj.get(a, {}).get(b, 0.0)) if a != b else 0.0
               for b in stems] for a in stems]
    if min_pct > 0 and stems:
        # Weight floor. Reference: the FOCUS ROW's max when focusing (hub
        # neighbor pairs can dwarf every focus cell — a global max would blank
        # exactly what was searched for), the matrix max otherwise. Stems left
        # without a visible cell drop out; the focus concept always stays.
        if focus_stem is not None:
            ref = max(matrix[stems.index(focus_stem)], default=0.0)
        else:
            ref = max((w for row in matrix for w in row), default=0.0)
        thr = (min_pct / 100.0) * ref
        matrix = [[w if w >= thr else 0.0 for w in row] for row in matrix]
        alive = [i for i, s in enumerate(stems)
                 if s == focus_stem or any(w > 0.0 for w in matrix[i])]
        if len(alive) < len(stems):
            stems = [stems[i] for i in alive]
            community = [community[i] for i in alive]
            matrix = [[matrix[i][j] for j in alive] for i in alive]
    return HeatmapView(stems=stems,
                       labels=[store.node_label(s) for s in stems],
                       df=[df.get(s, 0) for s in stems],
                       community=community, matrix=matrix, focus=focus_stem,
                       hits=hits, note=msg, cap=top_n, min_pct=min_pct)


def _pad_with_neighbors(seeds: list[str], adj: dict[str, dict[str, float]],
                        top_n: int) -> list[str]:
    """Seeds plus their strongest out-of-seed neighbors, weight desc, to top_n."""
    sset = set(seeds)
    pool: dict[str, float] = {}
    for m in seeds:
        for s, w in adj.get(m, {}).items():
            if s not in sset:
                pool[s] = max(pool.get(s, 0.0), w)
    kept = list(seeds)
    for s, _w in sorted(pool.items(), key=lambda kv: (-kv[1], kv[0])):
        if len(kept) >= top_n:
            break
        kept.append(s)
    return kept


def _resolve_note(store, query: str) -> str | None:
    """Vault-relative path or bare title -> store key. Exact path first (the
    drawer passes paths), then basename-minus-.md; ambiguity resolves to the
    lexicographically first path."""
    q = query.strip().lower().removesuffix(".md")
    if not q:
        return None
    by_base: dict[str, str] = {}
    for p in sorted(store.paths()):
        pl = p.lower().removesuffix(".md")
        if pl == q:
            return p
        by_base.setdefault(pl.rsplit("/", 1)[-1], p)
    return by_base.get(q.rsplit("/", 1)[-1])


def _resolve_focus(store, adj: dict[str, dict[str, float]], query: str) -> str | None:
    """Surface word -> store stem: exact stem, then label match, then the
    store's own stemmer (so "training" finds stem "train"). Only concepts
    with edges qualify — a match with no co-occurrences has nothing to show."""
    q = query.strip().lower()
    if not q:
        return None
    if q in adj:
        return q
    for stem in adj:
        if store.node_label(stem).lower() == q:
            return stem
    from silica.kernel.cooccurrence import tokenize
    for sent in tokenize(q, stem_lang=store.lang, stopword_lang=store.lang):
        for stem, _surface in sent:
            if stem in adj:
                return stem
    return None


def _communities(stems: list[str], adj: dict[str, dict[str, float]]) -> dict[str, int]:
    """Louvain over the selected concept subgraph; seed pinned so the page is
    reproducible."""
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    sset = set(stems)
    G = nx.Graph()
    G.add_nodes_from(stems)
    for a in stems:
        for b, w in adj.get(a, {}).items():
            if b in sset and a < b:
                G.add_edge(a, b, weight=w)
    if G.number_of_edges() == 0:
        return dict.fromkeys(stems, -1)
    comms = louvain_communities(G, weight="weight", seed=0)
    return {s: i for i, c in enumerate(comms) for s in c}


_CELL = 18
_GUT_L = 170   # left gutter: row labels
_GUT_T = 130   # top gutter: rotated column labels


def render_heatmap_svg(view: HeatmapView, title: str = "Concept Heatmap") -> str:
    """Self-contained HTML page (no external refs). Cell color = community
    color inside a block, gold across blocks; opacity encodes weight.

    The focus HUD is kept for standalone open (`/heatmap` in a browser tab) but
    hidden when embedded in an iframe (`body.embedded`, same idiom as the graph
    viewer): inside the app the explore toolbar drives ?q=/?n=/?p=, and the
    note-drawer preview wants no controls at all."""
    from silica.kernel.graph_export import _community_color

    # Focus search: plain GET form back onto /heatmap — the server reads ?q= / ?n=.
    # Each field carries its own visible caption (not just a hover title) so the
    # unit — "min wt %" — reads at a glance instead of only on mouseover.
    form = (
        '<form id="hud">'
        '<label class="hud-f"><span class="hud-lbl">concept</span>'
        f'<input name="q" value="{html.escape(view.focus or "", quote=True)}" '
        'placeholder="focus a concept…" autocomplete="off"/></label>'
        '<label class="hud-f hud-narrow"><span class="hud-lbl">max concepts</span>'
        f'<input name="n" type="number" min="5" max="120" value="{view.cap}"/></label>'
        '<label class="hud-f hud-narrow"><span class="hud-lbl">min wt %</span>'
        f'<input name="p" type="number" min="0" max="95" value="{view.min_pct}"/></label>'
        '<button>focus</button>'
        '</form>'
    )
    overlay = (f'<div id="note">[ {html.escape(view.note)} ]</div>'
               if view.note else "")

    if not view.stems:
        return _page(title, form + overlay + '<div id="empty">[ NO CO-OCCURRENCE '
                     'DATA — run /cooccur to build the index ]</div>')

    n = len(view.stems)
    max_w = max((w for row in view.matrix for w in row), default=0.0) or 1.0
    parts: list[str] = []
    for i, label in enumerate(view.labels):
        esc = html.escape(label)
        hit = view.stems[i] == view.focus or view.stems[i] in view.hits
        cls = "lbl focus" if hit else "lbl"
        y = _GUT_T + i * _CELL
        parts.append(f'<text class="{cls}" x="{_GUT_L - 8}" y="{y + _CELL * 0.72:.0f}" '
                     f'text-anchor="end">{esc}</text>')
        x = _GUT_L + i * _CELL
        parts.append(f'<text class="{cls}" x="0" y="0" text-anchor="start" transform='
                     f'"translate({x + _CELL * 0.72:.0f},{_GUT_T - 8}) rotate(-60)">{esc}</text>')
    for i in range(n):
        for j in range(n):
            x, y = _GUT_L + j * _CELL, _GUT_T + i * _CELL
            if i == j:
                parts.append(f'<rect class="diag" x="{x}" y="{y}" '
                             f'width="{_CELL - 1}" height="{_CELL - 1}"/>')
                continue
            w = view.matrix[i][j]
            if w <= 0.0:
                continue
            same = view.community[i] == view.community[j] and view.community[i] != -1
            color = _community_color(view.community[i]) if same else _BRIDGE_COLOR
            parts.append(
                f'<rect class="cell" x="{x}" y="{y}" width="{_CELL - 1}" '
                f'height="{_CELL - 1}" fill="{color}" '
                f'fill-opacity="{0.2 + 0.8 * w / max_w:.2f}">'
                f'<title>{html.escape(view.labels[i])} × {html.escape(view.labels[j])}'
                f' — {w:g}</title></rect>')
    # Block separators: a line after each community run, both axes.
    edge = _GUT_L + n * _CELL
    for i in range(1, n):
        if view.community[i] != view.community[i - 1]:
            px = _GUT_L + i * _CELL - 0.5
            py = _GUT_T + i * _CELL - 0.5
            parts.append(f'<line class="sep" x1="{px}" y1="{_GUT_T}" x2="{px}" y2="{_GUT_T + n * _CELL}"/>')
            parts.append(f'<line class="sep" x1="{_GUT_L}" y1="{py}" x2="{edge}" y2="{py}"/>')
    svg = (f'<svg width="{edge + 24}" height="{_GUT_T + n * _CELL + 24}" '
           f'xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>')
    return _page(title, form + overlay + svg)


def _page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root{{
    --void:#0A0D14;--slate-2:#161B27;--line:#232A3A;--line-2:#38425A;
    --frost:#E8ECF5;--ash:#8B95AC;--ash-dim:#566076;
    --mono:ui-monospace,"Cascadia Code","SF Mono",Menlo,Consolas,"DejaVu Sans Mono",monospace;
  }}
  *{{box-sizing:border-box}}
  html,body{{margin:0;min-height:100%;background:var(--void);overflow:auto}}
  svg{{display:block}}
  .lbl{{font-family:var(--mono);font-size:11px;fill:var(--ash)}}
  .lbl.focus{{fill:var(--frost);font-weight:700}}
  .cell:hover{{stroke:var(--frost);stroke-width:1}}
  .diag{{fill:var(--line);fill-opacity:.6}}
  .sep{{stroke:var(--line-2);stroke-width:1}}
  #empty{{font-family:var(--mono);color:var(--ash-dim);padding:2rem;font-size:13px;letter-spacing:.06em}}
  #hud{{position:fixed;top:14px;right:14px;z-index:2}}
  #hud form{{display:flex;align-items:flex-end;gap:0}}
  .hud-f{{display:flex;flex-direction:column;gap:3px}}
  .hud-lbl{{font-family:var(--mono);font-size:9px;color:var(--ash-dim);text-transform:uppercase;
            letter-spacing:.06em;padding-left:1px}}
  #hud input{{background:var(--slate-2);border:1px solid var(--line-2);border-right:none;color:var(--frost);
              padding:6px 8px;font-family:var(--mono);font-size:12px;width:190px;height:29px}}
  #hud input:focus{{outline:none;border-color:var(--frost)}}
  #hud input::placeholder{{color:var(--ash-dim)}}
  #hud button{{background:var(--slate-2);border:1px solid var(--line-2);
               color:var(--ash);padding:0 10px;height:29px;font-family:var(--mono);font-size:11px;
               cursor:pointer;text-transform:uppercase;letter-spacing:.06em;align-self:flex-end}}
  #hud input[type=number]{{width:66px}}
  #hud button:hover{{color:var(--frost)}}
  /* Embedded in the app iframe: the explore toolbar owns ?q=/?n=/?p=, and the
     note-drawer preview wants no controls — hide the in-page HUD either way.
     Kept visible for a standalone /heatmap open. */
  body.embedded #hud{{display:none}}
  #note{{position:fixed;top:14px;left:14px;z-index:2;background:rgba(22,27,39,.82);
         border:1px solid var(--line-2);color:var(--ash);padding:7px 11px;
         font-family:var(--mono);font-size:12px;letter-spacing:.04em}}
</style></head><body>
{body}
<script>if(window.parent!==window)document.body.classList.add("embedded");</script>
</body></html>
"""


def heatmap_page(focus: str | None = None, top_n: int = 40, min_pct: int = 0,
                 note: str | None = None, title: str = "Concept Heatmap") -> str:
    """Resolve the active vault's cooccurrence store and render the page.
    Lives here (allowlisted store access) so the UI layer never touches
    kernel.cooccurrence directly. top_n/min_pct come from the page's own
    form — clamp, don't trust.

    A `note`-scoped call is the drawer's preview: a glance at what this note's
    concepts correlate with, not a browsing tool, so it defaults small (5). The
    in-page HUD hides itself when embedded (see render_heatmap_svg)."""
    from silica.config import CONFIG
    from silica.kernel import cooccurrence

    top_n = max(5, min(120, top_n or (5 if note else 40)))
    min_pct = max(0, min(95, min_pct))
    lang = cooccurrence.frozen_lang(getattr(CONFIG, "vault_path", "") or "")
    store = cooccurrence.get_cooccur_store(lang or "english")
    return render_heatmap_svg(
        build_heatmap(store, focus=focus, top_n=top_n, min_pct=min_pct, note=note),
        title=title)
