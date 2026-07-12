# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Output renderers for a VaultReport: markdown, facts, digest, files.

Read-only over the report — no graph computation, no signal logic.
"""
from __future__ import annotations

import logging
from pathlib import Path

import orjson

from silica.kernel.graph_report.models import VaultReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output functions
# ---------------------------------------------------------------------------

_MEMBERS_CAP = 25  # max members shown per cluster in markdown
_LIST_CAP = 30     # max bullet/row items for long lists (orphans, dangling, …)
_MIN_CLUSTER = 2   # size-1 clusters are noise — summarised, never listed


def _short(p: str) -> str:
    return p.rsplit("/", 1)[-1].removesuffix(".md") if p else "—"


_TOTAL_LABELS = {
    "dangling_links": "Broken links (point nowhere)",
    "missing_links": "Missing links (proposed)",
    "duplicate_pairs": "Related pairs (borderline — link, not merge)",
    "confirmed_duplicates": "Likely duplicates (merge candidates)",
    "autolink_candidates": "Autolink candidates",
    "lean_notes": "Thin notes (enrich?)",
    "reformat_notes": "Notes to reformat",
    "orphans": "Orphans (no incoming links)",
    "structural_gaps": "Structural gaps (disconnected areas)",
}


def _fold(add, kind: str, title: str, items: list, fmt, *, cap: int = _LIST_CAP, more_json: bool = True) -> None:
    """Wrap a bulleted list in a collapsed OFM callout (`> [!kind]- title`).

    Every line is `>`-prefixed so it renders inside the callout and the
    `[[wikilinks]]` survive — an HTML <details> fold would swallow them.
    `fmt(item)` returns each bullet's text; the list is capped at `cap`.
    Trailing blank line separates this callout from the next block.
    """
    add(f"> [!{kind}]- {title}")
    for it in items[:cap]:
        add(f"> - {fmt(it)}")
    if len(items) > cap:
        tail = " (see GRAPH_REPORT.json)" if more_json else ""
        add(f"> - _… +{len(items) - cap} more{tail}_")
    add("")


def to_markdown(r: VaultReport, title: str = "Silica Vault Report") -> str:
    """Render a VaultReport as OFM-friendly, human-readable markdown.

    Long lists fold into collapsed callouts (`[!kind]-`) so the note opens
    compact; tables (Totals, God Nodes, Bridges) stay flat since tables render
    poorly inside callouts. Singleton clusters are summarised, lists capped;
    the full data lives in the sibling GRAPH_REPORT.json.
    """
    lines: list[str] = []
    add = lines.append

    add(f"# {title}")
    add(f"_Generated: {r.generated_at}_")
    if r.scope:
        add(f"_Scope: `{r.scope}`_")
    add("")

    # cluster_id -> hub name, to label god-nodes & bridges by area (not by number)
    hub_of = {c.cluster_id: _short(c.hub) for c in r.clusters}
    linked = sorted(
        (c for c in r.clusters if c.size >= _MIN_CLUSTER),
        key=lambda c: c.size, reverse=True,
    )
    singletons = sum(1 for c in r.clusters if c.size < _MIN_CLUSTER)
    t = r.totals

    # Summary (prose — the part a human actually reads)
    add("## Summary")
    add(
        f"This vault holds **{t.get('notes', 0)} notes** connected by "
        f"**{t.get('links', 0)} links**, forming **{len(linked)} linked areas** "
        f"plus {singletons} standalone notes."
    )
    if linked:
        top = ", ".join(f"[[{_short(c.hub)}]] ({c.size})" for c in linked[:5])
        add(f"Largest areas: {top}.")
    if r.discourse_state:
        add(f"Discourse shape: **{r.discourse_state}**.")
    add("")
    health = []
    if t.get("orphans"):
        health.append(f"{t['orphans']} orphans (no incoming links)")
    if t.get("dangling_links"):
        health.append(f"{t['dangling_links']} broken links (point to notes that don't exist)")
    if t.get("contested"):
        health.append(f"{t['contested']} contested notes (unresolved contradictions)")
    if t.get("source_drift"):
        health.append(f"{t['source_drift']} notes derived from a superseded source version")
    if health:
        add("> [!warning] Health")
        for h in health:
            add(f"> - {h}")
        add("")
    else:
        add("> [!success] Health")
        add("> No orphans, broken links, or contradictions.")
        add("")
    fixes = [
        f"{t[k]} {word}"
        for k, word in (
            ("autolink_candidates", "autolink candidates"),
            ("lean_notes", "notes to enrich"),
            ("confirmed_duplicates", "likely duplicates to merge"),
            ("duplicate_pairs", "borderline-related pairs"),
            ("reformat_notes", "notes to reformat"),
        )
        if t.get(k)
    ]
    if fixes:
        add("> [!tip] Suggestions ready (not applied)")
        add("> " + " · ".join(fixes) + ".")
        add("> Nothing changes until you approve — review below, then ask Silica to apply.")
        add("")

    # Totals
    add("## Totals")
    add("| Metric | Count |")
    add("|---|---|")
    for k, v in r.totals.items():
        add(f"| {_TOTAL_LABELS.get(k, k.replace('_', ' ').capitalize())} | {v} |")
    add("")

    # God nodes (PageRank dropped — it reads 0.0 at this scale; degree is the signal)
    add("## God Nodes (High-Degree Hubs)")
    if r.god_nodes:
        # Betweenness rides alongside degree: a hub with high betweenness is also
        # a bottleneck (its removal fragments the discourse), not just popular.
        add("| Note | Area | Links | In | Out | Between |")
        add("|---|---|---|---|---|---|")
        for n in r.god_nodes:
            area = hub_of.get(n.cluster, f"#{n.cluster}")
            add(f"| [[{n.label}]] | {area} | {n.degree} | {n.in_degree} | {n.out_degree} | {n.betweenness} |")
    else:
        add("_No connected notes found._")
    add("")

    # Surprising bridges
    add("## Surprising Cross-Cluster Connections")
    add("_Links joining two otherwise-separate areas — often the most interesting._")
    if r.bridges:
        add("| Source | Target | Areas joined | Surprise |")
        add("|---|---|---|---|")
        for b in r.bridges:
            sa = hub_of.get(b.source_cluster, f"#{b.source_cluster}")
            ta = hub_of.get(b.target_cluster, f"#{b.target_cluster}")
            add(f"| [[{_short(b.source)}]] | [[{_short(b.target)}]] | {sa} ↔ {ta} | {b.weight} |")
    else:
        add("_No cross-cluster bridges found._")
    add("")

    # Structural gaps — the mirror of bridges: areas that should connect but don't
    add("## Structural Gaps (Disconnected Knowledge Areas)")
    add("_Well-formed areas with no links between them — candidate bridges to build._")
    if r.structural_gaps:
        add("| Area A | Area B | Links | Gap score |")
        add("|---|---|---|---|")
        for g in r.structural_gaps:
            a = hub_of.get(g.cluster_a, f"#{g.cluster_a}")
            b = hub_of.get(g.cluster_b, f"#{g.cluster_b}")
            add(f"| {a} | {b} | {g.inter_edges} | {g.gap_score} |")
    else:
        add("_No disconnected areas (or too few clusters to compare)._")
    add("")

    # Clusters (named by hub, singletons collapsed, biggest first)
    add("## Clusters (Knowledge Areas)")
    if linked:
        for c in linked:
            add(f"> [!abstract]- [[{_short(c.hub)}]] — {c.size} notes · cohesion {c.cohesion}")
            member_links = ", ".join(f"[[{_short(m)}]]" for m in c.members[:_MEMBERS_CAP])
            if len(c.members) > _MEMBERS_CAP:
                member_links += f" … (+{len(c.members) - _MEMBERS_CAP} more)"
            add(f"> {member_links}")
            add("")
    else:
        add("_No linked clusters detected (vault has no resolved wikilinks)._")
        add("")
    if singletons:
        add(f"_Plus {singletons} standalone notes with no internal links (full list in GRAPH_REPORT.json)._")
        add("")

    # Orphans
    add("## Orphans (No Incoming Links)")
    if r.orphans:
        _fold(add, "warning", f"{len(r.orphans)} orphans", r.orphans,
              lambda o: f"[[{_short(o)}]]")
    else:
        add("_No orphans._")
        add("")

    # Dangling links — targets are unresolved (inline code, not wikilinks)
    add("## Dangling Links (Unresolved Wikilinks)")
    if r.dangling:
        _fold(add, "bug", f"{len(r.dangling)} broken links", r.dangling,
              lambda d: f"`{d['target']}` — {d['refs']}×", more_json=False)
    else:
        add("_No unresolved wikilinks._")
        add("")

    # Contested claims — authoritative, kept visible until a human resolves them
    if r.contested:
        add("## Contested Claims (Unresolved Contradictions)")
        _fold(add, "danger", f"{len(r.contested)} contested", r.contested,
              lambda c: f"[[{_short(c.path)}]] ↮ {'; '.join(c.refs) if c.refs else '—'}")

    # Source drift — authoritative, from <vault>/provenance.json: notes still
    # carrying claims from a source version that has since been re-ingested
    if r.source_drift:
        add("## Source Drift (Notes From a Superseded Source Version)")
        _fold(add, "warning", f"{len(r.source_drift)} drifted notes", r.source_drift,
              lambda d: f"[[{_short(d.note)}]] — derived from a superseded version of {d.source}")

    # Missing links (proposed)
    if r.missing_links:
        add("## Proposed Missing Links _(embedding candidates — not authoritative)_")
        add("| Source | Target | Cosine | d_prev | Novelty |")
        add("|---|---|---|---|---|")
        for ml in r.missing_links[:_LIST_CAP]:
            novelty = "🔴 novel" if ml.d_prev == 0 or ml.d_prev >= 3 else "🟡 likely"
            d_str = str(ml.d_prev) if ml.d_prev > 0 else "∞"
            add(f"| [[{_short(ml.source)}]] | [[{_short(ml.target)}]] | {ml.cosine} | {d_str} | {novelty} |")
        add("")

    # Likely duplicates (≥ τ_high — genuine merge candidates)
    if r.confirmed_duplicate_pairs:
        add(f"### Likely Duplicates ({len(r.confirmed_duplicate_pairs)}) _(≥ τ_high — review for merge)_")
        _fold(add, "warning", "Merge candidates", r.confirmed_duplicate_pairs,
              lambda dp: f"[[{_short(dp.source)}]] vs [[{_short(dp.target)}]] (score {dp.score:.3f})")

    # Borderline-related pairs (τ_low..τ_high — topically close, link not merge)
    if r.duplicate_pairs:
        add(f"### Related Pairs ({len(r.duplicate_pairs)}) _(borderline similarity — link, don't merge)_")
        _fold(add, "note", "Borderline pairs", r.duplicate_pairs,
              lambda dp: f"[[{_short(dp.source)}]] vs [[{_short(dp.target)}]] (score {dp.score:.3f})")

    # Co-occurrence delta (proposed, embedder-free)
    if r.autolink_candidates:
        lines.append("\n## Autolink Candidates _(co-occurrence − wikilink — not authoritative)_")
        lines.append("| Source | Target | Via | Weight | Hubs | Shared Concepts |")
        lines.append("|---|---|---|---|---|---|")
        for a in r.autolink_candidates:
            shared = ", ".join(a.shared) if a.shared else "_(associative)_"
            lines.append(f"| [[{_short(a.source)}]] | [[{_short(a.target)}]] | {a.provenance} | {a.weight} | {a.convergence} | {shared} |")

    if r.stale_links:
        add("\n## Stale Links _(wikilink − co-occurrence — review)_")
        _fold(add, "note", f"{len(r.stale_links)} stale links", r.stale_links,
              lambda s: f"[[{_short(s.source)}]] ↔ [[{_short(s.target)}]] _(linked, no shared concepts)_",
              more_json=False)

    if r.missing_hubs:
        lines.append("\n## Missing Hubs _(central concepts with no hub note)_")
        lines.append("| Concept | Centrality |")
        lines.append("|---|---|")
        for h in r.missing_hubs:
            lines.append(f"| {h.concept} | {h.centrality} |")

    if r.code_coverage:
        cc = r.code_coverage
        add("\n## Code Coverage _(codegraph — supported files documented by a note)_")
        add(f"**{cc.documented}/{cc.total}** supported source files are documented.")
        if cc.undocumented:
            _fold(add, "todo", f"{len(cc.undocumented)} undocumented files (by fan-in)",
                  cc.undocumented, lambda u: f"`{u[0]}` — {u[1]} importer(s)")
        else:
            add("")

    if r.attention_candidates:
        add("\n## Attention Candidates _(idle × weakly-linked — not authoritative)_")
        lines.append("| Note | Idle (days) | Links | Score |")
        lines.append("|---|---|---|---|")
        for ac in r.attention_candidates:
            lines.append(f"| [[{_short(ac.path)}]] | {ac.days_idle} | {ac.degree} | {ac.score} |")

    if r.lean_notes:
        add(f"\n### Lean Notes (Enrichment Candidates) ({len(r.lean_notes)})")
        _fold(add, "todo", "Enrichment candidates", r.lean_notes,
              lambda n: f"[[{_short(n)}]]")

    if r.reformat_notes:
        add(f"\n### Reformat Notes (Stylistic Refinement) ({len(r.reformat_notes)})")
        _fold(add, "todo", "Stylistic refinements", r.reformat_notes,
              lambda n: f"[[{_short(n)}]]")

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

    if report.contested:
        con = ", ".join(
            c.path.rsplit("/", 1)[-1].removesuffix(".md")
            for c in report.contested[:max_items]
        )
        lines.append(f"CONTESTED  {con}")

    if report.source_drift:
        sd = ", ".join(
            f"{d.note.rsplit('/', 1)[-1].removesuffix('.md')}←{d.source}"
            for d in report.source_drift[:max_items]
        )
        lines.append(f"SOURCE DRIFT  {sd}")

    if report.attention_candidates:
        att = ", ".join(
            f"{a.path.rsplit('/',1)[-1].removesuffix('.md')}(idle={a.days_idle}d,deg={a.degree})"
            for a in report.attention_candidates[:max_items]
        )
        lines.append(f"ATTENTION  {att}")

    if report.clusters:
        clist = ", ".join(
            f"C{c.cluster_id}(n={c.size},hub={c.hub.rsplit('/',1)[-1].removesuffix('.md') if c.hub else '-'})"
            for c in report.clusters[:max_items]
        )
        lines.append(f"CLUSTERS  {clist}")

    if report.missing_links:
        ml = ", ".join(
            f"{m.source.rsplit('/',1)[-1].removesuffix('.md')}→{m.target.rsplit('/',1)[-1].removesuffix('.md')}(cos={m.cosine},d={m.d_prev})"
            for m in report.missing_links[:max_items]
        )
        lines.append(f"PROPOSED  {ml}")

    if report.confirmed_duplicate_pairs:
        cd_list = ", ".join(
            f"{dp.source.rsplit('/',1)[-1].removesuffix('.md')}↔{dp.target.rsplit('/',1)[-1].removesuffix('.md')}(cos={dp.score})"
            for dp in report.confirmed_duplicate_pairs[:max_items]
        )
        lines.append(f"DUPS  {cd_list}")

    if report.duplicate_pairs:
        dp_list = ", ".join(
            f"{dp.source.rsplit('/',1)[-1].removesuffix('.md')}↔{dp.target.rsplit('/',1)[-1].removesuffix('.md')}(cos={dp.score})"
            for dp in report.duplicate_pairs[:max_items]
        )
        lines.append(f"RELATED  {dp_list}")

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
