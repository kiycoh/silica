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


def to_markdown(r: VaultReport, title: str = "Silica Vault Report") -> str:
    """Render a VaultReport as OFM-friendly markdown."""
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append(f"_Generated: {r.generated_at}_")
    if r.scope:
        lines.append(f"_Scope: `{r.scope}`_")
    lines.append("")

    # Totals
    lines.append("## Totals")
    lines.append("| Metric | Count |")
    lines.append("|---|---|")
    for k, v in r.totals.items():
        lines.append(f"| {k.capitalize()} | {v} |")
    lines.append("")

    # God nodes
    lines.append("## God Nodes (High-Degree Hubs)")
    if r.god_nodes:
        lines.append("| Note | Cluster | Degree | In | Out | PageRank |")
        lines.append("|---|---|---|---|---|---|")
        for n in r.god_nodes:
            lines.append(f"| [[{n.label}]] | {n.cluster} | {n.degree} | {n.in_degree} | {n.out_degree} | {n.pagerank} |")
    else:
        lines.append("_No connected notes found._")
    lines.append("")

    # Surprising bridges
    lines.append("## Surprising Cross-Cluster Connections")
    if r.bridges:
        lines.append("| Source | Target | Clusters | Surprise |")
        lines.append("|---|---|---|---|")
        for b in r.bridges:
            src_label = b.source.rsplit("/", 1)[-1].removesuffix(".md")
            tgt_label = b.target.rsplit("/", 1)[-1].removesuffix(".md")
            lines.append(f"| [[{src_label}]] | [[{tgt_label}]] | {b.source_cluster}↔{b.target_cluster} | {b.weight} |")
    else:
        lines.append("_No cross-cluster bridges found._")
    lines.append("")

    # Clusters
    lines.append("## Clusters")
    if r.clusters:
        for c in r.clusters:
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
    if r.orphans:
        for o in r.orphans:
            label = o.rsplit("/", 1)[-1].removesuffix(".md")
            lines.append(f"- [[{label}]]")
    else:
        lines.append("_No orphans._")
    lines.append("")

    # Dangling links
    lines.append("## Dangling Links (Unresolved Wikilinks)")
    if r.dangling:
        lines.append("| Target | References |")
        lines.append("|---|---|")
        for d in r.dangling:
            lines.append(f"| `{d['target']}` | {d['refs']} |")
    else:
        lines.append("_No unresolved wikilinks._")
    lines.append("")

    # Missing links (proposed)
    if r.missing_links:
        lines.append("## Proposed Missing Links _(embedding candidates — not authoritative)_")
        lines.append("| Source | Target | Cosine | d_prev | Novelty |")
        lines.append("|---|---|---|---|---|")
        for ml in r.missing_links:
            src_label = ml.source.rsplit("/", 1)[-1].removesuffix(".md")
            tgt_label = ml.target.rsplit("/", 1)[-1].removesuffix(".md")
            novelty = "🔴 novel" if ml.d_prev == 0 or ml.d_prev >= 3 else "🟡 likely"
            d_str = str(ml.d_prev) if ml.d_prev > 0 else "∞"
            lines.append(f"| [[{src_label}]] | [[{tgt_label}]] | {ml.cosine} | {d_str} | {novelty} |")
        lines.append("")

    # Duplicate pairs (proposed)
    if r.duplicate_pairs:
        lines.append(f"\n### Borderline Duplicates ({len(r.duplicate_pairs)})")
        for dp in r.duplicate_pairs:
            lines.append(f"- [[{dp.source}]] vs [[{dp.target}]] (score: {dp.score:.3f})")

    # Co-occurrence delta (proposed, embedder-free)
    def _short(p: str) -> str:
        return p.rsplit("/", 1)[-1].removesuffix(".md")

    if r.autolink_candidates:
        lines.append("\n## Autolink Candidates _(co-occurrence − wikilink — not authoritative)_")
        lines.append("| Source | Target | Weight | Hubs | Shared Concepts |")
        lines.append("|---|---|---|---|---|")
        for a in r.autolink_candidates:
            shared = ", ".join(a.shared) if a.shared else "_(associative)_"
            lines.append(f"| [[{_short(a.source)}]] | [[{_short(a.target)}]] | {a.weight} | {a.convergence} | {shared} |")

    if r.stale_links:
        lines.append("\n## Stale Links _(wikilink − co-occurrence — review)_")
        for s in r.stale_links:
            lines.append(f"- [[{_short(s.source)}]] ↔ [[{_short(s.target)}]] _(linked, no shared concepts)_")

    if r.missing_hubs:
        lines.append("\n## Missing Hubs _(central concepts with no hub note)_")
        lines.append("| Concept | Centrality |")
        lines.append("|---|---|")
        for h in r.missing_hubs:
            lines.append(f"| {h.concept} | {h.centrality} |")

    if r.lean_notes:
        lines.append(f"\n### Lean Notes (Enrichment Candidates) ({len(r.lean_notes)})")
        for n in r.lean_notes:
            lines.append(f"- [[{n}]]")

    if r.reformat_notes:
        lines.append(f"\n### Reformat Notes (Stylistic Refinement) ({len(r.reformat_notes)})")
        for n in r.reformat_notes:
            lines.append(f"- [[{n}]]")

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
            f"{m.source.rsplit('/',1)[-1].removesuffix('.md')}→{m.target.rsplit('/',1)[-1].removesuffix('.md')}(cos={m.cosine},d={m.d_prev})"
            for m in report.missing_links[:max_items]
        )
        lines.append(f"PROPOSED  {ml}")

    if report.duplicate_pairs:
        dp_list = ", ".join(
            f"{dp.source.rsplit('/',1)[-1].removesuffix('.md')}↔{dp.target.rsplit('/',1)[-1].removesuffix('.md')}(cos={dp.score})"
            for dp in report.duplicate_pairs[:max_items]
        )
        lines.append(f"DEDUP  {dp_list}")

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
