"""Curator dispatch — /curate: vault maintenance as a background policy.

The pure composer (silica.kernel.curator) turns an L1 VaultReport into a typed
CurationPlan. This module executes that plan on the *existing* machinery:

  * strong autolink candidate → the mechanical, LLM-free silica_autolink path
    (graph-safe direct commit);
  * orphan / dedup / refine    → WorkItems drained through run_subagent_batch,
    the same leashed-sub-agent seam /dedup and /refine already use — so every
    write goes through commit_ops + bounds + the snapshot/rollback undo path.

The curator gains no new power: only initiative. `silica_curate` defaults to a
dry-run (compose + return the plan, enqueue and write nothing); `apply=True`
routes through `apply_curation_plan`, which also appends one idempotent journal
line via run_log.
"""
from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from silica.agent.subagent import run_subagent_batch
from silica.kernel.curator import CurationPlan, compose_curation_plan
from silica.kernel.graph_report import compute_report
from silica.kernel.run_log import append_log_line, format_curate_event
from silica.kernel.workqueue import WorkItem
from silica.tools import tool


# ---------------------------------------------------------------------------
# I/O helpers (patched wholesale in tests so no driver/index/LLM is touched)
# ---------------------------------------------------------------------------

def _read_body(path: str) -> str:
    """Note body, or "" on any error (a missing note simply yields no excerpt)."""
    try:
        from silica.driver import DRIVER

        return DRIVER.read_note(path).content or ""
    except Exception:
        return ""


def _orphan_candidates(path: str, k: int = 5) -> list[dict]:
    """Offer link candidates for an orphan via the relatedness facade.

    Mirrors Coordinator._orphan_candidates: fuses the embedding + co-occurrence
    legs so a candidate survives when either leg is down. The orphan worker only
    links among offered candidates, so an empty list makes it a safe no-op.
    """
    try:
        from silica.agent.bounds import _norm_path
        from silica.config import CONFIG
        from silica.kernel.cooccurrence import get_cooccur_store
        from silica.kernel.embed import get_store
        from silica.kernel.relatedness import related_notes

        from silica.agent.providers import get_reranker
        from silica.kernel.rerank import note_document, rerank_related

        key = _norm_path(path)
        reranker = get_reranker(CONFIG)
        pool = max(k, 20) if reranker else k
        results = related_notes(
            key,
            embed_store=get_store(),
            cooccur_store=get_cooccur_store(lang=CONFIG.cooccurrence_lang),
            k=pool,
        )
        if reranker:
            results = rerank_related(reranker, note_document(key), results, k=k)
        return [{"name": r.name, "path": r.path} for r in results]
    except Exception:
        return []


def _run_autolink(sources: list[str]) -> dict[str, Any]:
    """Mechanical, LLM-free autolink of the given source notes (direct commit)."""
    from silica.tools.graph import silica_autolink

    return silica_autolink(note_paths=list(dict.fromkeys(sources)))


# ---------------------------------------------------------------------------
# plan → WorkItems
# ---------------------------------------------------------------------------

def _orphan_workitems(plan: CurationPlan) -> list[WorkItem]:
    items: list[WorkItem] = []
    for it in plan.by_kind("orphan"):
        items.append(WorkItem(
            kind="orphan",
            target_path=it.target,
            context={"candidates": _orphan_candidates(it.target)},
            reason=it.reason or "curate orphan",
        ))
    return items


def _dedup_workitems(plan: CurationPlan) -> list[WorkItem]:
    """Turn dedup pairs into merge WorkItems, collapsing duplicate *families*.

    Pairwise dedup leaves one survivor per local top-1 hub: a family {A,B,C,D,E}
    whose top-1 edges are A→B, C→B, D→E, F→E collapses to TWO notes, not one.
    So confirmed pairs (score ≥ τ_high) are union-found into connected components
    and each component funnels into its single largest note.

    Transitive closure is applied ONLY to confirmed pairs — chaining borderline
    (< τ_high) links would merge distant notes — so borderline pairs stay per-pair.
    Safety: every item still passes the ternary judge, and curate items carry no
    `target_dir`, so a "distinct" verdict is a no-op. Union-find proposes the
    merge target; the judge disposes. A false union costs a judge call, never a bad
    merge.
    """
    from silica.config import CONFIG

    tau_high = getattr(CONFIG, "sim_threshold_high", 0.85)
    pairs = list(plan.by_kind("dedup"))

    _body_cache: dict[str, str] = {}
    def body(p: str) -> str:
        if p not in _body_cache:
            _body_cache[p] = _read_body(p)
        return _body_cache[p]
    stem = lambda p: p.removesuffix(".md").rsplit("/", 1)[-1]

    # Union-find over confirmed pairs only.
    parent: dict[str, str] = {}
    node_score: dict[str, float] = {}
    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for it in pairs:
        if it.score >= tau_high:
            union(it.target, it.partner)
            for n in (it.target, it.partner):
                node_score[n] = max(node_score.get(n, 0.0), it.score)

    components: dict[str, set[str]] = {}
    for node in list(parent):
        components.setdefault(find(node), set()).add(node)

    items: list[WorkItem] = []
    # 1. Each confirmed component → collapse every member into its largest note.
    for members in components.values():
        canonical = max(members, key=lambda p: len(body(p)))
        for m in members:
            if m == canonical:
                continue
            sc = node_score.get(m, tau_high)
            items.append(WorkItem(
                kind="dedup",
                target_path=canonical,
                context={
                    "concept": stem(m),
                    "excerpt": body(m)[:4000],
                    "candidate": stem(canonical),
                    "score": sc,
                    "inbox_file": m,
                },
                reason=f"curate dedup family → {stem(canonical)} (score={sc:.3f})",
            ))

    # 2. Borderline pairs — historical per-pair behaviour (larger note is target),
    #    skipping any pair already absorbed by a shared confirmed component.
    for it in pairs:
        if it.score >= tau_high:
            continue
        source, target = it.target, it.partner
        if source in parent and target in parent and find(source) == find(target):
            continue
        body_s, body_t = body(source), body(target)
        if len(body_t) >= len(body_s):
            larger, smaller, smaller_body = target, source, body_s
        else:
            larger, smaller, smaller_body = source, target, body_t
        items.append(WorkItem(
            kind="dedup",
            target_path=larger,
            context={
                "concept": stem(smaller),
                "excerpt": smaller_body[:4000],
                "candidate": stem(larger),
                "score": it.score,
                "inbox_file": smaller,
            },
            reason=it.reason or f"curate dedup score={it.score:.3f}",
        ))
    return items


def _refine_workitems(plan: CurationPlan) -> list[WorkItem]:
    return [
        WorkItem(kind="refine", target_path=it.target, context={}, reason=it.reason or "curate refine")
        for it in plan.by_kind("refine")
    ]


def _execution_outcome_counts(
    autolink_result: dict[str, Any], batch: dict[str, Any]
) -> dict[str, int]:
    """Real per-item outcome counts for an apply run — NOT plan.counts().

    `batch["summary"]` (run_subagent_batch -> WorkQueue.summary()) is already
    a Counter of each dispatched WorkItem's REAL terminal status (e.g.
    "committed" via commit.py, "no_merge" for a dedup verdict of distinct,
    "no_link" when the orphan worker found nothing worth linking, "no_change",
    "failed", "cancelled") — a batch where every dedup came back distinct
    must show up as "no_merge", not as though the planned dedup succeeded.
    The mechanical autolink direct-commit isn't a WorkItem, so its real
    outcome (links actually added, from silica_autolink's own return value —
    not the candidate-pair count the plan carried) is folded in separately.
    silica_autolink returns {"notes_processed", "total_links_added"} (see
    silica/tools/graph.py); "added" is silica_backlink's key, not autolink's.
    """
    counts: dict[str, int] = dict(batch.get("summary", {}))
    added = (autolink_result or {}).get("total_links_added", 0)
    if added:
        counts["autolink"] = added
    return counts


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def apply_curation_plan(
    plan: CurationPlan,
    *,
    config: Any = None,
    run_id: str | None = None,
    vault_path: str | None = None,
    cancel_token: Any = None,
) -> dict[str, Any]:
    """Execute a CurationPlan on the existing seam, then journal it once.

    Fires the mechanical autolink direct-commit, enqueues orphan/dedup/refine
    WorkItems through run_subagent_batch (commit_ops + bounds + rollback), and
    appends one idempotent journal line. An empty plan is a no-op — nothing is
    enqueued, written, or journalled.
    """
    if plan.is_empty():
        return {"status": "nothing_to_do", "counts": {}}

    if config is None:
        from silica.config import CONFIG

        config = CONFIG
    run_id = run_id or uuid.uuid4().hex

    # 1. Mechanical autolink — LLM-free, graph-safe, reversible direct commit.
    autolink_sources = [it.target for it in plan.by_kind("autolink")]
    autolink_result = _run_autolink(autolink_sources) if autolink_sources else {}

    # 2. Orphan / dedup / refine → WorkItems on the leashed-sub-agent seam.
    work: list[WorkItem] = (
        _orphan_workitems(plan) + _dedup_workitems(plan) + _refine_workitems(plan)
    )
    batch = (
        run_subagent_batch(work, config, cancel_token=cancel_token)
        if work
        else {"items": 0, "summary": {}, "results": []}
    )

    # 3. Human journal — one line per run, deduped so a resume never doubles
    #    it. Reports the REAL outcome (what run_subagent_batch's per-item
    #    statuses and the autolink direct-commit actually did), not the
    #    planned item counts — those live in `counts` below for callers that
    #    want the plan shape, but "Applied" must mean applied.
    counts = plan.counts()
    outcome_counts = _execution_outcome_counts(autolink_result, batch)
    append_log_line(
        format_curate_event(outcome_counts),
        run_id,
        vault_path=vault_path,
        dedup_key="curate",
    )

    return {
        "status": "applied",
        "run_id": run_id,
        "counts": counts,
        "outcome_counts": outcome_counts,
        "autolink": autolink_result,
        "batch": batch,
    }


# ---------------------------------------------------------------------------
# tool
# ---------------------------------------------------------------------------

class CurateArgs(BaseModel):
    apply: bool = Field(
        default=False,
        description="If True, enqueue/execute the plan; default is a dry-run that only returns the plan.",
    )
    folder: str = Field(default="", description="Vault-relative folder to scope the audit (empty = whole vault)")


@tool(CurateArgs, cls="composed")
def silica_curate(apply: bool = False, folder: str = "", cancel_token: Any = None) -> dict[str, Any]:
    """Curate the vault: turn structural findings into executed maintenance work.

    Composes a plan from the vault report (strong autolinks, orphans to link,
    near-duplicate pairs, oversized/lean notes). Default is a dry-run: returns
    the plan, writes nothing. With apply=True it executes the plan (autolinks,
    orphan linking, dedup merges, refinements) with undo journaling.
    For the raw audit report without acting on it, use silica_vault_report.
    """
    report = compute_report(
        folder=folder,
        analytics=True,          # lean_notes / reformat_notes triage
        with_embeddings=True,    # duplicate pairs
        with_cooccurrence=True,  # autolink candidates
    )
    plan = compose_curation_plan(report)

    result: dict[str, Any] = {
        "apply": apply,
        "total": len(plan),
        "counts": plan.counts(),
        "items": [
            {"kind": i.kind, "target": i.target, "partner": i.partner, "reason": i.reason}
            for i in plan.items
        ],
    }

    if plan.is_empty():
        result["status"] = "nothing_to_do"
        return result

    if not apply:
        result["status"] = "dry_run"
        return result

    from silica.config import CONFIG

    execution = apply_curation_plan(
        plan,
        config=CONFIG,
        vault_path=getattr(CONFIG, "vault_path", None),
        cancel_token=cancel_token,
    )
    result["status"] = execution.get("status", "applied")
    result["execution"] = execution
    return result
