# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Dedup capability — merge a borderline-duplicate concept into an existing note.

Given an incoming concept vs. an existing larger note, decide whether they are
the same concept and, if so, append only the genuinely-new information into the
existing note as a single ``patch`` under the dedup bounds.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Literal

from pydantic import BaseModel

from silica.agent.commit import commit_ops
from silica.agent.bounds import dedup_bounds, dedup_spoke_bounds
from silica.kernel.ops import Op, OpType
from silica.kernel.workqueue import WorkItem
from silica.capabilities._base import emit_feedback, load_prompt, read_or_skip

logger = logging.getLogger(__name__)


class DedupDecision(BaseModel):
    # duplicate    → append only the genuinely-new info
    # distinct     → pipeline concepts: author the spoke note in the same call
    #                (giudice+autore); ad-hoc pairs: no write
    # contradicts  → record the conflicting claim as a contested patch (never resolve)
    verdict: Literal["duplicate", "distinct", "contradicts"] = "distinct"
    rationale: str = ""
    addition: str = ""
    # Authored spoke (distinct + pipeline item only; empty otherwise).
    title: str = ""
    body: str = ""


class DedupBatchDecision(BaseModel):
    """One verdict per incoming concept, same order as presented."""
    decisions: list[DedupDecision] = []


def passes_dedup_gate(
    score: float,
    incoming_len: int,
    candidate_len: int,
    *,
    threshold: float = 0.85,
    max_ratio: float = 2.0,
) -> bool:
    """Cheap gate before the LLM judge (spec 2.1). True iff the effective cosine
    clears `threshold` AND the two bodies are within `max_ratio` in size. The
    size guard rejects the spoke-in-hub false positive (small spoke, big hub:
    high cosine, huge size gap). Cosine is symmetric, so the spec's mutual
    requirement is the single self-normalized score.
    """
    if score < threshold:
        return False
    small, large = sorted((max(incoming_len, 1), max(candidate_len, 1)))
    return large / small <= max_ratio


def run_dedup(item: WorkItem, config: Any) -> dict[str, Any]:
    ctx = item.context
    candidate_path = item.target_path
    budget = 8000

    emit_feedback(item, "reading")
    candidate_body, skip = read_or_skip(candidate_path)
    if skip is not None:
        return skip

    if item.cancel_token.is_set():
        return {"status": "cancelled"}

    if ctx.get("concepts"):
        return _run_batch(item, ctx["concepts"], candidate_body[:budget], config)

    if os.getenv("SILICA_DEDUP_GATE"):
        eff = max(ctx.get("full_score", ctx.get("score", 0.0)),
                  ctx.get("title_score", 0.0))
        if not passes_dedup_gate(eff, len(ctx.get("excerpt", "")),
                                 len(candidate_body[:budget])):
            gate_decision = DedupDecision(
                verdict="distinct",
                rationale="dedup gate: below threshold or size guard",
            )
            return _route_verdict(item, ctx, gate_decision, config)

    emit_feedback(item, "calling_llm")
    decision = _decide_dedup(
        config,
        concept=ctx.get("concept", ""),
        excerpt=ctx.get("excerpt", ""),
        candidate_name=ctx.get("candidate", candidate_path),
        candidate_body=candidate_body[:budget],
        score=ctx.get("score", 0.0),
        full_score=ctx.get("full_score", ctx.get("score", 0.0)),
        title_score=ctx.get("title_score", 0.0),
        author_spoke=bool(ctx.get("target_dir")),
        hub=ctx.get("hub"),
    )

    if item.cancel_token.is_set():
        return {"status": "cancelled"}

    return _route_verdict(item, ctx, decision, config)


def _run_batch(
    item: WorkItem, concepts: list[dict], candidate_body: str, config: Any
) -> dict[str, Any]:
    """Judge a family of concepts against one candidate in a single LLM call,
    then route every verdict through the exact same code as a single item."""
    ctx = item.context
    emit_feedback(item, "calling_llm")
    if os.getenv("SILICA_DEDUP_GATE"):
        decisions = _gated_batch_decisions(config, item, concepts, candidate_body)
    else:
        decisions = _decide_dedup_batch(
            config,
            concepts=concepts,
            candidate_name=ctx.get("candidate", item.target_path),
            candidate_body=candidate_body,
            author_spoke=bool(ctx.get("target_dir")),
            hub=ctx.get("hub"),
        )
    results: list[dict[str, Any]] = []
    followups: list[dict[str, Any]] = []
    for entry, decision in zip(concepts, decisions):
        if item.cancel_token.is_set():
            return {"status": "cancelled", "results": results}
        sub_ctx = {k: v for k, v in ctx.items() if k != "concepts"} | entry
        res = _route_verdict(item, sub_ctx, decision, config)
        fu = res.pop("followup", None)
        if isinstance(fu, dict):
            followups.append(fu)
        results.append({"concept": entry.get("concept", ""), **res})
    statuses = {r.get("status") for r in results}
    out: dict[str, Any] = {
        "status": statuses.pop() if len(statuses) == 1 else "partial",
        "batch": len(results),
        "results": results,
    }
    if followups:
        out["followups"] = followups
    return out


def _gated_batch_decisions(
    config: Any, item: WorkItem, concepts: list[dict], candidate_body: str
) -> list[DedupDecision]:
    """SILICA_DEDUP_GATE batch path: concepts failing the cheap cosine+size gate
    skip the LLM and are pre-judged 'distinct' (clearly not duplicates); the rest
    are judged in one batch call. Returned full-length and aligned to `concepts`,
    so the caller's zip stays correct and every gated-out concept still routes
    through the normal distinct path (authoring its spoke for pipeline items),
    identical to an LLM 'distinct'.
    """
    ctx = item.context
    to_judge: list[dict] = []
    prejudged: dict[int, DedupDecision] = {}
    for i, c in enumerate(concepts):
        eff = max(c.get("full_score", c.get("score", 0.0)) or 0.0,
                  c.get("title_score", 0.0) or 0.0)
        if passes_dedup_gate(eff, len(c.get("excerpt", "")), len(candidate_body)):
            to_judge.append(c)
        else:
            prejudged[i] = DedupDecision(
                verdict="distinct",
                rationale="dedup gate: below threshold or size guard",
            )
    judged = (
        _decide_dedup_batch(
            config,
            concepts=to_judge,
            candidate_name=ctx.get("candidate", item.target_path),
            candidate_body=candidate_body,
            author_spoke=bool(ctx.get("target_dir")),
            hub=ctx.get("hub"),
        )
        if to_judge else []
    )
    judged_iter = iter(judged)
    return [
        prejudged[i] if i in prejudged else next(judged_iter, DedupDecision())
        for i in range(len(concepts))
    ]


def _route_verdict(
    item: WorkItem, ctx: dict, decision: DedupDecision, config: Any
) -> dict[str, Any]:
    candidate_path = item.target_path

    if decision.verdict == "distinct":
        return _route_distinct(item, ctx, decision, config)

    if not decision.addition.strip():
        return {
            "status": "no_merge",
            "verdict": decision.verdict,
            "rationale": decision.rationale,
        }

    emit_feedback(item, "committing")
    hub = ctx.get("hub")
    inbox_file = ctx.get("inbox_file", "")
    source_basename = os.path.basename(inbox_file) if inbox_file else "dedup"
    if decision.verdict == "contradicts":
        from silica.kernel.contested import contested_callout
        op = Op(
            op=OpType.patch,
            heading=ctx.get("concept", "") or "contested claim",
            source_basename=source_basename,
            path=candidate_path,
            snippet=contested_callout(decision.addition, source_basename),
            hub=hub,
            reason=f"contested: {decision.rationale[:120]}",
            contested_by=f"fonte: {source_basename}",
        )
    else:
        op = Op(
            op=OpType.patch,
            heading=ctx.get("concept", "") or "merged concept",
            source_basename=source_basename,
            path=candidate_path,
            snippet=decision.addition,
            hub=hub,
            reason=f"dedup merge: {decision.rationale[:120]}",
        )
    bounds = dedup_bounds(candidate_path, hub=hub)
    result = commit_ops(
        [op],
        target_dir=os.path.dirname(candidate_path),
        hub=hub,
        bounds=bounds,
    )
    result.setdefault("rationale", decision.rationale)
    result.setdefault("verdict", decision.verdict)
    if result.get("status") == "committed":
        _clean_twin_bundle(ctx)
    return result


def _route_distinct(
    item: WorkItem, ctx: dict, decision: DedupDecision, config: Any
) -> dict[str, Any]:
    """Distinct verdict routing (C2): the borderline concept becomes a spoke.

    Pipeline items (context carries ``target_dir``) commit the spoke the judge
    authored in the verdict call — or, when authoring failed, a mechanical
    write of the excerpt verbatim with provenance, refined right after
    (ADR-0001: mechanical inject + deferred refine). The parked twin bundle is
    cleaned only on verified commit, so the op degrades but is never lost.

    Ad-hoc pairs (two existing notes, no ``target_dir``) keep the historical
    contract: distinct → no write.
    """
    target_dir = ctx.get("target_dir", "")
    no_merge = {"status": "no_merge", "verdict": "distinct", "rationale": decision.rationale}
    if not target_dir:
        return no_merge

    from silica.kernel.templates import slugify

    concept = ctx.get("concept", "")
    candidate_name = ctx.get("candidate", "")
    inbox_file = ctx.get("inbox_file", "")
    source_basename = os.path.basename(inbox_file) if inbox_file else "dedup"
    hub = ctx.get("hub")

    title = decision.title.strip()
    body = decision.body.strip()
    mechanical = not (title and body)
    if mechanical:
        excerpt = (ctx.get("excerpt") or "").strip()
        if not excerpt:
            return no_merge  # nothing to materialize the spoke from
        title = concept or candidate_name
        body = f"{excerpt}\n\n*(da {source_basename})*"
    # The framework, not the model, guarantees the spoke is born linked.
    if candidate_name and f"[[{candidate_name}]]" not in body:
        body += f"\n\nCorrelati: [[{candidate_name}]]"

    emit_feedback(item, "committing")
    spoke_path = f"{target_dir}/{slugify(title) or title}.md"
    op = Op(
        op=OpType.write,
        heading=concept or title,
        source_basename=source_basename,
        path=spoke_path,
        title=title,
        snippet=body,
        hub=hub,
        reason=f"dedup distinct spoke: {decision.rationale[:120]}",
    )
    result = commit_ops(
        [op],
        target_dir=target_dir,
        hub=hub,
        bounds=dedup_spoke_bounds(spoke_path, hub=hub),
    )
    result.setdefault("verdict", "distinct")
    result.setdefault("rationale", decision.rationale)
    result["spoke_path"] = spoke_path
    if result.get("status") == "committed":
        _clean_twin_bundle(ctx)
        if mechanical:
            # ADR-0001: mechanical inject + deferred refine. Capabilities are
            # peers (P9) — dedup proposes the follow-up; the BoundedSubAgent
            # engine dispatches it through the registry.
            result["followup"] = {
                "kind": "refine",
                "target_path": spoke_path,
                "context": {"hub": hub} if hub else {},
            }
    return result


def _clean_twin_bundle(ctx: dict) -> None:
    """Drop this concept's op from the deferred bundle COLLISION parked.

    Called only after a verified commit: the verdict has been routed into the
    vault, so the parked copy is no longer the durable one. Best-effort — a
    missing bundle (retry already flushed it, pre-C2 stub) is not an error.
    """
    content_hash = ctx.get("content_hash", "")
    if not content_hash:
        return
    try:
        from silica.kernel.deferred import get_deferred_store
        get_deferred_store().remove_op(content_hash, ctx.get("concept", ""))
    except Exception as e:
        logger.debug("dedup: twin bundle cleanup failed (non-fatal): %s", e)


def _decide_dedup(
    config: Any,
    *,
    concept: str,
    excerpt: str,
    candidate_name: str,
    candidate_body: str,
    score: float = 0.0,
    full_score: float = 0.0,
    title_score: float = 0.0,
    author_spoke: bool = False,
    hub: str | None = None,
) -> DedupDecision:
    from silica.agent.providers import get_provider
    from silica.kernel.sanitize import parse_json

    prompt = load_prompt("dedup_prompt.txt")
    if author_spoke:
        # Giudice+autore (C2): the same call that judges "distinct" also
        # authors the spoke note — a second pass would just re-read the
        # context this call already has.
        hub_hint = f" and to the parent note [[{hub}]]" if hub else ""
        prompt += (
            "\n\nIf (and only if) your verdict is \"distinct\", ALSO author the new note"
            " for the INCOMING CONCEPT in the same response, adding two more JSON keys:"
            "\n  \"title\" — clean note name (no extension, no quotes)."
            f"\n  \"body\" — well-formed Obsidian Markdown grounded ONLY in the incoming"
            f" excerpt (never invent facts); no top-level heading; include a wikilink"
            f" to [[{candidate_name}]]{hub_hint}."
            "\nFor any other verdict leave \"title\" and \"body\" empty."
        )

    score_block = _score_block(score, full_score, title_score)

    user_message = (
        f"{prompt}\n\n"
        f"---\n{score_block}\n"
        f"---\nCANDIDATE NOTE ({candidate_name}):\n{candidate_body}\n\n"
        f"---\nINCOMING CONCEPT: {concept}\nEXCERPT:\n{excerpt}\n"
    )
    provider = get_provider(config, role="worker")
    response = provider.call_llm(
        messages=[{"role": "user", "content": user_message}],
        tools=None,
        response_schema=DedupDecision,
        max_tokens=int(os.getenv("DEDUP_MAX_TOKENS", "2048")),
    )
    raw = response.text or ""
    try:
        parsed, _ = parse_json(raw, strict=False)
        if isinstance(parsed, dict):
            verdict = parsed.get("verdict")
            if verdict not in ("duplicate", "distinct", "contradicts"):
                # Legacy binary schema, or anything unrecognised → conservative.
                legacy = parsed.get("is_duplicate")
                verdict = "duplicate" if legacy is True else "distinct"
            return DedupDecision(
                verdict=verdict,
                rationale=str(parsed.get("rationale", "")),
                addition=str(parsed.get("addition", "")),
                title=str(parsed.get("title", "") or ""),
                body=str(parsed.get("body", "") or ""),
            )
    except Exception as e:
        logger.debug("dedup decision parse failed: %s", e)
    # Conservative default: when in doubt, do not merge and do not contest.
    return DedupDecision(verdict="distinct", rationale="unparseable decision")


def _score_block(score: float, full_score: float, title_score: float) -> str:
    # When both metrics are available we surface them separately so the model
    # can interpret the signal correctly: a high title score with a low body
    # score means "topically related but distinct" — very different from a
    # uniformly high score which strongly suggests a true duplicate.
    if title_score > 0.0 and full_score > 0.0:
        return (
            f"SEMANTIC CLOSENESS SCORE: {score:.3f} (effective = max of the two below)\n"
            f"  • Full-note similarity (body + title):  {full_score:.3f}\n"
            f"  • Title-only similarity:                {title_score:.3f}\n"
            f"Interpretation:\n"
            f"  - High full-note score (>0.80): bodies cover the same topic → likely duplicate.\n"
            f"  - High title score with low body score: notes are topically related but\n"
            f"    cover distinct aspects (e.g. 'ROS' vs 'JSON in ROS 2') → prefer linking\n"
            f"    over merging; set is_duplicate=false unless content genuinely overlaps."
        )
    return (
        f"SEMANTIC CLOSENESS SCORE: {score:.3f} (0.0 to 1.0, where 1.0 is identical)\n"
        f"Use this metric as an indicator. High scores (>0.85) strongly suggest "
        f"duplicates, while lower scores might represent related but distinct topics."
    )


def _decide_dedup_batch(
    config: Any,
    *,
    concepts: list[dict],
    candidate_name: str,
    candidate_body: str,
    author_spoke: bool = False,
    hub: str | None = None,
) -> list[DedupDecision]:
    """One LLM call judging every concept of a family against the candidate.

    The single-verdict prompt is reused verbatim; batch mode only appends the
    array contract and the numbered concept blocks, so the judging criteria
    cannot drift between the per-item and the family path.
    """
    from silica.agent.providers import get_provider

    prompt = load_prompt("dedup_prompt.txt")
    n = len(concepts)
    batch_note = (
        f"\n\nBATCH MODE: below are {n} INCOMING CONCEPTS, all matched against the SAME"
        " candidate note. Judge each one INDEPENDENTLY — verdicts within a batch may"
        " differ. Respond with JSON: {\"decisions\": [...]} containing EXACTLY"
        f" {n} entries, in the same order as the concepts, each with the single-verdict"
        " schema (verdict, rationale, addition)."
    )
    if author_spoke:
        hub_hint = f" and to the parent note [[{hub}]]" if hub else ""
        batch_note += (
            "\nFor every entry whose verdict is \"distinct\", ALSO author the new note"
            " in that entry, adding \"title\" (clean note name, no extension) and"
            " \"body\" (well-formed Obsidian Markdown grounded ONLY in that concept's"
            " excerpt — never invent facts; no top-level heading; include a wikilink"
            f" to [[{candidate_name}]]{hub_hint}). For any other verdict leave"
            " \"title\" and \"body\" empty."
        )
    blocks = []
    for i, c in enumerate(concepts, 1):
        score = c.get("score") or 0.0
        blocks.append(
            f"---\nINCOMING CONCEPT {i}/{n}: {c.get('concept', '')}\n"
            f"{_score_block(score, c.get('full_score') or score, c.get('title_score') or 0.0)}\n"
            f"EXCERPT:\n{c.get('excerpt', '')}\n"
        )
    user_message = (
        f"{prompt}{batch_note}\n\n"
        f"---\nCANDIDATE NOTE ({candidate_name}):\n{candidate_body}\n\n"
        + "\n".join(blocks)
    )
    provider = get_provider(config, role="worker")
    response = provider.call_llm(
        messages=[{"role": "user", "content": user_message}],
        tools=None,
        response_schema=DedupBatchDecision,
        max_tokens=int(os.getenv("DEDUP_MAX_TOKENS", "2048")) * n,
    )
    return _parse_batch(response.text or "", n)


def _parse_batch(raw: str, n: int) -> list[DedupDecision]:
    """Positional decisions, padded/truncated to exactly n.

    A missing or unparseable entry degrades to the same conservative default
    as the single path (distinct, no authorship → mechanical spoke or
    no_merge downstream) — never to a merge.
    """
    from silica.kernel.sanitize import parse_json

    def fallback() -> DedupDecision:
        return DedupDecision(verdict="distinct", rationale="missing from batch response")

    decisions: list[DedupDecision] = []
    try:
        parsed, _ = parse_json(raw, strict=False)
        entries = parsed.get("decisions") if isinstance(parsed, dict) else parsed
        if isinstance(entries, list):
            for e in entries[:n]:
                if not isinstance(e, dict):
                    decisions.append(fallback())
                    continue
                verdict = e.get("verdict")
                if verdict not in ("duplicate", "distinct", "contradicts"):
                    # Legacy binary schema, or anything unrecognised → conservative.
                    verdict = "duplicate" if e.get("is_duplicate") is True else "distinct"
                decisions.append(DedupDecision(
                    verdict=verdict,
                    rationale=str(e.get("rationale", "")),
                    addition=str(e.get("addition", "")),
                    title=str(e.get("title", "") or ""),
                    body=str(e.get("body", "") or ""),
                ))
    except Exception as e:
        logger.debug("dedup batch parse failed: %s", e)
    while len(decisions) < n:
        decisions.append(fallback())
    return decisions
