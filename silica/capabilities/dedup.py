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

    if decision.verdict == "distinct":
        return _route_distinct(item, decision, config)

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


def _route_distinct(item: WorkItem, decision: DedupDecision, config: Any) -> dict[str, Any]:
    """Distinct verdict routing (C2): the borderline concept becomes a spoke.

    Pipeline items (context carries ``target_dir``) commit the spoke the judge
    authored in the verdict call — or, when authoring failed, a mechanical
    write of the excerpt verbatim with provenance, refined right after
    (ADR-0001: mechanical inject + deferred refine). The parked twin bundle is
    cleaned only on verified commit, so the op degrades but is never lost.

    Ad-hoc pairs (two existing notes, no ``target_dir``) keep the historical
    contract: distinct → no write.
    """
    ctx = item.context
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

    # Build the score block shown to the model.
    # When both metrics are available we surface them separately so the model
    # can interpret the signal correctly: a high title score with a low body
    # score means "topically related but distinct" — very different from a
    # uniformly high score which strongly suggests a true duplicate.
    if title_score > 0.0 and full_score > 0.0:
        score_block = (
            f"SEMANTIC CLOSENESS SCORE: {score:.3f} (effective = max of the two below)\n"
            f"  • Full-note similarity (body + title):  {full_score:.3f}\n"
            f"  • Title-only similarity:                {title_score:.3f}\n"
            f"Interpretation:\n"
            f"  - High full-note score (>0.80): bodies cover the same topic → likely duplicate.\n"
            f"  - High title score with low body score: notes are topically related but\n"
            f"    cover distinct aspects (e.g. 'ROS' vs 'JSON in ROS 2') → prefer linking\n"
            f"    over merging; set is_duplicate=false unless content genuinely overlaps."
        )
    else:
        score_block = (
            f"SEMANTIC CLOSENESS SCORE: {score:.3f} (0.0 to 1.0, where 1.0 is identical)\n"
            f"Use this metric as an indicator. High scores (>0.85) strongly suggest "
            f"duplicates, while lower scores might represent related but distinct topics."
        )

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
