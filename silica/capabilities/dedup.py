"""Dedup capability — merge a borderline-duplicate concept into an existing note.

Given an incoming concept vs. an existing larger note, decide whether they are
the same concept and, if so, append only the genuinely-new information into the
existing note as a single ``patch`` under the dedup leash.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from pydantic import BaseModel

from silica.agent.commit import commit_ops
from silica.agent.leash import dedup_leash
from silica.kernel.ops import Op, OpType
from silica.planner.workqueue import WorkItem
from silica.capabilities._base import emit_feedback, load_prompt, read_or_skip

logger = logging.getLogger(__name__)


class DedupDecision(BaseModel):
    is_duplicate: bool
    rationale: str = ""
    addition: str = ""


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
    )

    if not decision.is_duplicate or not decision.addition.strip():
        return {
            "status": "no_merge",
            "is_duplicate": decision.is_duplicate,
            "rationale": decision.rationale,
        }

    if item.cancel_token.is_set():
        return {"status": "cancelled"}

    emit_feedback(item, "committing")
    hub = ctx.get("hub")
    inbox_file = ctx.get("inbox_file", "")
    op = Op(
        op=OpType.patch,
        heading=ctx.get("concept", "") or "merged concept",
        source_basename=os.path.basename(inbox_file) if inbox_file else "dedup",
        path=candidate_path,
        snippet=decision.addition,
        hub=hub,
        reason=f"dedup merge: {decision.rationale[:120]}",
    )
    leash = dedup_leash(candidate_path, hub=hub)
    result = commit_ops(
        [op],
        target_dir=os.path.dirname(candidate_path),
        hub=hub,
        leash=leash,
    )
    result.setdefault("rationale", decision.rationale)
    return result


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
) -> DedupDecision:
    from silica.agent.providers import get_provider
    from silica.kernel.sanitize import parse_json

    prompt = load_prompt("dedup_prompt.txt")

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
        max_tokens=1024,
    )
    raw = response.text or ""
    try:
        parsed, _ = parse_json(raw, strict=False)
        if isinstance(parsed, dict):
            return DedupDecision(**{
                "is_duplicate": bool(parsed.get("is_duplicate", False)),
                "rationale": str(parsed.get("rationale", "")),
                "addition": str(parsed.get("addition", "")),
            })
    except Exception as e:
        logger.debug("dedup decision parse failed: %s", e)
    # Conservative default: when in doubt, do not merge.
    return DedupDecision(is_duplicate=False, rationale="unparseable decision")
