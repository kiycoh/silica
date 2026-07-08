# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Expand capability — author the body for a «snippet too short» rejection.

The MIN_WRITE_SNIPPET_CHARS gate rejects write ops whose body the distiller
omitted (real incident: whole chunks with snippet="" despite full inbox
excerpts). Re-validation alone can never clear them, so this worker re-prompts
the LLM with the concept's inbox excerpt — max MAX_EXPAND_ATTEMPTS tries, the
second carrying corrective feedback — and commits through the same gate. After
that the op stays in the deferred bundle, final. Enqueued only for PARTIAL
chunk rejections: an all-rejected chunk already re-delegates via the steer arc.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from silica.agent.commit import commit_ops
from silica.agent.bounds import expand_bounds
from silica.kernel.ops import Op, OpType
from silica.kernel.validate import MIN_WRITE_SNIPPET_CHARS
from silica.kernel.workqueue import WorkItem
from silica.capabilities._base import NoteContent, emit_feedback, load_prompt

logger = logging.getLogger(__name__)

MAX_EXPAND_ATTEMPTS = 2


def run_expand(item: WorkItem, config: Any) -> dict[str, Any]:
    ctx = item.context
    op_raw: dict = ctx.get("op", {}) or {}
    heading = op_raw.get("heading", "")
    excerpt = (ctx.get("excerpt") or "").strip()
    if not excerpt:
        return {"status": "skipped", "reason": "no inbox excerpt to expand from"}

    hub = ctx.get("hub")
    source_basename = op_raw.get("source_basename") or os.path.basename(
        ctx.get("inbox_file", "") or ""
    )

    body = ""
    feedback = ""
    for attempt in range(1, MAX_EXPAND_ATTEMPTS + 1):
        if item.cancel_token.is_set():
            return {"status": "cancelled"}
        emit_feedback(item, "calling_llm", f"attempt {attempt}/{MAX_EXPAND_ATTEMPTS}")
        body = _author_body(
            config,
            concept=heading,
            excerpt=excerpt,
            source_basename=source_basename,
            hub=hub,
            feedback=feedback,
        ).strip()
        if len(body) >= MIN_WRITE_SNIPPET_CHARS:
            break
        feedback = (
            f"Your previous body was too short ({len(body)} chars; minimum "
            f"{MIN_WRITE_SNIPPET_CHARS}). Write a complete body grounded in the excerpt."
        )
    else:
        return {
            "status": "still_short",
            "attempts": MAX_EXPAND_ATTEMPTS,
            "reason": f"body still under {MIN_WRITE_SNIPPET_CHARS} chars — op stays deferred",
        }

    if item.cancel_token.is_set():
        return {"status": "cancelled"}

    emit_feedback(item, "committing")
    op = Op(
        op=OpType.write,
        heading=heading,
        source_basename=source_basename or "expand",
        path=item.target_path,
        title=op_raw.get("title") or None,
        snippet=body,
        hub=hub,
        tags=op_raw.get("tags") or None,
        related=op_raw.get("related") or None,
        parent=op_raw.get("parent") or None,
        reason=f"expand retry: {ctx.get('reason', '')[:120]}",
    )
    result = commit_ops(
        [op],
        target_dir=ctx.get("target_dir", "") or os.path.dirname(item.target_path),
        hub=hub,
        bounds=expand_bounds(item.target_path, hub=hub),
    )
    if result.get("status") == "committed":
        _clean_twin_bundle(ctx, heading)
    return result


def _clean_twin_bundle(ctx: dict, heading: str) -> None:
    """Drop the expanded op from the deferred bundle VALIDATE parked (same
    contract as dedup: only after a verified commit; best-effort)."""
    content_hash = ctx.get("content_hash", "")
    if not content_hash:
        return
    try:
        from silica.kernel.deferred import get_deferred_store
        get_deferred_store().remove_op(content_hash, heading)
    except Exception as e:
        logger.debug("expand: twin bundle cleanup failed (non-fatal): %s", e)


def _author_body(
    config: Any,
    *,
    concept: str,
    excerpt: str,
    source_basename: str,
    hub: str | None = None,
    feedback: str = "",
) -> str:
    from silica.agent.providers import get_provider
    from silica.kernel.sanitize import parse_json

    prompt = load_prompt("expand_prompt.txt")
    hub_hint = f"\nParent note: [[{hub}]]" if hub else ""
    feedback_block = f"\n\nCORRECTION: {feedback}" if feedback else ""
    user_message = (
        f"{prompt}\n\n"
        f"---\nCONCEPT: {concept}{hub_hint}\n"
        f"SOURCE ({source_basename}) EXCERPT:\n{excerpt}\n"
        f"{feedback_block}"
    )
    provider = get_provider(config, role="worker")
    response = provider.call_llm(
        messages=[{"role": "user", "content": user_message}],
        tools=None,
        response_schema=NoteContent,
        max_tokens=int(os.getenv("EXPAND_MAX_TOKENS", "4096")),
    )
    raw = response.text or ""
    try:
        parsed, _ = parse_json(raw, strict=False)
        if isinstance(parsed, dict) and "content" in parsed:
            return str(parsed["content"])
    except Exception as e:
        logger.debug("expand parse failed: %s", e)
    return ""
