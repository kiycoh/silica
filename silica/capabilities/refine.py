"""Refine capability — stylistic rewrite of a note under the anti-info-loss leash."""
from __future__ import annotations

import logging
import os
from typing import Any

from silica.agent.commit import commit_ops
from silica.agent.leash import refiner_leash
from silica.kernel.ops import Op, OpType
from silica.planner.workqueue import WorkItem
from silica.capabilities._base import NoteContent, emit_feedback, load_prompt, read_or_skip

logger = logging.getLogger(__name__)


def run_refine(item: WorkItem, config: Any) -> dict[str, Any]:
    target_path = item.target_path

    emit_feedback(item, "reading")
    original, skip = read_or_skip(target_path)
    if skip is not None:
        return skip

    if not original.strip():
        return {"status": "skipped", "reason": "empty note"}

    if item.cancel_token.is_set():
        return {"status": "cancelled"}

    emit_feedback(item, "calling_llm")
    refined = _refine_note(config, target_path, original)
    if not refined.content.strip():
        return {"status": "no_change", "reason": "refiner produced no content"}

    if item.cancel_token.is_set():
        return {"status": "cancelled"}

    emit_feedback(item, "committing")
    hub = item.context.get("hub")
    op = Op(
        op=OpType.overwrite,
        heading=os.path.splitext(os.path.basename(target_path))[0],
        source_basename=os.path.basename(target_path),
        path=target_path,
        content=refined.content,
        hub=hub,
        reason="stylistic refine",
    )
    # refiner_leash enforces anti-info-loss (wikilinks preserved + length floor).
    leash = refiner_leash(target_path, hub=hub)
    result = commit_ops(
        [op],
        target_dir=os.path.dirname(target_path),
        hub=hub,
        leash=leash,
        read_note=lambda _p: original,
    )
    return result


def _refine_note(config: Any, target_path: str, original: str) -> NoteContent:
    from silica.agent.providers import get_provider
    from silica.kernel.sanitize import parse_json

    prompt = load_prompt("refiner_prompt.txt")
    user_message = f"{prompt}\n\n---\nNOTE ({target_path}):\n{original}\n"
    provider = get_provider(config, role="worker")
    response = provider.call_llm(
        messages=[{"role": "user", "content": user_message}],
        tools=None,
        response_schema=NoteContent,
        max_tokens=8192,
    )
    raw = response.text or ""
    try:
        parsed, _ = parse_json(raw, strict=False)
        if isinstance(parsed, dict) and "content" in parsed:
            return NoteContent(content=str(parsed["content"]))
    except Exception as e:
        logger.debug("refine parse failed: %s", e)
    return NoteContent(content="")
