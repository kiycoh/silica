"""Enrich capability — semantic expansion of a note under the anti-info-loss bounds."""
from __future__ import annotations

import logging
import os
from typing import Any

from silica.agent.commit import commit_ops
from silica.agent.bounds import refiner_bounds
from silica.kernel.ops import Op, OpType
from silica.planner.workqueue import WorkItem
from silica.capabilities._base import NoteContent, emit_feedback, read_or_skip

logger = logging.getLogger(__name__)


def run_enrich(item: WorkItem, config: Any) -> dict[str, Any]:
    target_path = item.target_path

    emit_feedback(item, "reading")
    original, skip = read_or_skip(target_path)
    if skip is not None:
        return skip

    if item.cancel_token.is_set():
        return {"status": "cancelled"}

    emit_feedback(item, "calling_llm")
    hub = item.context.get("hub") or os.path.splitext(os.path.basename(target_path))[0]
    enriched = _enrich_note(config, target_path, original, hub)
    if not enriched.content.strip():
        return {"status": "no_change", "reason": "enricher produced no content"}

    if item.cancel_token.is_set():
        return {"status": "cancelled"}

    emit_feedback(item, "committing")
    op = Op(
        op=OpType.overwrite,
        heading=os.path.splitext(os.path.basename(target_path))[0],
        source_basename=os.path.basename(target_path),
        path=target_path,
        content=enriched.content,
        hub=hub,
        reason="semantic enrichment",
    )
    # refiner_bounds guarantees anti-info-loss (wikilinks preserved + length floor).
    bounds = refiner_bounds(target_path, hub=hub)
    result = commit_ops(
        [op],
        target_dir=os.path.dirname(target_path),
        hub=hub,
        bounds=bounds,
        read_note=lambda _p: original,
    )
    return result


def _enrich_note(config: Any, target_path: str, original: str, hub: str) -> NoteContent:
    from silica.agent.providers import get_provider
    from silica.kernel.sanitize import parse_json
    from silica.kernel.context_builder import build_context

    system_prompt = (
        "You are an academic assistant expert in writing and structuring notes in Obsidian Flavored Markdown (OFM) in English.\n"
        "Your task is to enrich the note specified by the target.\n"
        "Fundamental rules:\n"
        "1. Produce a rigorous, complete, and exhaustive academic text in English.\n"
        "2. Preserve all factual information and concepts already present in the note (anti-deletion policy). Do not remove pre-existing information, but expand upon it.\n"
        "3. Perform structuring in Obsidian Flavored Markdown: use callouts (> [!tip], > [!note]), LaTeX equation blocks ($$ ... $$) if appropriate, lists, and bold text.\n"
        f"4. You must include a wikilink [[{hub}]] to the hub/parent note (for example in a final section called '# Relations' or '# Connections').\n"
        "5. Return the result structured in JSON format containing a single key 'content' with the full body of the note (including normalized and updated YAML frontmatter tags, and the enriched body)."
    )

    title = os.path.splitext(os.path.basename(target_path))[0]
    note_payload = f"Title: {title}\nPath: {target_path}\nCurrent content:\n{original}"
    ctx = build_context(checkpoint_id="enrich", payload=note_payload)
    user_message = f"Enrich the following note.\n\n{ctx}"

    provider = get_provider(config, role="worker")
    response = provider.call_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
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
        logger.debug("enrich parse failed: %s", e)
    return NoteContent(content="")
