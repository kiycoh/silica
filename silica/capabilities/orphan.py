# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Orphan-connector capability — link a lonely note to offered candidate notes.

The model may only choose among candidates that were actually offered; an
invented target is filtered out so we never create another dangling link.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from pydantic import BaseModel

from silica.agent.commit import commit_ops
from silica.agent.bounds import orphan_bounds
from silica.kernel.ops import Op, OpType
from silica.kernel.workqueue import WorkItem
from silica.capabilities._base import emit_feedback, load_prompt, read_or_skip

logger = logging.getLogger(__name__)


class OrphanLinkDecision(BaseModel):
    links: list[str] = []
    rationale: str = ""


def run_orphan(item: WorkItem, config: Any) -> dict[str, Any]:
    target = item.target_path
    candidates = item.context.get("candidates", [])  # [{"name":..., "path":...}]
    if not candidates:
        return {"status": "no_candidates"}

    emit_feedback(item, "reading")
    body, skip = read_or_skip(target)
    if skip is not None:
        return skip

    if item.cancel_token.is_set():
        return {"status": "cancelled"}

    emit_feedback(item, "calling_llm")
    decision = _decide_links(config, target, body[:8000], candidates)
    # Only keep links that were actually offered as candidates — never let the
    # model invent a target (which would just create another dangling link).
    candidate_names = {c.get("name", "") for c in candidates}
    valid = [n for n in decision.links if n in candidate_names]
    if not valid:
        return {"status": "no_link", "rationale": decision.rationale}

    if item.cancel_token.is_set():
        return {"status": "cancelled"}

    emit_feedback(item, "committing")
    snippet = "## Related\n\n" + "\n".join(f"- [[{n}]]" for n in valid) + "\n"
    hub = item.context.get("hub")
    op = Op(
        op=OpType.patch,
        heading="Related",
        source_basename="orphan",
        path=target,
        snippet=snippet,
        hub=hub,
        reason=f"orphan connect: {decision.rationale[:120]}",
    )
    bounds = orphan_bounds(target, hub=hub)
    result = commit_ops([op], target_dir=os.path.dirname(target), hub=hub, bounds=bounds)
    result.setdefault("linked", valid)
    return result


def _decide_links(
    config: Any,
    target_path: str,
    body: str,
    candidates: list[dict],
) -> OrphanLinkDecision:
    from silica.agent.providers import get_provider
    from silica.kernel.sanitize import parse_json

    prompt = load_prompt("orphan_prompt.txt")
    cand_block = "\n".join(
        f"{i+1}. {c.get('name', c.get('path', '?'))}"
        for i, c in enumerate(candidates)
    )
    user_message = (
        f"{prompt}\n\n---\nORPHAN NOTE ({target_path}):\n{body}\n\n"
        f"---\nCANDIDATES:\n{cand_block}\n"
    )
    provider = get_provider(config, role="worker")
    response = provider.call_llm(
        messages=[{"role": "user", "content": user_message}],
        tools=None,
        response_schema=OrphanLinkDecision,
        max_tokens=int(os.getenv("ORPHAN_MAX_TOKENS", "512")),
    )
    raw = response.text or ""
    try:
        parsed, _ = parse_json(raw, strict=False)
        if isinstance(parsed, dict):
            links = parsed.get("links", [])
            return OrphanLinkDecision(
                links=[str(x) for x in links] if isinstance(links, list) else [],
                rationale=str(parsed.get("rationale", "")),
            )
    except Exception as e:
        logger.debug("orphan link decision parse failed: %s", e)
    return OrphanLinkDecision(links=[], rationale="unparseable decision")
