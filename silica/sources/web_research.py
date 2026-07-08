# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""`/web-search` — agentic web-research loop → cited findings note in the Inbox.

ADR-0015 staged acquisition: Silica may *fetch* on request but never *decides*
what enters the vault. The loop is constrained to the single `web_search` tool
(it physically cannot write to the vault); its output is one findings note in
the Inbox, with sources cited. The note enters the vault only via /ingest.

`web_search` is `sensitive=True` (ADR-0009): the main agent's default toolset
excludes it, so it is reachable only here, where web_research() names it
explicitly in AgentConstraints.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import httpx

from silica.agent.constraints import AgentConstraints
from silica.agent.loop import run_agent
from silica.config import CONFIG
from silica.kernel.templates import slugify
from silica.tools import tool
from pydantic import BaseModel

_TAVILY_URL = "https://api.tavily.com/search"
_MAX_RESULTS = 5            # ponytail: module constant; per-query result cap
_HTTP_TIMEOUT = 30
_DEFAULT_MAX_SEARCHES = 12

_RESEARCH_SYSTEM_PROMPT = """You are a focused web-research agent. Given a \
concept, research it on the web and write a findings note.

Method (iterative deepening):
1. Decompose the concept into what you need to know.
2. Call `web_search(query)` for the most important sub-question.
3. Read the results, identify gaps and adjacent areas of knowledge.
4. Search again only where a gap remains. STOP when you have enough — one \
search if the concept is trivial, up to ~8-10 if it is genuinely complex. Do \
not pad with redundant searches.
5. When done, reply with NO tool call — your final message IS the note body.

The note body must be markdown prose synthesising what you found, with inline \
citations like [1], [2] tied to specific sources, and end with a section:

## Sources
1. <title> — <url>

Do not write YAML frontmatter; it is added for you. Write only the prose and \
the Sources section."""


class WebSearchArgs(BaseModel):
    query: str


@tool(WebSearchArgs, cls="atomic", sensitive=True)
def web_search(query: str) -> str:
    """Search the web for a single query. Returns a JSON list of
    {title, url, content} results. Use iteratively to research a concept."""
    key = (CONFIG.tavily_api_key or "").strip()
    if not key:
        raise ValueError(
            "web_search requires a TAVILY API key "
            "(set SILICA_TAVILY_API_KEY or TAVILY_API_KEY)."
        )
    # ponytail: direct REST, no tavily-python SDK until their API changes.
    resp = httpx.post(
        _TAVILY_URL,
        json={
            "api_key": key,
            "query": query,
            "search_depth": "advanced",
            "max_results": _MAX_RESULTS,
            "include_answer": False,
        },
        timeout=_HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    compact = [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in results
        if r.get("url")
    ]
    return json.dumps(compact, ensure_ascii=False)


def web_research(
    concept: str,
    max_searches: int = _DEFAULT_MAX_SEARCHES,
    tool_progress_callback=None,
) -> str:
    """Run the constrained web-research loop and write one findings note to the
    Inbox. Returns the note's vault-relative path.

    Raises ValueError if no TAVILY key is configured (fail fast, no loop) or if
    the loop produced no findings (sentinel return — no note is written).
    """
    if not (CONFIG.tavily_api_key or "").strip():
        raise ValueError(
            "web-search requires a TAVILY API key "
            "(set SILICA_TAVILY_API_KEY or TAVILY_API_KEY)."
        )

    messages = [
        {"role": "system", "content": _RESEARCH_SYSTEM_PROMPT},
        {"role": "user", "content": concept},
    ]
    body = run_agent(
        messages,
        model=CONFIG.model,
        tool_progress_callback=tool_progress_callback,
        constraints=AgentConstraints(
            tools=("web_search",), max_iterations=max_searches
        ),
    )

    if not body or body.startswith("(silica:"):
        raise ValueError(
            f"web-research produced no findings for {concept!r} "
            "(loop hit its limit, was cancelled, or all searches failed)."
        )

    note = _build_note(concept, body, _collect_sources(messages))
    note_rel = _unique_inbox_path(concept)
    from silica.driver import DRIVER

    DRIVER.create(note_rel, note)
    return note_rel


def _collect_sources(messages: list[dict]) -> list[tuple[str, str]]:
    """Pull (url, title) pairs from the web_search tool-result trace, deduped,
    first-seen order. These back the ADR-0015 Sources guarantee."""
    seen: dict[str, str] = {}
    for m in messages:
        if m.get("role") != "tool":
            continue
        try:
            items = json.loads(m.get("content") or "")
        except (ValueError, TypeError):
            continue
        if not isinstance(items, list):
            continue
        for it in items:
            if isinstance(it, dict) and it.get("url"):
                seen.setdefault(it["url"], it.get("title") or it["url"])
    return list(seen.items())


def _build_note(concept: str, body: str, sources: list[tuple[str, str]]) -> str:
    """Deterministic frontmatter + model body + guaranteed ## Sources.

    The date is set here, never trusted to the model. If the model already
    wrote a ## Sources section it is kept as-is; otherwise we append one from
    the collected trace (ADR-0015: sources are mandatory, not a courtesy)."""
    today = datetime.date.today().isoformat()
    front = (
        "---\n"
        f"title: {json.dumps(concept)}\n"
        "source: web-research\n"
        f"fetched: {today}\n"
        "tags: [inbox, web-research]\n"
        "---\n"
    )
    out = body.strip()
    if "## Sources" not in out:
        lines = [f"{i}. {title} — {url}" for i, (url, title) in enumerate(sources, 1)]
        sources_block = "\n".join(lines) or "(no sources captured)"
        out = f"{out}\n\n## Sources\n{sources_block}"
    return f"{front}\n{out}\n"


def _unique_inbox_path(concept: str) -> str:
    """`<inbox>/<slug>.md`, with a numeric suffix on filename collision."""
    slug = slugify(concept) or "web-research"
    candidate = f"{CONFIG.inbox_dir}/{slug}.md"
    n = 2
    while (Path(CONFIG.vault_path) / candidate).exists():
        candidate = f"{CONFIG.inbox_dir}/{slug} {n}.md"
        n += 1
    return candidate
