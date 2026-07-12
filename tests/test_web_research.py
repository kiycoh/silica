"""web_search tool + web_research orchestrator (ADR-0015 staged acquisition).

No real network (httpx.post is monkeypatched) and no real LLM (run_agent is
monkeypatched). Asserts: Tavily request shape, compact result mapping, missing
key error, sensitivity, and (later tasks) the inbox findings note.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from silica.config import CONFIG
from silica.sources import web_research as wr
from silica.tools import TOOLS


# --- web_search tool --------------------------------------------------------

def test_web_search_registered_and_sensitive():
    assert "web_search" in TOOLS
    assert TOOLS["web_search"].sensitive is True


def test_web_search_missing_key_raises(monkeypatch):
    monkeypatch.setattr(CONFIG, "tavily_api_key", "")
    with pytest.raises(ValueError, match="TAVILY"):
        wr.web_search("anything")


def test_web_search_posts_and_returns_compact_results(monkeypatch):
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k-123")
    seen = {}

    class _FakeResp:
        def raise_for_status(self):
            return self

        def json(self):
            return {
                "results": [
                    {"title": "T1", "url": "https://a.test", "content": "c1", "score": 0.9},
                    {"title": "T2", "url": "https://b.test", "content": "c2"},
                ]
            }

    def fake_post(url, json=None, timeout=None):
        seen["url"] = url
        seen["body"] = json
        seen["timeout"] = timeout
        return _FakeResp()

    monkeypatch.setattr(wr.httpx, "post", fake_post)

    out = wr.web_search("graph theory")
    items = json.loads(out)

    assert seen["url"] == wr._TAVILY_URL
    assert seen["body"]["api_key"] == "k-123"
    assert seen["body"]["query"] == "graph theory"
    assert seen["body"]["max_results"] == wr._MAX_RESULTS
    assert items == [
        {"title": "T1", "url": "https://a.test", "content": "c1"},
        {"title": "T2", "url": "https://b.test", "content": "c2"},
    ]


# --- web_research orchestrator ----------------------------------------------

def _tool_msg(items):
    return {"role": "tool", "tool_call_id": "c1", "content": json.dumps(items)}


def _patch_run_agent(monkeypatch, body, tool_results=None):
    """Fake run_agent: append tool-result messages (the source trace), return body."""
    captured = {}

    def fake_run_agent(messages, model, tool_progress_callback=None, constraints=None, **kw):
        captured["constraints"] = constraints
        captured["model"] = model
        for items in (tool_results or []):
            messages.append(_tool_msg(items))
        return body

    monkeypatch.setattr(wr, "run_agent", fake_run_agent)
    return captured


def test_web_research_writes_inbox_note_with_deterministic_frontmatter(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")
    _patch_run_agent(
        monkeypatch,
        body="Findings about graph theory [1][2].",
        tool_results=[[
            {"title": "T1", "url": "https://a.test", "content": "c1"},
            {"title": "T2", "url": "https://b.test", "content": "c2"},
        ]],
    )

    note_rel = wr.web_research("graph theory")

    assert note_rel.startswith(f"{CONFIG.inbox_dir}/")
    body = (Path(CONFIG.vault_path) / note_rel).read_text(encoding="utf-8")
    today = datetime.date.today().isoformat()
    assert 'title: "graph theory"' in body
    assert "source: web-research" in body
    assert f"fetched: {today}" in body
    assert "tags: [inbox, web-research]" in body
    assert "Findings about graph theory" in body


def test_web_research_appends_sources_when_model_omits_them(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")
    _patch_run_agent(
        monkeypatch,
        body="Findings with no sources section.",  # model forgot ## Sources
        tool_results=[[
            {"title": "T1", "url": "https://a.test", "content": "c1"},
        ]],
    )

    body = (Path(CONFIG.vault_path) / wr.web_research("x")).read_text(encoding="utf-8")
    assert body.count("## Sources") == 1
    assert "https://a.test" in body


def test_web_research_keeps_model_sources_section(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")
    _patch_run_agent(
        monkeypatch,
        body="Findings [1].\n\n## Sources\n1. T1 — https://a.test",
        tool_results=[[{"title": "T1", "url": "https://a.test", "content": "c1"}]],
    )

    body = (Path(CONFIG.vault_path) / wr.web_research("x")).read_text(encoding="utf-8")
    assert body.count("## Sources") == 1  # not doubled


def test_web_research_constrains_loop_to_web_search(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")
    captured = _patch_run_agent(
        monkeypatch,
        body="Findings.",
        tool_results=[[{"title": "T1", "url": "https://a.test", "content": "c"}]],
    )

    wr.web_research("x", max_searches=7)

    assert captured["constraints"].tools == ("web_search",)
    assert captured["constraints"].max_iterations == 7


def test_web_research_no_findings_raises_and_writes_nothing(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")
    _patch_run_agent(monkeypatch, body="(silica: maximum iterations reached)")

    with pytest.raises(ValueError, match="no findings"):
        wr.web_research("x")
    inbox = Path(CONFIG.vault_path) / CONFIG.inbox_dir
    assert not inbox.exists() or not list(inbox.glob("*.md"))


def test_web_research_missing_key_raises_before_loop(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "tavily_api_key", "")

    called = {"run": False}

    def fake_run_agent(*a, **k):
        called["run"] = True
        return "x"

    monkeypatch.setattr(wr, "run_agent", fake_run_agent)
    with pytest.raises(ValueError, match="TAVILY"):
        wr.web_research("x")
    assert called["run"] is False  # fail fast, no loop, no note


def test_web_research_empty_body_raises_and_writes_nothing(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")
    _patch_run_agent(monkeypatch, body="")

    with pytest.raises(ValueError, match="no findings"):
        wr.web_research("x")
    inbox = Path(CONFIG.vault_path) / CONFIG.inbox_dir
    assert not inbox.exists() or not list(inbox.glob("*.md"))


def test_web_research_sources_section_nonempty_when_no_sources(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")
    _patch_run_agent(monkeypatch, body="Findings with no sources and no trace.", tool_results=[])

    body = (Path(CONFIG.vault_path) / wr.web_research("x")).read_text(encoding="utf-8")
    assert body.count("## Sources") == 1
    assert "(no sources captured)" in body


def test_web_research_title_with_colon_is_valid_yaml(tmp_vault, monkeypatch):
    """A concept containing a colon must produce parseable YAML frontmatter."""
    import yaml

    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")
    _patch_run_agent(
        monkeypatch,
        body="Findings about RAG.",
        tool_results=[[{"title": "T1", "url": "https://a.test", "content": "c1"}]],
    )

    note_rel = wr.web_research("RAG: a survey")
    body = (Path(CONFIG.vault_path) / note_rel).read_text(encoding="utf-8")

    # Extract the frontmatter block between the first two --- delimiters
    parts = body.split("---\n", 2)
    assert len(parts) >= 3, "frontmatter delimiters not found"
    fm_block = parts[1]
    fm = yaml.safe_load(fm_block)
    assert fm["title"] == "RAG: a survey"
    # Ensure the malformed bare form is not present
    assert "title: RAG: a survey\n" not in body


# --- ADR-0015 / ADR-0009 boundary, as wired in production --------------------

def test_main_agent_default_toolset_excludes_web_search():
    """With web_search registered (module imported), run_agent without
    constraints must NOT expose it to the main agent."""
    from unittest.mock import patch
    from types import SimpleNamespace
    from silica.agent.loop import run_agent

    assert "web_search" in TOOLS  # registered by importing this module's target

    captured = {}

    def fake_call_llm(model, messages, tools=None, cancel=None):
        captured["tools"] = tools
        return SimpleNamespace(
            assistant_message={"role": "assistant", "content": "ok"},
            tool_calls=[], text="ok", reasoning=None,
        )

    with patch("silica.agent.loop.call_llm", fake_call_llm):
        run_agent(messages=[{"role": "user", "content": "hi"}], model="m")

    names = {t["function"]["name"] for t in (captured["tools"] or [])}
    assert "web_search" not in names
