"""Wiring tests: compaction levers actually run in the live loop.

- eager projection: run_agent stores the one-line summary in history,
  while the TUI event still carries the full result.
- lazy compaction: cli._compact_context collapses old reads when the
  meter is over budget and refreshes the token count.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from silica.agent.loop import run_agent


class _FakeTool:
    sensitive = False
    internal = False
    summarize = None

    def __init__(self, name: str, collapse: str, result: str):
        self.name = name
        self.collapse = collapse
        self._result = result

    def json_schema(self):
        return {
            "type": "function",
            "function": {"name": self.name, "description": "", "parameters": {"type": "object", "properties": {}}},
        }

    def run(self, _cancel_token=None, **kw):
        return self._result


def _two_turn_llm(tool_name: str):
    """First call: one tool call. Second call: final text."""
    calls = [0]

    def fake_call_llm(*a, **k):
        calls[0] += 1
        if calls[0] == 1:
            return SimpleNamespace(
                assistant_message={
                    "role": "assistant",
                    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": tool_name, "arguments": "{}"}}],
                },
                tool_calls=[SimpleNamespace(id="c1", name=tool_name, args={})],
                text="",
                reasoning=None,
            )
        return SimpleNamespace(
            assistant_message={"role": "assistant", "content": "done"},
            tool_calls=[],
            text="done",
            reasoning=None,
        )

    return fake_call_llm


def test_eager_tool_result_is_projected_in_history_but_full_in_event():
    fat = json.dumps({"written": 3, "ops": ["op"] * 50})
    tool = _FakeTool("fake_write", collapse="eager", result=fat)
    tool.summarize = staticmethod(lambda d: f"written={d['written']}")

    events = []
    messages = [{"role": "user", "content": "go"}]
    with patch.dict("silica.agent.loop.TOOLS", {"fake_write": tool}, clear=True), \
         patch("silica.agent.loop.call_llm", _two_turn_llm("fake_write")):
        run_agent(messages, model="test", tool_progress_callback=events.append)

    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert tool_msgs == [{"role": "tool", "tool_call_id": "c1", "content": "written=3"}]
    # the TUI event still carried the fat payload
    from silica.agent.events import ToolCompleteEvent
    complete = [e for e in events if isinstance(e, ToolCompleteEvent)]
    assert complete and complete[0].result == fat


def test_lazy_tool_result_stays_verbatim_in_history():
    fat = "x" * 500
    tool = _FakeTool("fake_read", collapse="lazy", result=fat)

    messages = [{"role": "user", "content": "go"}]
    with patch.dict("silica.agent.loop.TOOLS", {"fake_read": tool}, clear=True), \
         patch("silica.agent.loop.call_llm", _two_turn_llm("fake_read")):
        run_agent(messages, model="test")

    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == fat


def test_compact_context_collapses_old_read_and_recounts(monkeypatch):
    from silica import cli
    from silica.config import CONFIG

    big = "x" * 300
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "fake_read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": big},   # 2 — old read, past the floor
        {"role": "assistant", "content": "t1"},
        {"role": "assistant", "content": "t2"},
        {"role": "assistant", "content": "t3"},
    ]
    monkeypatch.setattr(CONFIG, "max_context_tokens", 100)   # budget = 60
    monkeypatch.setattr(CONFIG, "context_tokens", 1_000)     # over budget → trigger

    with patch.dict("silica.tools.TOOLS", {"fake_read": SimpleNamespace(collapse="lazy")}, clear=True):
        collapsed = cli._compact_context(messages, set())

    assert collapsed == {2}
    assert "re-call fake_read" in messages[2]["content"]
    assert CONFIG.context_tokens < 1_000  # meter refreshed after the collapse


def test_write_gate_tools_are_classified_eager():
    """Invariant: the write/gate toolset is projected at emission; everything
    else stays lazy (collapsible later, but readable in full by the model)."""
    import silica.cli  # noqa: F401 — registers the full toolset
    from silica.tools import TOOLS

    eager = {n for n, t in TOOLS.items() if t.collapse == "eager"}
    assert eager == {
        "silica_move", "silica_delete", "silica_snapshot", "silica_restore",
        "silica_cleanup", "silica_patch_note", "silica_write_note",
        "silica_flag_note",
        "silica_autolink", "silica_backlink", "silica_embed_refresh",
        "silica_cooccurrence_refresh", "silica_lexical_refresh", "silica_bulk_write",
        "silica_deferred_retry", "silica_deferred_flush", "silica_run_injector",
        "silica_anneal",
    }


def test_compact_context_noop_under_budget(monkeypatch):
    from silica import cli
    from silica.config import CONFIG

    messages = [{"role": "user", "content": "q"}]
    monkeypatch.setattr(CONFIG, "max_context_tokens", 100_000)
    monkeypatch.setattr(CONFIG, "context_tokens", 10)

    assert cli._compact_context(messages, set()) == set()
    assert CONFIG.context_tokens == 10  # no recount on the no-op path
