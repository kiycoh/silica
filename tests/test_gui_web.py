"""GUI web backend — the seam that fails if sync→async streaming breaks.

Ponytail: one check per contract (event map, chat stream, ingest, reset, stop,
messages). No browser e2e in v1. Skipped whole if fastapi isn't installed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from silica.agent.events import (  # noqa: E402
    BatchRunStartEvent,
    LLMStreamEvent,
    ReasoningEvent,
    ToolCompleteEvent,
    ToolErrorEvent,
    ToolStartEvent,
)


def _read_sse(response) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


@pytest.fixture
def client(tmp_vault):
    """Fresh module-level session per test, backed by a tmp fs vault."""
    from silica.ui.web import server

    server._reset_session()
    return TestClient(server.app), server


def test_event_to_json_maps_the_render_event_seam():
    from silica.ui.web.callback import event_to_json

    assert event_to_json(LLMStreamEvent("content", "hi", 0)) == {
        "type": "delta",
        "kind": "content",
        "text": "hi",
    }
    assert event_to_json(ToolStartEvent("t", {}, "c1", 0)) == {
        "type": "tool_start",
        "name": "t",
        "id": "c1",
    }
    assert event_to_json(ToolCompleteEvent("t", {}, "c1", "ok", 0.1, 0)) == {
        "type": "tool_done",
        "name": "t",
        "id": "c1",
    }
    assert event_to_json(ToolErrorEvent("t", "c1", "boom", 0)) == {
        "type": "tool_error",
        "name": "t",
        "id": "c1",
        "error": "boom",
    }
    assert event_to_json(BatchRunStartEvent("r", "refine", "X", 3)) == {
        "type": "batch",
        "kind": "refine",
        "label": "X",
    }
    # v1 ignores reasoning/thinking events (no JSON emitted).
    assert event_to_json(ReasoningEvent("thinking", 0)) is None


def test_chat_streams_events_and_appends_the_user_message(client, monkeypatch):
    tc, server = client

    def fake_run_agent(messages, model, tool_progress_callback=None, cancel_token=None, **kw):
        tool_progress_callback(ToolStartEvent("silica_x", {}, "c1", 0))
        tool_progress_callback(LLMStreamEvent("content", "Hello", 0))
        tool_progress_callback(ToolCompleteEvent("silica_x", {}, "c1", "ok", 0.0, 0))
        messages.append({"role": "assistant", "content": "Hello"})
        return "Hello"

    monkeypatch.setattr(server, "run_agent", fake_run_agent)

    resp = tc.post("/chat", json={"text": "hi there"})
    assert resp.status_code == 200
    events = _read_sse(resp)
    types = [e["type"] for e in events]
    assert "tool_start" in types
    assert "delta" in types
    assert types[-1] == "done"
    assert events[-1]["answer"] == "Hello"
    assert any(m["role"] == "user" and m["content"] == "hi there" for m in server.messages)


def test_ingest_saves_to_inbox_and_triggers_the_ingest_command(client, monkeypatch):
    tc, server = client
    import silica.cli as cli

    calls: list[str] = []

    def fake_expand(text):
        calls.append(text)
        return f"INGEST {text}"  # non-empty → agent turn

    monkeypatch.setattr(cli, "_expand_workflow_shortcut", fake_expand)

    ran: dict = {}

    def fake_run_agent(messages, model, tool_progress_callback=None, cancel_token=None, **kw):
        ran["msgs"] = list(messages)
        messages.append({"role": "assistant", "content": "ok"})
        return "ok"

    monkeypatch.setattr(server, "run_agent", fake_run_agent)

    resp = tc.post("/ingest", files={"file": ("note.md", b"# Hi", "text/markdown")})
    assert resp.status_code == 200

    from silica.config import CONFIG

    saved = Path(CONFIG.vault_path) / "Inbox" / "note.md"
    assert saved.exists()
    assert any("/ingest" in c and "note.md" in c for c in calls)
    assert any(m["role"] == "user" and "note.md" in m["content"] for m in ran["msgs"])


def test_reset_restores_a_fresh_session(client, monkeypatch):
    tc, server = client

    def fake_run_agent(messages, model, tool_progress_callback=None, cancel_token=None, **kw):
        messages.append({"role": "assistant", "content": "a"})
        return "a"

    monkeypatch.setattr(server, "run_agent", fake_run_agent)

    tc.post("/chat", json={"text": "hi"})
    assert any(m["role"] == "user" for m in server.messages)

    r = tc.post("/reset")
    assert r.status_code == 200
    assert not any(m["role"] in ("user", "assistant") for m in server.messages)


def test_stop_signals_the_in_flight_cancel_token(client):
    tc, server = client
    import threading

    server.current_cancel = threading.Event()
    r = tc.post("/stop")
    assert r.status_code == 200
    assert server.current_cancel.is_set()


def test_messages_endpoint_returns_user_and_assistant_turns(client, monkeypatch):
    tc, server = client

    def fake_run_agent(messages, model, tool_progress_callback=None, cancel_token=None, **kw):
        messages.append({"role": "assistant", "content": "Reply"})
        return "Reply"

    monkeypatch.setattr(server, "run_agent", fake_run_agent)

    tc.post("/chat", json={"text": "question"})
    data = tc.get("/messages").json()
    roles = [m["role"] for m in data]
    assert "user" in roles and "assistant" in roles
    assert not any(m["role"] == "system" for m in data)
