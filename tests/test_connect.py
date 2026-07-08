"""`silica connect` — the WS bridge server the Obsidian plugin dials into.

Covers PROTOCOL.md's server half: bridge-file discovery, the handshake gate
(token / protocolVersion / Origin), the chat channel framed over `run_turn`
(chat_event*/chat_done, busy refusal, cancel), and the driver channel — the
accepted connection is wrapped in an attached ObsidianWSBackend, installed as
the global driver, and uninstalled (fallback to CONFIG.backend) on drop.

The fake plugin here is a plain websockets client: production topology
(Python hosts, plugin dials) — the inverse of test_ws_backend's unit-3 seam.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path

import pytest

pytest.importorskip("websockets")
pytest.importorskip("fastapi")  # run_turn lives in the web server module
from websockets.asyncio.client import connect  # noqa: E402

from silica.agent.events import LLMStreamEvent  # noqa: E402
from silica.config import CONFIG  # noqa: E402
from silica.driver.base import NoteRef  # noqa: E402


@pytest.fixture
def bridge_env(tmp_vault, tmp_path, monkeypatch):
    """Fresh web-session state + driver singleton around each bridge test."""
    from silica.ui.web import server as web

    monkeypatch.setattr(web, "SESSIONS_DIR", tmp_path / "web_sessions")
    web._reset_session()
    yield web


def _run(scenario) -> None:
    """Boot a BridgeServer, run the async scenario against it, tear down."""
    from silica.ui.connect import BridgeServer

    async def main():
        srv = BridgeServer()
        await srv.start()
        try:
            await asyncio.wait_for(scenario(srv), timeout=15)
        finally:
            await srv.stop()

    asyncio.run(main())


async def _dial(srv, token=None, **kwargs):
    """Dial + hello like the plugin; return (ws, first reply frame)."""
    ws = await connect(f"ws://127.0.0.1:{srv.port}", **kwargs)
    await ws.send(json.dumps({
        "type": "hello",
        "token": srv.token if token is None else token,
        "protocolVersion": 1, "role": "plugin",
    }))
    return ws, json.loads(await ws.recv())


# ---------------------------------------------------------------------------
# Discovery + handshake gate
# ---------------------------------------------------------------------------

def test_bridge_file_written_and_removed(bridge_env):
    path = Path(CONFIG.vault_path) / ".obsidian" / "silica-bridge.json"

    async def scenario(srv):
        rec = json.loads(path.read_text(encoding="utf-8"))
        assert rec == {"port": srv.port, "token": srv.token,
                       "pid": os.getpid(), "protocolVersion": 1}
        assert path.stat().st_mode & 0o777 == 0o600

    _run(scenario)
    assert not path.exists()  # removed on stop


def test_handshake_welcome(bridge_env):
    async def scenario(srv):
        ws, welcome = await _dial(srv)
        assert welcome["type"] == "welcome"
        assert welcome["protocolVersion"] == 1
        assert welcome["vault"] == Path(CONFIG.vault_path).name
        await ws.close()

    _run(scenario)


def test_bad_token_gets_bye(bridge_env):
    async def scenario(srv):
        ws, reply = await _dial(srv, token="wrong")
        assert reply["type"] == "bye"
        assert "token" in reply["reason"]
        await ws.close()

    _run(scenario)


def test_browser_origin_refused(bridge_env):
    async def scenario(srv):
        # Browsers set Origin; the plugin's native WebSocket does not.
        ws, reply = await _dial(srv, origin="https://evil.example")
        assert reply["type"] == "bye"
        assert "origin" in reply["reason"].lower()
        await ws.close()

    _run(scenario)


# ---------------------------------------------------------------------------
# Chat channel — run_turn framed as chat_event*/chat_done
# ---------------------------------------------------------------------------

_EMPTY_GRAPH_RPC = {
    "list_files": [],
    "resolved_links": {"resolved": {}, "unresolved": {}},
    "mention_index": {},
}


async def _recv_chat(ws):
    """Next chat frame, answering interleaved driver rpc frames (the graph
    reads note_resolver issues while rendering chat_done's html) with empty
    results — this is what proves rpc and chat share the socket without
    deadlocking the loop."""
    while True:
        frame = json.loads(await ws.recv())
        if frame.get("type") == "rpc":
            await ws.send(json.dumps({"type": "rpc_result", "id": frame["id"],
                                      "result": _EMPTY_GRAPH_RPC.get(frame["method"])}))
            continue
        return frame

def test_chat_streams_events_then_done(bridge_env, monkeypatch):
    web = bridge_env

    def fake_run_agent(messages, model, tool_progress_callback=None, cancel_token=None, **kw):
        tool_progress_callback(LLMStreamEvent("content", "Hello", 0))
        messages.append({"role": "assistant", "content": "Hello"})
        return "Hello"

    monkeypatch.setattr(web, "run_agent", fake_run_agent)

    async def scenario(srv):
        ws, _ = await _dial(srv)
        await ws.send(json.dumps({"type": "chat", "turnId": "t1", "text": "hi"}))
        frames = []
        while True:
            frame = await _recv_chat(ws)
            frames.append(frame)
            if frame["type"] in ("chat_done", "chat_error"):
                break
        assert all(f["turnId"] == "t1" for f in frames)
        assert any(f["type"] == "chat_event" and f["event"]["type"] == "delta"
                   and f["event"]["text"] == "Hello" for f in frames)
        assert frames[-1]["type"] == "chat_done"
        assert frames[-1]["answer"] == "Hello"
        assert frames[-1]["html"]  # rendered markdown travels with the answer
        await ws.close()

    _run(scenario)


def test_second_chat_while_busy_is_refused(bridge_env, monkeypatch):
    web = bridge_env
    release = threading.Event()

    def fake_run_agent(messages, model, tool_progress_callback=None, cancel_token=None, **kw):
        tool_progress_callback(LLMStreamEvent("content", "…", 0))
        release.wait(10)
        messages.append({"role": "assistant", "content": "ok"})
        return "ok"

    monkeypatch.setattr(web, "run_agent", fake_run_agent)

    async def scenario(srv):
        ws, _ = await _dial(srv)
        await ws.send(json.dumps({"type": "chat", "turnId": "t1", "text": "first"}))
        first = await _recv_chat(ws)  # t1 is in flight once its delta lands
        assert first["type"] == "chat_event"
        await ws.send(json.dumps({"type": "chat", "turnId": "t2", "text": "second"}))
        refusal = await _recv_chat(ws)
        assert refusal == {"type": "chat_error", "turnId": "t2",
                           "error": "a turn is already in progress"}
        release.set()
        done = await _recv_chat(ws)
        assert done["type"] == "chat_done" and done["turnId"] == "t1"
        await ws.close()

    _run(scenario)


def test_chat_cancel_sets_the_cancel_token(bridge_env, monkeypatch):
    web = bridge_env

    def fake_run_agent(messages, model, tool_progress_callback=None, cancel_token=None, **kw):
        tool_progress_callback(LLMStreamEvent("content", "…", 0))
        cancelled = cancel_token.wait(10)
        answer = "cancelled" if cancelled else "timeout"
        messages.append({"role": "assistant", "content": answer})
        return answer

    monkeypatch.setattr(web, "run_agent", fake_run_agent)

    async def scenario(srv):
        ws, _ = await _dial(srv)
        await ws.send(json.dumps({"type": "chat", "turnId": "t1", "text": "go"}))
        first = await _recv_chat(ws)
        assert first["type"] == "chat_event"
        await ws.send(json.dumps({"type": "chat_cancel", "turnId": "t1"}))
        done = await _recv_chat(ws)
        assert done["type"] == "chat_done"
        assert done["answer"] == "cancelled"
        await ws.close()

    _run(scenario)


# ---------------------------------------------------------------------------
# Driver channel — attached ws backend installed on handshake, dropped on close
# ---------------------------------------------------------------------------

def test_rpc_frames_serve_the_installed_driver(bridge_env):
    from silica.driver import get_driver
    from silica.driver.ws_backend import ObsidianWSBackend

    async def scenario(srv):
        ws, _ = await _dial(srv)
        be = get_driver()
        assert isinstance(be, ObsidianWSBackend)

        async def plugin():
            frame = json.loads(await ws.recv())
            assert frame["type"] == "rpc" and frame["method"] == "read"
            await ws.send(json.dumps({
                "type": "rpc_result", "id": frame["id"],
                "result": {"path": frame["params"]["path"],
                           "content": "over the bridge", "size": 15},
            }))

        replier = asyncio.create_task(plugin())
        # Driver calls arrive from the sync agent worker thread in production.
        nc = await asyncio.to_thread(be.read_note, NoteRef(name="X", path="X.md"))
        await replier
        assert nc.content == "over the bridge"
        await ws.close()

    _run(scenario)


def test_plugin_drop_falls_back_to_local_backend(bridge_env):
    from silica.driver import get_driver
    from silica.driver.ws_backend import ObsidianWSBackend

    async def scenario(srv):
        ws, _ = await _dial(srv)
        assert isinstance(get_driver(), ObsidianWSBackend)
        await ws.close()
        for _ in range(200):  # the handler notices the drop asynchronously
            if not isinstance(get_driver(), ObsidianWSBackend):
                break
            await asyncio.sleep(0.02)
        assert not isinstance(get_driver(), ObsidianWSBackend)  # fs fallback (tmp_vault)

    _run(scenario)


# ---------------------------------------------------------------------------
# Startup fallback + CLI wiring
# ---------------------------------------------------------------------------

def test_ws_config_resolves_to_local_fallback(monkeypatch):
    import silica.ui.connect as connect_mod

    monkeypatch.setattr(connect_mod.shutil, "which", lambda n: "/usr/bin/obsidian")
    assert connect_mod._fallback_backend() == "cli"
    monkeypatch.setattr(connect_mod.shutil, "which", lambda n: None)
    assert connect_mod._fallback_backend() == "fs"


def test_cli_dispatches_silica_connect(monkeypatch):
    import silica.cli as cli
    import silica.ui.connect as connect_mod

    calls = []
    monkeypatch.setattr(connect_mod, "run_connect", lambda: calls.append(1) or 0)
    monkeypatch.setattr(cli, "_activate_repo_mode", lambda: None)
    monkeypatch.setattr(cli, "_resolve_context_budget", lambda: None)
    assert cli._dispatch_subcommand(["connect"]) == 0
    assert calls == [1]
