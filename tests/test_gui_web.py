"""GUI web backend — the seam that fails if sync→async streaming breaks.

Ponytail: one check per contract (event map, chat stream, nucleate, reset, stop,
messages). No browser e2e in v1. Skipped whole if fastapi isn't installed.
"""
from __future__ import annotations

import asyncio
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
def client(tmp_vault, tmp_path, monkeypatch):
    """Fresh module-level session per test, backed by a tmp fs vault."""
    from silica.ui.web import server

    monkeypatch.setattr(server, "SESSIONS_DIR", tmp_path / "web_sessions")
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
        "notes": [],
    }
    # note refs are pulled from the tool args (allowlisted keys) → sources chips
    assert event_to_json(ToolStartEvent("t", {"path": "a/b.md"}, "c2", 0)) == {
        "type": "tool_start",
        "name": "t",
        "id": "c2",
        "notes": ["a/b.md"],
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


def test_index_cache_busts_churning_assets(client):
    # app.js/app.css must carry a ?v= content hash so an edited asset can't be
    # served stale from the browser's heuristic cache; vendored bundles don't.
    tc, _ = client
    html = tc.get("/").text
    import re

    assert re.search(r"/static/app\.js\?v=[0-9a-f]{8}", html), "app.js not cache-busted"
    assert re.search(r"/static/app\.css\?v=[0-9a-f]{8}", html), "app.css not cache-busted"
    assert "/static/app.js\"" not in html, "unversioned app.js reference still present"


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


def test_run_turn_yields_raw_dicts_not_sse_frames(client, monkeypatch):
    """The transport-neutral core: raw wire dicts, no `data: ` framing, ending
    in one `done` dict. This is what both `--gui` (SSE) and `connect` (WS) wrap."""
    tc, server = client

    def fake_run_agent(messages, model, tool_progress_callback=None, cancel_token=None, **kw):
        tool_progress_callback(LLMStreamEvent("text", "Hi", 0))
        messages.append({"role": "assistant", "content": "Hi"})
        return "Hi"

    monkeypatch.setattr(server, "run_agent", fake_run_agent)

    async def collect():
        return [item async for item in server.run_turn("hello")]

    items = asyncio.run(collect())
    assert all(isinstance(i, dict) for i in items)  # dicts, not SSE strings
    assert any(i["type"] == "delta" and i["text"] == "Hi" for i in items[:-1])
    assert items[-1]["type"] == "done"
    assert items[-1]["answer"] == "Hi"
    assert any(m["role"] == "user" and m["content"] == "hello" for m in server.messages)
    assert server._busy is False  # gate freed on normal completion


def test_run_turn_error_path_yields_one_error_and_frees_the_gate(client, monkeypatch):
    """A worker crash ends the stream with exactly one `error` dict, and the
    busy-gate is freed (never leave the UI stuck, never wedge the next turn)."""
    tc, server = client

    def boom(messages, model, tool_progress_callback=None, cancel_token=None, **kw):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(server, "run_agent", boom)

    async def collect():
        return [item async for item in server.run_turn("hi")]

    items = asyncio.run(collect())
    assert sum(1 for i in items if i["type"] == "error") == 1
    assert items[-1]["type"] == "error"
    assert "kaboom" in items[-1]["error"]
    assert server._busy is False


def test_run_turn_abandonment_holds_gate_until_worker_exits(client, monkeypatch):
    """Consumer stops iterating mid-stream (dropped SSE/WS client): the worker
    is a zombie until it observes the cancel. The gate MUST stay closed until it
    actually exits, or a second turn mutates `messages` concurrently."""
    import threading
    import time

    tc, server = client
    started = threading.Event()

    def slow(messages, model, tool_progress_callback=None, cancel_token=None, **kw):
        tool_progress_callback(LLMStreamEvent("text", "partial", 0))
        started.set()
        deadline = time.monotonic() + 3.0  # bounded so a broken fix FAILS, never hangs
        while (cancel_token is None or not cancel_token.is_set()) and time.monotonic() < deadline:
            time.sleep(0.005)  # spin until cancelled — the abandonment signal
        messages.append({"role": "assistant", "content": "partial"})
        return "partial"

    monkeypatch.setattr(server, "run_agent", slow)

    async def scenario():
        gen = server.run_turn("hi")
        first = await gen.__anext__()  # one delta, then abandon
        assert first["type"] == "delta"
        await asyncio.to_thread(started.wait, 1.0)
        await gen.aclose()  # GeneratorExit into run_turn

        # zombie still alive → gate closed, cancel signalled
        assert server._busy is True
        assert server.current_cancel is not None and server.current_cancel.is_set()

        # once the worker sees the cancel and exits, its done-callback frees the gate
        for _ in range(400):
            if not server._busy:
                break
            await asyncio.sleep(0.005)
        assert server._busy is False

    asyncio.run(scenario())


def test_sweep_frees_the_gate_when_no_worker_ever_started(client):
    """Never-iterated generator (client drops between POST and first __anext__):
    run_turn never runs, so the SSE background sweep frees the eagerly-claimed
    gate. Guards against a permanently 409-locked server."""
    tc, server = client
    assert server._begin_turn() is True
    assert server._busy is True
    server.current_task = None  # no worker was created
    server._sweep_if_orphaned()
    assert server._busy is False


def test_nucleate_stages_uploads_and_hands_files_to_the_agent(client, monkeypatch):
    tc, server = client

    ran: dict = {}

    def fake_run_agent(messages, model, tool_progress_callback=None, cancel_token=None, **kw):
        ran["msgs"] = list(messages)
        messages.append({"role": "assistant", "content": "ok"})
        return "ok"

    monkeypatch.setattr(server, "run_agent", fake_run_agent)

    resp = tc.post(
        "/nucleate",
        files=[("files", ("note.md", b"# Hi\n\nsome body text to stage", "text/markdown"))],
        data={"text": "file these under Concepts/AI"},
    )
    assert resp.status_code == 200

    from silica.config import CONFIG

    saved = Path(CONFIG.vault_path) / "Inbox" / "note.md"
    assert saved.exists()  # upload landed in the inbox (not nucleated yet)
    # the agent turn carries the user's instruction *and* the staged file path
    user = next(m for m in ran["msgs"] if m["role"] == "user")
    assert "file these under Concepts/AI" in user["content"]
    assert "Inbox/note.md" in user["content"]


def test_compose_nucleate_turn_defaults_empty_text_and_lists_files():
    from silica.ui.web.server import _compose_nucleate_turn

    # empty instruction → default nucleate ask; markdown vs code stubs both listed
    msg = _compose_nucleate_turn("", ["Inbox/a.md"], ["Code/b.md"])
    assert "Nucleate the attached file(s)" in msg
    assert "Inbox/a.md" in msg and "Code/b.md" in msg

    # a real instruction is kept verbatim as the turn's lead
    msg2 = _compose_nucleate_turn("summarize these", ["Inbox/a.md"], [])
    assert msg2.startswith("summarize these")
    assert "Inbox/a.md" in msg2


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


def test_sessions_persist_across_reset_and_reload(client, monkeypatch):
    tc, server = client

    def fake_run_agent(messages, model, tool_progress_callback=None, cancel_token=None, **kw):
        messages.append({"role": "assistant", "content": "Reply one"})
        return "Reply one"

    monkeypatch.setattr(server, "run_agent", fake_run_agent)

    tc.post("/chat", json={"text": "first question"})
    listed = tc.get("/sessions")
    sessions = listed.json()
    assert len(sessions) == 1
    assert sessions[0]["title"] == "first question"
    sid = sessions[0]["id"]
    assert listed.headers["X-Silica-Session"] == sid

    # new chat clears the live session; the saved one survives on disk
    tc.post("/reset")
    assert not any(m["role"] in ("user", "assistant") for m in server.messages)

    r = tc.post("/session/load", json={"id": sid})
    assert r.status_code == 200
    assert any(m.get("content") == "Reply one" for m in server.messages)
    assert server.current_session_id == sid

    # unknown / path-traversal ids are rejected
    assert tc.post("/session/load", json={"id": "../../etc/passwd"}).status_code == 404
    assert tc.post("/session/load", json={"id": "deadbeef"}).status_code == 404


# ---------------------------------------------------------------------------
# _linkify — resolvable note refs become .note-link anchors (token-stream, so
# code is never touched). Pure: driven by a fake dict resolver, no vault.
# ---------------------------------------------------------------------------

_FAKE_INDEX = {
    "Foo": "Foo.md",
    "a/b": "sub/a-b.md",
    "concepts/mind-maps.md": "concepts/mind-maps.md",
    "concepts/x.md": "concepts/x.md",
    "index": "index.md",  # resolvable, but not path-shaped → must NOT link
}


def _fake_resolve(ref: str):
    return _FAKE_INDEX.get(ref)


def test_linkify_resolved_wikilink_becomes_clean_anchor():
    from silica.ui.web.server import _linkify

    html = _linkify("see [[Foo]] here", _fake_resolve)
    assert '<a class="note-link" data-path="Foo.md">Foo</a>' in html
    assert "[[" not in html and "]]" not in html


def test_linkify_wikilink_alias_shows_alias_but_resolves_target():
    from silica.ui.web.server import _linkify

    html = _linkify("read [[a/b|Bar]] now", _fake_resolve)
    assert 'data-path="sub/a-b.md"' in html
    assert ">Bar</a>" in html


def test_linkify_unresolved_wikilink_renders_as_broken_anchor():
    from silica.ui.web.server import _linkify

    html = _linkify("a [[nope]] ref", _fake_resolve)
    assert '<a class="note-link broken">nope</a>' in html
    assert "data-path" not in html  # click is a no-op by construction
    assert "[[" not in html


def test_linkify_pathlike_md_token_becomes_link_with_clean_name():
    from silica.ui.web.server import _linkify

    html = _linkify("open concepts/mind-maps.md today", _fake_resolve)
    assert 'data-path="concepts/mind-maps.md"' in html
    assert ">mind-maps</a>" in html


def test_linkify_bare_word_is_never_linked():
    from silica.ui.web.server import _linkify

    # `index` resolves in the fake index, but has no `/` and no `.md` → not a
    # link candidate, so predictability wins over resolvability.
    html = _linkify("the index of notes", _fake_resolve)
    assert "note-link" not in html


def test_linkify_never_touches_code():
    from silica.ui.web.server import _linkify

    html = _linkify("run `concepts/x.md` inline", _fake_resolve)
    assert "note-link" not in html
    assert "<code>concepts/x.md</code>" in html


def test_linkify_without_resolver_is_plain_render():
    from silica.ui.web.server import _linkify

    assert _linkify("see [[Foo]] here").strip() == "<p>see [[Foo]] here</p>"


def test_embed_with_subpath_fragment_still_renders_image():
    # Obsidian embeds carry a #center/#heading subpath and a width alias:
    # the fragment must not defeat the asset-extension check (regression).
    from silica.ui.web.server import _linkify

    html = _linkify("![[Pasted image 1.png#center|500]]", _fake_resolve)
    assert '<img src="/asset?path=Pasted%20image%201.png"' in html
    assert 'width="500"' in html
    assert "note-link broken" not in html


# ---------------------------------------------------------------------------
# OFM sugar — highlights, tags, callouts, tasks, mermaid, comments/block-ids,
# frontmatter. Same pure-resolver setup as the _linkify tests above.
# ---------------------------------------------------------------------------

def test_ofm_highlight_and_tag_render():
    from silica.ui.web.server import _linkify

    html = _linkify("a ==hot== take on #graph/theory", _fake_resolve)
    assert "<mark>hot</mark>" in html
    assert '<span class="tag">#graph/theory</span>' in html


def test_ofm_sugar_never_fires_in_code():
    from silica.ui.web.server import _linkify

    html = _linkify("run `#foo` now\n\n```\n#bar\n==nope==\n```", _fake_resolve)
    assert 'class="tag"' not in html
    assert "<mark>" not in html


def test_ofm_callout_gets_class_and_title():
    from silica.ui.web.server import _linkify

    html = _linkify("> [!warning] Watch out\n> the body", _fake_resolve)
    assert 'class="callout callout-warning"' in html
    assert '<p class="callout-title">Watch out</p>' in html
    assert "the body" in html
    assert "[!warning]" not in html


def test_ofm_plain_blockquote_is_untouched():
    from silica.ui.web.server import _linkify

    html = _linkify("> just a quote", _fake_resolve)
    assert "callout" not in html


def test_ofm_task_items_become_checkboxes():
    from silica.ui.web.server import _linkify

    html = _linkify("- [ ] open\n- [x] done", _fake_resolve)
    assert html.count('<input type="checkbox" disabled') == 2
    assert 'disabled checked' in html
    assert "[ ]" not in html and "[x]" not in html


def test_ofm_mermaid_fence_becomes_client_hook():
    from silica.ui.web.server import _linkify

    html = _linkify("```mermaid\ngraph TD; A-->B;\n```", _fake_resolve)
    assert '<pre class="mermaid">' in html
    assert "A--&gt;B" in html  # content is escaped, mermaid.js reads textContent
    assert "mermaid" not in _linkify("```python\nx = 1\n```", _fake_resolve)


def test_ofm_comments_and_block_ids_stripped():
    from silica.ui.web.server import _linkify

    html = _linkify("keep %%hidden%% this ^anchor-id\nnext line", _fake_resolve)
    assert "hidden" not in html
    assert "anchor-id" not in html
    assert "keep" in html and "next line" in html


def test_ofm_image_embed_becomes_img_via_asset():
    from silica.ui.web.server import _linkify

    html = _linkify("see ![[img/pic 1.png]] and ![[shot.jpg|300]]", _fake_resolve)
    assert '<img src="/asset?path=img/pic%201.png" alt="pic 1">' in html
    assert '<img src="/asset?path=shot.jpg" alt="shot" width="300">' in html


def test_markdown_relative_image_src_routes_through_asset():
    from silica.ui.web.server import _linkify

    html = _linkify("![alt](img/pic.png) ![ext](https://x.io/p.png)", _fake_resolve)
    assert 'src="/asset?path=img/pic.png"' in html
    assert 'src="https://x.io/p.png"' in html


def test_fence_gets_pygments_spans():
    from silica.ui.web.server import _linkify

    html = _linkify('```python\ndef f():\n    return "x"\n```', _fake_resolve)
    assert '<span class="k">def</span>' in html
    assert 'language-python' in html
    # unknown language degrades to a plain escaped fence
    assert "<span" not in _linkify("```nolang\nx\n```", _fake_resolve)


def test_asset_endpoint_serves_vault_images_and_closes_traversal(client, tmp_vault):
    from pathlib import Path as _Path

    from silica.config import CONFIG

    tc, _server = client
    tmp_vault.note("img/pic.png", "fake-bytes")
    tmp_vault.note("secret.txt", "no")
    # image that only exists one level above the vault root
    (_Path(CONFIG.vault_path).parent / "outside.png").write_text("leak", encoding="utf-8")

    assert tc.get("/asset", params={"path": "img/pic.png"}).status_code == 200
    # `![[pic.png]]` names the attachment by basename though it lives in img/
    assert tc.get("/asset", params={"path": "pic.png"}).status_code == 200
    assert tc.get("/asset", params={"path": "secret.txt"}).status_code == 404  # not whitelisted
    assert tc.get("/asset", params={"path": "missing.png"}).status_code == 404
    # traversal stays closed: the basename fallback only ever serves an in-vault
    # file, never one living outside the vault, whatever the path spelling.
    assert tc.get("/asset", params={"path": "outside.png"}).status_code == 404
    assert tc.get("/asset", params={"path": "../outside.png"}).status_code == 404
    assert tc.get("/asset", params={"path": "../../outside.png"}).status_code == 404


def test_latex_inline_and_block_become_mathml():
    from silica.ui.web.server import _linkify

    html = _linkify("energy $E=mc^2$ here", _fake_resolve)
    assert "<math" in html and "$" not in html

    html = _linkify("$$\n\\frac{a}{b}\n$$", _fake_resolve)
    assert '<div class="math">' in html
    assert 'display="block"' in html


def test_latex_prose_dollars_and_code_stay_literal():
    from silica.ui.web.server import _linkify

    html = _linkify("costs $5 and $10 today", _fake_resolve)
    assert "<math" not in html
    html = _linkify("run `$x^2$` inline", _fake_resolve)
    assert "<math" not in html and "$x^2$" in html


def test_split_frontmatter_returns_props_and_body():
    from silica.ui.web.server import _split_frontmatter

    props, body = _split_frontmatter("---\ntags: [a, b]\nstatus: seed\n---\n# Title\n")
    assert props == {"tags": ["a", "b"], "status": "seed"}
    assert body == "# Title\n"


def test_split_frontmatter_absent_or_non_mapping_is_none():
    from silica.ui.web.server import _split_frontmatter

    assert _split_frontmatter("# no fm")[0] is None
    assert _split_frontmatter("---\n- just\n- a list\n---\nbody")[0] is None


def test_note_endpoint_renders_frontmatter_properties_box(client, tmp_vault):
    tc, _server = client
    tmp_vault.note("Foo.md", "---\ntags: [x]\nstatus: seed\n---\nbody ==lit==")

    html = tc.get("/note", params={"path": "Foo.md"}).json()["html"]
    assert '<details class="fm"' in html
    assert '<span class="fm-key">tags</span>' in html
    assert '<span class="fm-val">x</span>' in html
    assert "<mark>lit</mark>" in html
    assert "<hr" not in html  # the --- fences never reach the markdown renderer


# ---------------------------------------------------------------------------
# GET /note — read-only rendered note for the drawer.
# ---------------------------------------------------------------------------

def test_note_endpoint_returns_title_and_linkified_html(client, tmp_vault):
    tc, _server = client
    tmp_vault.note("Foo.md", "# Foo")
    tmp_vault.note("concepts/mind-maps.md", "body links to [[Foo]] inside")

    data = tc.get("/note", params={"path": "concepts/mind-maps.md"}).json()
    assert data["title"] == "mind-maps"
    assert 'class="note-link"' in data["html"]
    assert 'data-path="Foo.md"' in data["html"]


def test_note_endpoint_missing_path_is_graceful_not_500(client, tmp_vault):
    tc, _server = client
    r = tc.get("/note", params={"path": "does/not/exist.md"})
    assert r.status_code == 200
    assert "html" in r.json()


def test_note_endpoint_rejects_path_outside_vault(client, tmp_vault):
    tc, _server = client
    r = tc.get("/note", params={"path": "../../etc/passwd"})
    assert r.status_code == 200
    assert "note-link" not in r.json()["html"]  # nothing read, graceful message


# ---------------------------------------------------------------------------
# GET /find — direct semantic-search panel, bypasses the agent.
# ---------------------------------------------------------------------------

def test_find_endpoint_requires_a_query(client):
    tc, _server = client
    r = tc.get("/find", params={"q": ""})
    assert r.status_code == 200
    assert "usage: /find" in r.text


def test_find_endpoint_reports_empty_index_gracefully(client, tmp_path, monkeypatch):
    tc, _server = client
    monkeypatch.setattr("silica.kernel.embed._index_path", lambda: tmp_path / "empty.json")
    r = tc.get("/find", params={"q": "gears"})
    assert r.status_code == 200
    # Both legs empty (embed + co-occurrence) → the facade reports no index.
    assert "No index available" in r.text


def test_find_endpoint_renders_results_as_note_links(client, tmp_path, monkeypatch):
    from unittest.mock import MagicMock, patch
    from silica.kernel.embed import EmbedStore

    tc, _server = client
    idx = tmp_path / "embeddings.json"
    monkeypatch.setattr("silica.kernel.embed._index_path", lambda: idx)
    store = EmbedStore(idx)
    store.upsert("Concepts/A", "A", [1.0, 0.0])
    store.save()

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[1.0, 0.0]]
    with patch("silica.agent.providers.get_embedder", return_value=mock_embedder):
        r = tc.get("/find", params={"q": "gears", "k": 1})

    assert r.status_code == 200
    assert 'data-path="Concepts/A"' in r.text
    assert "find-score" in r.text


# ---------------------------------------------------------------------------
# GET /messages — context-token usage rides response headers.
# ---------------------------------------------------------------------------

def test_messages_endpoint_reports_context_token_headers(client, monkeypatch):
    tc, server = client
    from silica.config import CONFIG

    monkeypatch.setattr(CONFIG, "context_tokens", 42)
    monkeypatch.setattr(CONFIG, "max_context_tokens", 1000)
    r = tc.get("/messages")
    assert r.headers["X-Silica-Context-Tokens"] == "42"
    assert r.headers["X-Silica-Max-Context-Tokens"] == "1000"


def test_chat_done_html_linkifies_a_cited_note(client, tmp_vault, monkeypatch):
    tc, server = client
    tmp_vault.note("Foo.md", "# Foo")

    def fake_run_agent(messages, model, tool_progress_callback=None, cancel_token=None, **kw):
        messages.append({"role": "assistant", "content": "look at [[Foo]]"})
        return "look at [[Foo]]"

    monkeypatch.setattr(server, "run_agent", fake_run_agent)
    events = _read_sse(tc.post("/chat", json={"text": "where?"}))
    done = events[-1]
    assert done["type"] == "done"
    assert 'class="note-link"' in done["html"]
    assert 'data-path="Foo.md"' in done["html"]
