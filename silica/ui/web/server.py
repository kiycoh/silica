"""FastAPI backend for the localhost GUI.

Single in-memory session (localhost, one user, no auth). The critical seam is
sync `run_agent` (blocking) -> async SSE: run it in a worker thread and bridge
its callback events onto the event loop with `call_soon_threadsafe`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from silica.agent.loop import run_agent
from silica.config import CONFIG
from silica.kernel.mindmap import note_resolver
from silica.ui.web.callback import event_to_json

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# --- module-level session state (spec: single session) -----------------------
messages: list[dict] = []
current_cancel: threading.Event | None = None  # cancel token of the in-flight turn
_collapsed: set[int] = set()  # message indices elided by compaction, across turns
_busy = False  # one turn at a time; a second /chat is refused with 409
current_session_id: str | None = None  # file backing the live conversation, if saved
SESSIONS_DIR = Path.home() / ".silica" / "web_sessions"  # persisted chat transcripts

# Direct-tool buttons the REPL handles console-side; the GUI routes them through
# the agent (the tools are non-internal, so the agent may call them).
# ponytail: bare command only — button-sent, ignores folder/--force args.
_WEB_EXPANSIONS = {
    "/embed": "Refresh the embedding index: call `silica_embed_refresh` and report how many notes were indexed.",
    "/cooccur": "Refresh the co-occurrence index: call `silica_cooccurrence_refresh` and report how many notes were indexed.",
}


def _reset_session() -> None:
    from silica.cli import _fresh_messages

    global current_cancel, _busy, current_session_id
    messages[:] = _fresh_messages()
    _collapsed.clear()
    current_cancel = None
    _busy = False
    current_session_id = None  # next turn opens a new file


def _session_title(msgs: list[dict]) -> str:
    for m in msgs:
        if m.get("role") == "user" and m.get("content"):
            line = str(m["content"]).strip().splitlines()[0]
            return line[:57] + "…" if len(line) > 58 else line
    return "untitled"


def _save_session() -> None:
    """Persist the live conversation to SESSIONS_DIR/<id>.json (per vault).

    No-op until there's a user turn to name it. Called after every turn so a
    refresh/close never loses history; overwrites the same file in place.
    """
    global current_session_id
    if not any(m.get("role") == "user" and m.get("content") for m in messages):
        return
    if current_session_id is None:
        current_session_id = uuid.uuid4().hex
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "id": current_session_id,
        "title": _session_title(messages),
        "vault": CONFIG.vault_path or "",
        "updated": time.time(),
        "messages": messages,
    }
    # default=str: any non-JSON tool payload degrades to text rather than crash.
    (SESSIONS_DIR / f"{current_session_id}.json").write_text(
        json.dumps(record, default=str), encoding="utf-8"
    )


def _list_sessions() -> list[dict]:
    """Saved conversations for the current vault, newest first."""
    if not SESSIONS_DIR.exists():
        return []
    out = []
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # skip corrupt/half-written files
        if rec.get("vault", "") != (CONFIG.vault_path or ""):
            continue
        out.append(
            {"id": rec.get("id"), "title": rec.get("title", "untitled"),
             "updated": rec.get("updated", 0)}
        )
    out.sort(key=lambda r: r["updated"], reverse=True)
    return out


def _agent_message_for(text: str) -> str | None:
    """Map raw input to the agent-turn message, or None if it's not a chat turn.

    Plain text -> itself. A `/command` is expanded the same way the REPL does;
    `""` means the REPL handled it inline (nothing for the agent).
    """
    from silica.cli import _expand_workflow_shortcut

    if not text.startswith("/"):
        return text
    expanded = _expand_workflow_shortcut(text)
    if expanded is not None:
        return expanded or None
    return _WEB_EXPANSIONS.get(text.split()[0].lower() if text.split() else "")


import html as _html
import re

# A whitespace-delimited path-like token: contains "/" or ends in ".md".
_PATHLIKE = re.compile(r"[^\s\[\]]*(?:/[^\s\[\]]*|\.md)")
_WIKILINK = re.compile(r"\[\[([^\]\[]+)\]\]")
_TRAIL = ".,;:!?)"  # sentence punctuation to peel off a bare path token


def _clean_name(ref: str) -> str:
    """Display name: basename without folders or `.md` (`a/b.md` -> `b`)."""
    return ref.rsplit("/", 1)[-1].removesuffix(".md")


def _anchor(path: str, display: str) -> str:
    return (
        f'<a class="note-link" data-path="{_html.escape(path, quote=True)}">'
        f"{_html.escape(display)}</a>"
    )


def _linkify_text(text: str, resolve) -> str:
    """Turn resolvable note refs in one plain-text run into `.note-link` anchors.

    Two layers: wikilinks first (explicit `[[...]]` delimiters), then bare
    path-like tokens in the surviving prose. Unresolved refs are left verbatim.
    Returns an HTML fragment (safe parts escaped).
    """

    def link_paths(prose: str) -> str:
        out, pos = [], 0
        for m in _PATHLIKE.finditer(prose):
            out.append(_html.escape(prose[pos:m.start()]))
            tok = m.group(0)
            core = tok.rstrip(_TRAIL)
            tail = tok[len(core):]
            hit = resolve(core)
            if hit:
                out.append(_anchor(hit, _clean_name(core)) + _html.escape(tail))
            else:
                out.append(_html.escape(tok))
            pos = m.end()
        out.append(_html.escape(prose[pos:]))
        return "".join(out)

    out, pos = [], 0
    for m in _WIKILINK.finditer(text):
        out.append(link_paths(text[pos:m.start()]))
        target, _, alias = m.group(1).partition("|")
        hit = resolve(target.strip())
        if hit:
            out.append(_anchor(hit, (alias.strip() or _clean_name(target.strip()))))
        else:
            out.append(_html.escape(m.group(0)))  # broken link stays literal
        pos = m.end()
    out.append(link_paths(text[pos:]))
    return "".join(out)


def _linkify(text: str, resolve=None) -> str:
    """Render markdown to HTML, linkifying resolvable note refs when `resolve`
    is given. Works on the markdown-it token stream, so `code_inline`/`fence`
    are separate token types and code is never linkified by construction."""
    from markdown_it import MarkdownIt
    from markdown_it.token import Token

    md = MarkdownIt().enable("table")
    tokens = md.parse(text or "")
    if resolve is not None:
        for tok in tokens:
            if tok.type != "inline" or not tok.children:
                continue
            new = []
            for child in tok.children:
                if child.type != "text":
                    new.append(child)
                    continue
                frag = _linkify_text(child.content, resolve)
                raw = Token("html_inline", "", 0)
                raw.content = frag
                new.append(raw)
            tok.children = new
    return md.renderer.render(tokens, md.options, {})


def _render_md(text: str) -> str:
    return _linkify(text)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _turn_response(text: str) -> StreamingResponse:
    """One agent turn as an SSE stream. Shared by /chat and /ingest."""
    from silica.cli import _compact_context, _update_context_tokens

    global _busy
    _busy = True

    async def gen():
        global _busy, current_cancel, _collapsed
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        current_cancel = threading.Event()  # module-level so /stop can see it

        def cb(ev):  # runs in the agent/LLM worker thread
            data = event_to_json(ev)
            if data is not None:
                loop.call_soon_threadsafe(q.put_nowait, data)

        try:
            agent_msg = _agent_message_for(text)
            if agent_msg is None:
                yield _sse({"type": "error", "error": f"'{text}' not available in the GUI v1"})
                return

            msg = {"role": "user", "content": agent_msg}
            if text.startswith("/"):
                msg["origin"] = "cli"
            messages.append(msg)

            sentinel = object()
            task = asyncio.create_task(
                asyncio.to_thread(run_agent, messages, CONFIG.model, cb, cancel_token=current_cancel)
            )
            task.add_done_callback(lambda t: q.put_nowait(sentinel))

            while True:
                item = await q.get()
                if item is sentinel:
                    break
                yield _sse(item)

            answer = await task  # re-raises if run_agent failed
            _update_context_tokens(messages)
            _collapsed = _compact_context(messages, _collapsed)
            yield _sse({"type": "done", "answer": answer, "html": _linkify(answer, note_resolver())})
        except Exception as exc:  # never leave the UI stuck on the spinner
            logger.exception("web turn failed")
            yield _sse({"type": "error", "error": str(exc)})
        finally:
            _save_session()  # persist even on error so the user's turn isn't lost
            _busy = False
            current_cancel = None

    return StreamingResponse(gen(), media_type="text/event-stream")


app = FastAPI()


@app.post("/chat")
async def chat(payload: dict):
    if _busy:
        raise HTTPException(status_code=409, detail="a turn is already in progress")
    return _turn_response(payload.get("text", ""))


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if _busy:
        raise HTTPException(status_code=409, detail="a turn is already in progress")
    inbox = Path(CONFIG.vault_path or ".") / "Inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    dest = inbox / Path(file.filename or "dropped").name
    dest.write_bytes(await file.read())
    # ponytail: a dropped bare .md needs a --target; v1 surfaces that error in
    # chat and the user re-runs. PDFs/code (the happy path) stage on their own.
    return _turn_response(f'/ingest "{dest}"')


@app.get("/graph")
def graph():
    import tempfile

    from silica.tools import TOOLS

    out = Path(tempfile.gettempdir()) / "silica_web_graph.html"  # regenerated each request
    try:
        TOOLS["silica_graph_export"].run(output_path=str(out), folder="")
        return HTMLResponse(out.read_text(encoding="utf-8"))
    except Exception as exc:
        return HTMLResponse(f"<p style='font-family:monospace'>graph unavailable: {exc}</p>")


@app.get("/map")
def mindmap(note: str = ""):
    """Static-SVG radial map rooted on `note` — ephemeral, in-session (not written).

    Consumes the same precomputed positions as the .canvas serializer, so the two
    surfaces cannot diverge. Empty/unknown note degrades to a message, like /graph.
    """
    from silica.config import CONFIG
    from silica.kernel.mindmap import (
        build_mapview,
        gather_materials,
        render_map_svg,
        resolve_note_path,
    )

    if not note.strip():
        return HTMLResponse("<p style='font-family:monospace;color:#8a93a3'>enter a note: /map?note=…</p>")
    try:
        # Accept a title or a path — the input field usually gives a title.
        root = resolve_note_path(note)
        if root is None:
            return HTMLResponse(
                f"<p style='font-family:monospace;color:#8a93a3'>'{note}' not found in vault.</p>"
            )
        materials = gather_materials(root, latent_k=CONFIG.mindmap_latent_k)
        mv = build_mapview(
            root, materials, max_nodes=CONFIG.mindmap_max_nodes, hops=CONFIG.mindmap_hops
        )
        if len(mv.nodes) <= 1:
            return HTMLResponse(
                f"<p style='font-family:monospace;color:#8a93a3'>'{root}' has no neighbors to map "
                "(isolated in the graph).</p>"
            )
        return HTMLResponse(render_map_svg(mv, title=f"map · {root}"))
    except Exception as exc:
        return HTMLResponse(f"<p style='font-family:monospace'>map unavailable: {exc}</p>")


@app.get("/note")
def note(path: str = ""):
    """Read-only rendered note for the drawer. Graceful on miss (never 500).

    Only keys present in the vault index resolve, so an out-of-vault `path`
    falls through to the graceful message — path traversal is closed for free.
    """
    from silica.driver import get_driver
    from silica.driver.base import NoteRef

    resolve = note_resolver()
    canon = resolve(path)
    if not canon:
        return {"title": path, "html": "<p>note not found in vault.</p>"}
    try:
        content = get_driver().read_note(NoteRef(name=_clean_name(canon), path=canon)).content
    except Exception:
        return {"title": _clean_name(canon), "html": "<p>note unreadable.</p>"}
    return {"title": _clean_name(canon), "html": _linkify(content, resolve)}


@app.get("/messages")
def get_messages():
    resolve = note_resolver()
    data = [
        {"role": m["role"], "content": m["content"], "html": _linkify(m["content"], resolve)}
        for m in messages
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    # Vault label rides a header so the body stays a plain list (client re-render).
    return JSONResponse(data, headers={"X-Silica-Vault": CONFIG.vault_path or ""})


@app.get("/sessions")
def list_sessions():
    # Current id rides a header so the body stays a plain list (matches /messages).
    return JSONResponse(_list_sessions(), headers={"X-Silica-Session": current_session_id or ""})


@app.post("/session/load")
def load_session(payload: dict):
    global current_session_id, _collapsed
    if _busy:
        raise HTTPException(status_code=409, detail="a turn is already in progress")
    from silica.cli import _update_context_tokens

    sid = str(payload.get("id", ""))
    if not sid.isalnum():  # ids are uuid4 hex — blocks path traversal
        raise HTTPException(status_code=404, detail="no such session")
    path = SESSIONS_DIR / f"{sid}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no such session")
    rec = json.loads(path.read_text(encoding="utf-8"))
    messages[:] = rec.get("messages", [])
    _collapsed = set()
    current_session_id = sid
    _update_context_tokens(messages)
    return {"ok": True}


@app.post("/reset")
def reset():
    _reset_session()
    return {"ok": True, "vault": CONFIG.vault_path}


@app.post("/stop")
def stop():
    if current_cancel is not None:
        current_cancel.set()
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def serve(port: int = 8765) -> None:
    """Apply config, open the browser on startup, then block on uvicorn."""
    import uvicorn

    _reset_session()

    @app.on_event("startup")
    async def _open_browser():  # fires once the server is actually listening
        import webbrowser

        webbrowser.open(f"http://127.0.0.1:{port}")

    uvicorn.run(app, host="127.0.0.1", port=port)
