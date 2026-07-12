# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

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
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from silica.agent.loop import run_agent
from silica.config import CONFIG
from silica.kernel.mindmap import note_resolver
from silica.ui.web.callback import event_to_json

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# --- module-level session state (spec: single session) -----------------------
messages: list[dict] = []
current_cancel: threading.Event | None = None  # cancel token of the in-flight turn
current_task: asyncio.Task | None = None  # in-flight worker; owns the busy-gate release
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


# Fresh-session seed, precomputed so /reset ("new chat") is instant instead of
# rebuilding the vault map + token count on the click path (~seconds on a real
# vault). Built at startup, refreshed in the background after each turn (the
# turn may have written notes). (messages, their token count).
_seed: tuple[list[dict], int] | None = None


def _build_seed() -> None:
    """Compute the fresh-session seed. Never touches the live session state:
    uses the pure token counter so a background rebuild can't clobber the
    context meter of the conversation in progress."""
    global _seed
    from silica.cli import _count_context_tokens, _inject_vault_map
    from silica.kernel.vault_manifest import get_active_manifest
    from silica.prompts import system_prompt

    conv = get_active_manifest().conventions
    reply = conv.reply_language or conv.language
    msgs: list[dict] = [{"role": "system", "content": system_prompt(reply, math=True)}]
    _inject_vault_map(msgs)
    _seed = (msgs, _count_context_tokens(msgs))


def _prewarm_seed() -> None:
    """Refresh the seed off the request path; failures only cost freshness."""

    def work():
        try:
            _build_seed()
        except Exception:
            logger.exception("seed prewarm failed")

    threading.Thread(target=work, daemon=True).start()


def _reset_session() -> None:
    global current_cancel, current_task, _busy, current_session_id
    if _seed is None:
        _build_seed()
    seed_msgs, seed_tokens = _seed
    messages[:] = [dict(m) for m in seed_msgs]  # per-message copy; contents are never mutated
    CONFIG.context_tokens = seed_tokens
    _collapsed.clear()
    current_cancel = None
    current_task = None
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
from urllib.parse import quote as _quote

# A whitespace-delimited path-like token: contains "/" or ends in ".md".
_PATHLIKE = re.compile(r"[^\s\[\]]*(?:/[^\s\[\]]*|\.md)")
_WIKILINK = re.compile(r"(!?)\[\[([^\]\[]+)\]\]")  # optional ! marks an embed
_TRAIL = ".,;:!?)"  # sentence punctuation to peel off a bare path token

# Vault attachments the drawer may inline; served only through /asset, only as
# <img> (so an SVG's scripts never execute — img context runs no JS).
_ASSET_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp"}

# --- OFM (Obsidian-flavored markdown) sugar ----------------------------------
# ==highlight== | #tag (letter-first, so #123 and hex colors stay literal)
_MARK_OR_TAG = re.compile(r"==([^=\n]+)==|(?<![\w#])#([A-Za-z_][\w/-]*)")
# ponytail: regex-level strip of %%comments%% and trailing ^block-ids; move to
# the token stream if a code-heavy vault ever gets bitten.
_COMMENT = re.compile(r"%%.*?%%", re.S)
_BLOCK_ID = re.compile(r"[ \t]+\^[\w-]+[ \t]*$", re.M)
_CALLOUT_HEAD = re.compile(r"\[!(\w+)\][+-]?[ \t]*(.*)")  # first line of a callout quote
_TASK_HEAD = re.compile(r"^\[([ xX])\][ \t]+")  # first inline text of a task list item
_FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)", re.S)


def _clean_name(ref: str) -> str:
    """Display name: basename without folders or `.md` (`a/b.md` -> `b`)."""
    return ref.rsplit("/", 1)[-1].removesuffix(".md")


def _anchor(path: str, display: str) -> str:
    return (
        f'<a class="note-link" data-path="{_html.escape(path, quote=True)}">'
        f"{_html.escape(display)}</a>"
    )


def _embed_img(target: str, alias: str) -> str:
    """<img> for a `![[file.png]]` embed; a numeric alias is Obsidian's width.
    ponytail: target is taken vault-root-relative — no shortest-name resolution
    for attachments; index attachment names if that ever bites."""
    src = "/asset?path=" + _quote(target)
    width = f' width="{alias}"' if alias.isdigit() else ""
    stem = target.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    alt = stem if alias.isdigit() or not alias else alias
    return f'<img src="{_html.escape(src, quote=True)}" alt="{_html.escape(alt, quote=True)}"{width}>'


def _linkify_text(text: str, resolve) -> str:
    """Turn resolvable note refs in one plain-text run into `.note-link` anchors.

    Two layers: wikilinks first (explicit `[[...]]` delimiters), then bare
    path-like tokens in the surviving prose. Unresolved wikilinks render like
    resolved ones but tagged `.broken` (no data-path — the click is a no-op);
    unresolved bare paths stay verbatim. `resolve=None` means plain escape.
    Returns an HTML fragment (safe parts escaped).
    """
    if resolve is None:
        return _html.escape(text)

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
        bang, inner = m.group(1), m.group(2)
        target, _, alias = inner.partition("|")
        target, alias = target.strip(), alias.strip()
        if bang and "." + target.rsplit(".", 1)[-1].lower() in _ASSET_EXTS:
            out.append(_embed_img(target, alias))
        else:
            hit = resolve(target)
            display = alias or _clean_name(target)
            if hit:
                out.append(_anchor(hit, display))
            else:
                out.append(f'<a class="note-link broken">{_html.escape(display)}</a>')
        pos = m.end()
    out.append(link_paths(text[pos:]))
    return "".join(out)


def _inline_ofm(text: str, resolve) -> str:
    """OFM inline sugar over one plain-text run: ==highlight== -> <mark>,
    #tag -> chip. Prose between matches still goes through note-ref linking."""
    out, pos = [], 0
    for m in _MARK_OR_TAG.finditer(text):
        out.append(_linkify_text(text[pos:m.start()], resolve))
        if m.group(1) is not None:
            out.append(f"<mark>{_linkify_text(m.group(1), resolve)}</mark>")
        else:
            out.append(f'<span class="tag">#{_html.escape(m.group(2))}</span>')
        pos = m.end()
    out.append(_linkify_text(text[pos:], resolve))
    return "".join(out)


def _ofm_blocks(tokens) -> None:
    """OFM block sugar, rewriting the token stream in place: ```mermaid fences
    become client-rendered <pre class="mermaid">, `> [!kind] title` blockquotes
    become callouts, and `- [ ]` list items become checkbox tasks."""
    from markdown_it.token import Token

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "fence" and tok.info.strip() == "mermaid":
            raw = Token("html_block", "", 0)
            raw.content = f'<pre class="mermaid">{_html.escape(tok.content)}</pre>\n'
            tokens[i] = raw
        elif tok.type == "math_block":
            raw = Token("html_block", "", 0)
            raw.content = f'<div class="math">{_mathml(tok.content, display=True)}</div>\n'
            tokens[i] = raw
        elif tok.type == "blockquote_open":
            j = next((k for k in range(i + 1, len(tokens)) if tokens[k].type == "inline"), None)
            kids = tokens[j].children if j is not None else None
            first = kids[0] if kids else None
            m = _CALLOUT_HEAD.match(first.content) if first is not None and first.type == "text" else None
            if m:
                kind = m.group(1).lower()
                tok.attrJoin("class", f"callout callout-{kind}")
                rest = kids[1:]
                if rest and rest[0].type == "softbreak":
                    rest = rest[1:]
                tokens[j].children = rest
                head = Token("html_block", "", 0)
                title = m.group(2).strip() or kind
                head.content = f'<p class="callout-title">{_html.escape(title)}</p>\n'
                tokens.insert(i + 1, head)
                i += 1  # skip the injected title
        elif (
            tok.type == "list_item_open"
            and i + 2 < len(tokens)
            and tokens[i + 1].type == "paragraph_open"
            and tokens[i + 2].type == "inline"
            and tokens[i + 2].children
        ):
            first = tokens[i + 2].children[0]
            m = _TASK_HEAD.match(first.content) if first.type == "text" else None
            if m:
                tok.attrJoin("class", "task")
                first.content = first.content[m.end():]
                box = Token("html_inline", "", 0)
                checked = " checked" if m.group(1) in "xX" else ""
                box.content = f'<input type="checkbox" disabled{checked}> '
                tokens[i + 2].children.insert(0, box)
        i += 1


def _mathml(tex: str, display: bool) -> str:
    """LaTeX -> MathML, rendered natively by the browser (no client JS/fonts).
    A failed conversion degrades to the escaped source in a code span."""
    try:
        from latex2mathml.converter import convert

        return convert(tex, display="block" if display else "inline")
    except Exception:
        fence = "$$" if display else "$"
        return f'<code class="math-err">{_html.escape(fence + tex + fence)}</code>'


def _highlight(code: str, lang: str, _attrs: str) -> str:
    """Pygments fence highlighting; empty string falls back to a plain fence.
    Token colors live in app.css, mapped onto the site palette."""
    try:
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import get_lexer_by_name

        lexer = get_lexer_by_name(lang)
    except Exception:  # no/unknown language — markdown-it escapes it plain
        return ""
    return highlight(code, lexer, HtmlFormatter(nowrap=True))


def _linkify(text: str, resolve=None) -> str:
    """Render markdown (+ OFM sugar) to HTML, linkifying resolvable note refs
    when `resolve` is given. Works on the markdown-it token stream, so
    `code_inline`/`fence` are separate token types and code is never linkified
    or tag-ified by construction."""
    from markdown_it import MarkdownIt
    from markdown_it.token import Token
    from mdit_py_plugins.dollarmath import dollarmath_plugin

    text = _BLOCK_ID.sub("", _COMMENT.sub("", text or ""))
    md = MarkdownIt(options_update={"highlight": _highlight}).enable("table").enable("strikethrough")
    # allow_space=False keeps prose prices ("$5 and $10") out of math
    md.use(dollarmath_plugin, allow_space=False, allow_digits=False)
    tokens = md.parse(text)
    _ofm_blocks(tokens)
    for tok in tokens:
        if tok.type != "inline" or not tok.children:
            continue
        new = []
        for child in tok.children:
            if child.type == "image":
                # vault-relative image: route through /asset (absolute/external
                # and data: URLs pass untouched)
                src = child.attrGet("src") or ""
                if src and not src.startswith(("http://", "https://", "data:", "/")):
                    child.attrSet("src", "/asset?path=" + _quote(src))
                new.append(child)
                continue
            if child.type == "math_inline":
                raw = Token("html_inline", "", 0)
                raw.content = _mathml(child.content, display=False)
                new.append(raw)
                continue
            if child.type != "text":
                new.append(child)
                continue
            frag = _inline_ofm(child.content, resolve)
            raw = Token("html_inline", "", 0)
            raw.content = frag
            new.append(raw)
        tok.children = new
    return md.renderer.render(tokens, md.options, {})


def _split_frontmatter(text: str) -> tuple[dict | None, str]:
    """Split a leading YAML frontmatter block. Returns (props, body); props is
    None unless the block parses to a mapping."""
    import yaml

    m = _FRONTMATTER.match(text or "")
    if not m:
        return None, text
    try:
        props = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return None, text
    if not isinstance(props, dict):
        return None, text
    return props, text[m.end():]


def _render_frontmatter(props: dict) -> str:
    """Properties box for the note drawer: native <details>, one row per key,
    list values as individual chips."""
    rows = []
    for key, val in props.items():
        vals = val if isinstance(val, (list, tuple)) else [val]
        chips = "".join(
            f'<span class="fm-val">{_html.escape("" if v is None else str(v))}</span>'
            for v in vals
        )
        rows.append(
            f'<div class="fm-row"><span class="fm-key">{_html.escape(str(key))}</span>{chips}</div>'
        )
    return '<details class="fm" open><summary>properties</summary>' + "".join(rows) + "</details>"


def _render_md(text: str) -> str:
    return _linkify(text)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _begin_turn() -> bool:
    """Claim the single-turn slot. Sync with no `await` between the test and the
    set, so two racing POSTs can't both pass. Returns False if one's in flight."""
    global _busy
    if _busy:
        return False
    _busy = True
    return True


def _end_turn() -> None:
    """Release the turn slot. Idempotent (a completed turn and its worker's
    done-callback may both call it)."""
    global _busy, current_cancel, current_task
    _busy = False
    current_cancel = None
    current_task = None


def _sweep_if_orphaned() -> None:
    """Free a gate claimed for a turn whose `run_turn` never ran — the client
    dropped between POST and the SSE body's first `__anext__`, so nothing else
    releases it. Runs after the response closes; a no-op once a worker exists."""
    if _busy and (current_task is None or current_task.done()):
        _end_turn()


async def run_turn(text: str) -> AsyncIterator[dict]:
    """One agent turn as a stream of transport-neutral wire dicts.

    Yields `event_to_json(...)` dicts as the agent streams, then exactly one
    terminal dict: `{"type": "done", ...}` or `{"type": "error", ...}`. Owns the
    whole turn lifecycle (session append, sync→async queue bridge, cancel token,
    context compaction, save). Both `--gui` (SSE) and `connect` (WS) consume this
    — the framing is the transport's job, not this core's.

    Gate lifecycle: the slot is freed on normal end/error at once; on abandonment
    (the consumer stops iterating — a dropped SSE/WS client) the worker keeps
    running, so we signal cancel and defer the release to the worker's exit, so
    no second turn overlaps a zombie still mutating `messages`.
    """
    from silica.cli import _compact_context, _update_context_tokens

    global _busy, current_cancel, current_task, _collapsed
    if not _busy:  # direct callers (tests, future WS) that didn't pre-claim
        _busy = True
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    current_cancel = threading.Event()  # module-level so /stop can see it
    task: asyncio.Task | None = None

    def cb(ev):  # runs in the agent/LLM worker thread
        data = event_to_json(ev)
        if data is not None:
            loop.call_soon_threadsafe(q.put_nowait, data)

    try:
        agent_msg = _agent_message_for(text)
        if agent_msg is None:
            yield {"type": "error", "error": f"'{text}' not available in this session"}
            return

        msg = {"role": "user", "content": agent_msg}
        if text.startswith("/"):
            msg["origin"] = "cli"
        messages.append(msg)

        sentinel = object()
        task = asyncio.create_task(
            asyncio.to_thread(run_agent, messages, CONFIG.model, cb, cancel_token=current_cancel)
        )
        current_task = task
        task.add_done_callback(lambda t: q.put_nowait(sentinel))

        while True:
            item = await q.get()
            if item is sentinel:
                break
            yield item

        answer = await task  # re-raises if run_agent failed
        _update_context_tokens(messages)
        _collapsed = _compact_context(messages, _collapsed)
        # note_resolver reads the DRIVER graph — with the ws backend installed
        # (silica connect) a driver call on the loop thread deadlocks (`_rpc`
        # blocks the very loop that must send the frame), so render off-loop.
        html = await asyncio.to_thread(lambda: _linkify(answer, note_resolver()))
        yield {
            "type": "done",
            "answer": answer,
            "html": html,
            "context_tokens": CONFIG.context_tokens,
            "max_context_tokens": CONFIG.max_context_tokens,
        }
    except Exception as exc:  # never leave the UI stuck on the spinner
        logger.exception("web turn failed")
        yield {"type": "error", "error": str(exc)}
    finally:
        _save_session()  # persist even on error so the user's turn isn't lost
        _prewarm_seed()  # the turn may have written notes — refresh the new-chat seed
        if task is not None and not task.done():
            current_cancel.set()  # abandonment: stop the zombie...
            task.add_done_callback(lambda t: _end_turn())  # ...free the gate when it exits
        else:
            _end_turn()  # normal / error / early-return: free now


def _turn_response(text: str) -> StreamingResponse:
    """One agent turn as an SSE stream. Caller must claim the slot via
    `_begin_turn()` first; `_sweep_if_orphaned` frees it if the body never runs."""

    async def gen():
        async for item in run_turn(text):
            yield _sse(item)

    return StreamingResponse(
        gen(), media_type="text/event-stream", background=BackgroundTask(_sweep_if_orphaned)
    )


app = FastAPI()


@app.post("/chat")
async def chat(payload: dict):
    if not _begin_turn():
        raise HTTPException(status_code=409, detail="a turn is already in progress")
    return _turn_response(payload.get("text", ""))


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not _begin_turn():
        raise HTTPException(status_code=409, detail="a turn is already in progress")
    try:
        inbox = Path(CONFIG.vault_path or ".") / "Inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        dest = inbox / Path(file.filename or "dropped").name
        dest.write_bytes(await file.read())
    except Exception:
        _end_turn()  # release the slot the upload never got to use
        raise
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


@app.get("/find")
def find(q: str = "", k: int = 5):
    """Direct semantic-search panel: calls the tool straight, same pattern as /graph and /map."""
    from silica.tools import TOOLS

    q = q.strip()
    if not q:
        return HTMLResponse("<p style='font-family:monospace;color:#8a93a3'>usage: /find &lt;query&gt; [--k=N]</p>")
    try:
        parsed = json.loads(TOOLS["silica_semantic_search"].run(query=q, k=k))
    except Exception as exc:
        return HTMLResponse(f"<p style='font-family:monospace'>find unavailable: {exc}</p>")
    if "error" in parsed:
        return HTMLResponse(f"<p style='font-family:monospace;color:#8a93a3'>{_html.escape(parsed['error'])}</p>")
    results = parsed.get("results", [])
    if not results:
        return HTMLResponse(f"<p style='font-family:monospace;color:#8a93a3'>no results for '{_html.escape(q)}'.</p>")
    rows = []
    for r in results:
        p = r.get("path") or r.get("name") or "?"
        rows.append(
            f'<div class="find-result">{_anchor(p, _clean_name(p))}'
            f'<span class="find-score">{r.get("score", 0.0):.3f}</span></div>'
        )
    return HTMLResponse("".join(rows))


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
    props, body = _split_frontmatter(content)
    html = _linkify(body, resolve)
    if props:
        html = _render_frontmatter(props) + html
    return {"title": _clean_name(canon), "html": html}


@app.get("/asset")
def asset(path: str = ""):
    """Vault-relative attachment for the note drawer, `<img>`-only by contract.
    Extension whitelist + resolved-inside-the-vault check close traversal."""
    if not path or not CONFIG.vault_path:
        raise HTTPException(status_code=404)
    root = Path(CONFIG.vault_path).resolve()
    target = (root / path).resolve()
    if (
        not target.is_relative_to(root)
        or target.suffix.lower() not in _ASSET_EXTS
        or not target.is_file()
    ):
        raise HTTPException(status_code=404)
    return FileResponse(target)


@app.get("/vault_info")
def vault_info():
    """Sidebar data: vault stats + file tree, from the same builders as the
    graph view so the numbers can't disagree between the two surfaces."""
    from silica.kernel.graph_export import build_graph_data, detect_communities
    from silica.ui.web.graph_view import render_tree

    try:
        nodes, edges = build_graph_data(folder="")
        communities = detect_communities(nodes, edges)
    except Exception as exc:
        return {"error": str(exc)}
    return {
        "notes": sum(1 for n in nodes if n.get("type") != "ghost"),
        "links": sum(1 for e in edges if e.get("type") == "EXTRACTED"),
        "clusters": len(communities),
        "unresolved": sum(1 for n in nodes if n.get("type") == "ghost"),
        "tree": render_tree(nodes),
    }


@app.get("/messages")
def get_messages():
    resolve = note_resolver()
    data = [
        {"role": m["role"], "content": m["content"], "html": _linkify(m["content"], resolve)}
        for m in messages
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    # Vault label + context usage ride headers so the body stays a plain list.
    return JSONResponse(data, headers={
        "X-Silica-Vault": CONFIG.vault_path or "",
        "X-Silica-Context-Tokens": str(CONFIG.context_tokens),
        "X-Silica-Max-Context-Tokens": str(CONFIG.max_context_tokens),
    })


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
    # Cache-bust app.js/app.css by content hash: StaticFiles sets no
    # Cache-Control, so browsers serve them stale from heuristic freshness
    # (edited JS never reaches the page). A content-keyed URL can't be stale.
    # The big vendored bundles keep their long-lived cache — only these churn.
    import hashlib

    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for asset in ("app.js", "app.css"):
        ver = hashlib.sha256((STATIC_DIR / asset).read_bytes()).hexdigest()[:8]
        html = html.replace(f"/static/{asset}", f"/static/{asset}?v={ver}")
    return HTMLResponse(html)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def serve(port: int = 8765) -> None:
    """Apply config, open the browser on startup, then block on uvicorn."""
    import uvicorn

    from silica.ui.banner import print_banner
    from silica.ui.console import CONSOLE

    _reset_session()

    print_banner()
    CONSOLE.print(f"  [dim]GUI live at[/] [cyan]http://127.0.0.1:{port}[/]\n")

    uvicorn.run(app, host="127.0.0.1", port=port)
