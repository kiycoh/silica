# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Obsidian WS Backend — L0 implementation over the Silica ⇄ Obsidian bridge.

Speaks PROTOCOL.md's `rpc`/`rpc_result` channel to the Obsidian plugin. Driver
calls arrive on the sync agent worker thread, so the socket lives on a dedicated
event-loop thread and `_rpc` marshals across via `run_coroutine_threadsafe` +
`future.result(timeout)`. Each write is one RPC; the plugin holds the
postcondition and its reply is the settle, so writes need no polling.

Nothing from `cli_backend`'s CDP machinery is ported (no `_js_str`, no `_eval`,
no subprocess, no settle waiters) — the plugin holds the postconditions and the
reply *is* the settle (PROTOCOL.md §2.4).
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any

import networkx as nx
from websockets.asyncio.client import connect

from silica.driver.base import (
    GraphIndexMixin,
    GraphSnapshot,
    Heading,
    Hit,
    Link,
    NoteContent,
    NoteRef,
    Txn,
    build_title_trie,
    mentions_in,
)
from silica.kernel.ast import extract_links
from silica.kernel.ops import InverseOp, InverseOpKind

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1


class ObsidianWSBackend(GraphIndexMixin):
    """ObsidianDriver over the WebSocket bridge (read path)."""

    def __init__(self, url: str, token: str = "", timeout: float = 10.0):
        self._url = url
        self._token = token
        self._timeout = timeout

        # Connection lives on its own loop thread; per-id futures correlate RPCs.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ws: Any = None
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._start_lock = threading.Lock()

        # In-memory graph index (same shape the mixin/base helpers expect).
        self._graph = nx.DiGraph()
        self._unresolved_links: set[tuple[str, str]] = set()
        self._notes: dict[str, NoteRef] = {}
        self._notes_by_name: dict[str, list[NoteRef]] = {}
        self._mention_index: dict[str, set[str]] = {}
        self._is_graph_built = False

    # ------------------------------------------------------------------
    # Connection + RPC transport
    # ------------------------------------------------------------------

    @classmethod
    def attached(cls, ws: Any, loop: asyncio.AbstractEventLoop) -> ObsidianWSBackend:
        """Wrap an already-accepted connection (production: `silica connect`
        hosts the server, the plugin dialed in). The caller owns socket and
        loop — it routes rpc_result/rpc_error frames to `_on_frame` and must
        use `detach()` (never `close()`, which would stop the shared loop)."""
        be = cls(url="", token="")
        be._ws = ws
        be._loop = loop
        be._ready.set()
        return be

    def detach(self, reason: str) -> None:
        """Disconnect an attached backend: fail in-flight RPCs, refuse new ones."""
        self._error = RuntimeError(reason)
        self._ws = None
        self._fail_pending(self._error)

    def _ensure_connected(self) -> None:
        with self._start_lock:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                self._thread = threading.Thread(
                    target=self._loop.run_forever, name="silica-ws", daemon=True
                )
                self._thread.start()
                asyncio.run_coroutine_threadsafe(self._run(), self._loop)
        if not self._ready.wait(self._timeout):
            raise RuntimeError(f"WS handshake timed out after {self._timeout:.0f}s ({self._url})")
        if self._error is not None:
            raise RuntimeError(f"WS connection failed: {self._error}")

    async def _run(self) -> None:
        try:
            async with connect(self._url) as ws:
                self._ws = ws
                await ws.send(json.dumps({
                    "type": "hello", "token": self._token,
                    "protocolVersion": PROTOCOL_VERSION, "role": "driver",
                }))
                welcome = json.loads(await ws.recv())
                if welcome.get("type") != "welcome":
                    raise RuntimeError(f"handshake refused: {welcome.get('reason', welcome)}")
                if welcome.get("protocolVersion") != PROTOCOL_VERSION:
                    raise RuntimeError(f"protocol mismatch: {welcome.get('protocolVersion')}")
                self._ready.set()
                async for raw in ws:
                    self._on_frame(json.loads(raw))
        except Exception as exc:  # unblock the waiter so it sees the failure
            self._error = exc
            self._ready.set()
            self._fail_pending(exc)

    def _on_frame(self, frame: dict) -> None:
        # Only rpc replies concern the driver; chat_*/event frames are the
        # transport's (unit 5) and are ignored here.
        kind = frame.get("type")
        rid = frame.get("id")
        if kind not in ("rpc_result", "rpc_error") or rid is None:
            return
        fut = self._pending.pop(rid, None)
        if fut is None or fut.done():
            return
        if kind == "rpc_result":
            fut.set_result(frame.get("result"))
        else:
            fut.set_exception(RuntimeError(frame.get("error", "rpc_error")))

    def _fail_pending(self, exc: Exception) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()

    async def _send_rpc(self, method: str, params: dict) -> Any:
        rid = self._next_id
        self._next_id += 1
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        await self._ws.send(json.dumps({"type": "rpc", "id": rid, "method": method, "params": params}))
        return await fut

    def _rpc(self, method: str, **params: Any) -> Any:
        """Issue one RPC from the sync caller thread and block on its reply."""
        self._ensure_connected()
        assert self._loop is not None  # set by _ensure_connected
        cf = asyncio.run_coroutine_threadsafe(self._send_rpc(method, params), self._loop)
        return cf.result(self._timeout)

    def close(self) -> None:
        """Close the socket and tear down the loop thread. Idempotent."""
        loop = self._loop
        if loop is None:
            return
        self._loop = None

        async def _shutdown():
            if self._ws is not None:
                await self._ws.close()

        try:
            asyncio.run_coroutine_threadsafe(_shutdown(), loop).result(self._timeout)
        except Exception:
            pass  # closing a dead socket is fine — we only want the loop stopped
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=self._timeout)
        loop.close()

    # ------------------------------------------------------------------
    # Discovery / Read
    # ------------------------------------------------------------------

    def _path_arg(self, ref: NoteRef | str) -> str:
        """Vault-relative path for a `path` RPC param.

        A NoteRef carries its path; a bare name is resolved through the graph
        index (same name→path lookup the fs backend does), falling back to
        `<name>.md` so a not-yet-indexed note still round-trips.
        """
        if isinstance(ref, NoteRef):
            return ref.path or f"{ref.name}.md"
        if ref.endswith(".md") or "/" in ref:
            return ref
        self._ensure_graph()
        matched = self._notes_by_name.get(ref.lower(), [])
        return matched[0].path if matched else f"{ref}.md"

    def read_note(self, ref: NoteRef | str) -> NoteContent:
        path = self._path_arg(ref)
        data = self._rpc("read", path=path)
        name = path.rsplit("/", 1)[-1].removesuffix(".md")
        return NoteContent(
            ref=NoteRef(name=name, path=data.get("path", path)),
            content=data.get("content", ""),
            size=data.get("size", 0),
        )

    def list_files(self, folder: str = "") -> list[NoteRef]:
        rows = self._rpc("list_files", folder=folder)
        return [NoteRef(name=r["name"], path=r["path"]) for r in (rows or [])]

    def list_inbox_files(self) -> list[NoteRef]:
        from silica.config import CONFIG
        if not CONFIG.inbox_dir:
            return []
        return self.list_files(folder=CONFIG.inbox_dir)

    def search_names(self, query: str) -> list[NoteRef]:
        q = query.lower()
        return [r for r in self.list_files() if q in r.name.lower()]

    def props_of(self, ref: NoteRef | str) -> dict:
        data = self._rpc("props_of", path=self._path_arg(ref))
        return data if isinstance(data, dict) else {}

    def outline(self, ref: NoteRef | str) -> list[Heading]:
        rows = self._rpc("outline", path=self._path_arg(ref))
        return [
            Heading(level=h.get("level", 1), text=str(h.get("text", "")), position=h.get("position", 0))
            for h in (rows or [])
        ]

    def search_context(self, query: str) -> list[Hit]:
        return self._hits_from_groups(self._rpc("search_context", query=query))

    def search_context_batch(self, queries: list[str]) -> dict[str, list[Hit]]:
        if not queries:
            return {}
        data = self._rpc("search_context_batch", queries=queries) or {}
        return {q: self._hits_from_groups(data.get(q, [])) for q in queries}

    @staticmethod
    def _hits_from_groups(groups) -> list[Hit]:
        """Flatten PROTOCOL's `[{path, name, matches:[{line, content}]}]` into Hits."""
        hits: list[Hit] = []
        for g in (groups or []):
            ref = NoteRef(name=g.get("name", ""), path=g.get("path", ""))
            for m in g.get("matches", []):
                hits.append(Hit(ref=ref, line=m.get("line", 0), snippet=str(m.get("content", ""))))
        return hits

    # ------------------------------------------------------------------
    # Graph — built from two bulk RPCs, read from the in-memory index
    # ------------------------------------------------------------------

    def _ensure_graph(self) -> None:
        """Populate the in-memory index from `resolved_links` + `mention_index`.

        Mirrors cli_backend._load_graph_from_obsidian: one bulk edge map and one
        bulk mention map replace N per-note round-trips. `list_files` seeds the
        node set so orphans (no links either way) are still present.
        """
        if self._is_graph_built:
            return
        self._graph.clear()
        self._unresolved_links.clear()
        self._notes.clear()
        self._notes_by_name.clear()

        for ref in self.list_files():
            self._notes[ref.path] = ref
            self._graph.add_node(ref.path, ref=ref)
            self._notes_by_name.setdefault(ref.name.lower(), []).append(ref)

        data = self._rpc("resolved_links") or {}
        for source, targets in (data.get("resolved") or {}).items():
            if source not in self._notes:
                continue
            for target in targets:
                if target in self._notes:
                    self._graph.add_edge(source, target)
        for source, targets in (data.get("unresolved") or {}).items():
            if source not in self._notes:
                continue
            for target in targets:
                self._unresolved_links.add((source, target))

        titles = [ref.name.lower() for ref in self._notes.values() if len(ref.name) >= 2]
        mentions = self._rpc("mention_index", titles=titles) or {}
        self._mention_index.clear()
        for title_lower, paths in mentions.items():
            if isinstance(paths, list):
                self._mention_index[title_lower] = set(paths)

        self._is_graph_built = True

    def _path_of(self, ref: NoteRef | str) -> str | None:
        if isinstance(ref, NoteRef):
            return ref.path or None
        if ref.endswith(".md"):
            return ref
        matched = self._notes_by_name.get(ref.lower(), [])
        return matched[0].path if matched else None

    def links(self, ref: NoteRef | str) -> list[NoteRef]:
        self._ensure_graph()
        path = self._path_of(ref)
        if not path:
            return []
        results = [self._node_ref(t) for t in self._graph.successors(path)] if path in self._graph else []
        for s, t in self._unresolved_links:
            if s == path:
                name = t.rsplit("/", 1)[-1].removesuffix(".md")
                results.append(NoteRef(name=name, path=f"{name}.md"))
        return results

    def backlinks(self, ref: NoteRef | str) -> list[NoteRef]:
        self._ensure_graph()
        path = self._path_of(ref)
        if not path or path not in self._graph:
            return []
        return [self._node_ref(s) for s in self._graph.predecessors(path)]

    def orphans(self) -> list[NoteRef]:
        self._ensure_graph()
        return [self._graph.nodes[n]["ref"] for n, d in self._graph.in_degree() if d == 0]

    def unresolved(self) -> list[Link]:
        self._ensure_graph()
        return [Link(source=self._node_ref(s), target=t.removesuffix(".md"))
                for s, t in self._unresolved_links]

    def graph_snapshot(self, refs: list[NoteRef] | None = None) -> GraphSnapshot:
        self._ensure_graph()
        neighborhood = None
        if refs is not None:
            neighborhood = set()
            for r in refs:
                if not r.path:
                    continue
                neighborhood.add(r.path)
                if r.path in self._graph:
                    neighborhood.update(self._graph.successors(r.path))
                    neighborhood.update(self._graph.predecessors(r.path))

        paths = self._notes.keys() if neighborhood is None else neighborhood
        link_counts: dict[str, int] = {}
        backlink_counts: dict[str, int] = {}
        for path in paths:
            if path not in self._notes:
                continue
            resolved = self._graph.out_degree(path) if path in self._graph else 0
            unresolved = sum(1 for s, _t in self._unresolved_links if s == path)
            key = path.removesuffix(".md")
            link_counts[key] = resolved + unresolved
            backlink_counts[key] = self._graph.in_degree(path) if path in self._graph else 0

        orphans = [self._notes[p] for p in paths
                   if p in self._notes and (p not in self._graph or self._graph.in_degree(p) == 0)]
        unresolved_links = [Link(source=self._node_ref(s), target=t.removesuffix(".md"))
                            for s, t in self._unresolved_links
                            if s in self._notes and (neighborhood is None or s in neighborhood)]
        return GraphSnapshot(
            orphans=orphans, unresolved=unresolved_links,
            link_counts=link_counts, backlink_counts=backlink_counts,
        )

    # ------------------------------------------------------------------
    # Write (graph-safe) — one RPC each; the reply IS the settle (§2.4)
    # ------------------------------------------------------------------

    def create(self, path: str, content: str) -> NoteRef:
        data = self._rpc("create", path=path, content=content)
        ref = NoteRef(name=data["name"], path=data["path"])
        self._patch_graph_add(ref.path, ref, content)
        return ref

    def overwrite(self, path: str, content: str) -> NoteRef:
        self._rpc("overwrite", path=path, content=content)
        name = path.rsplit("/", 1)[-1].removesuffix(".md")
        ref = NoteRef(name=name, path=path)
        self._patch_graph_add(path, ref, content)
        return ref

    def append(self, ref: NoteRef | str, content: str) -> None:
        path = self._path_arg(ref)
        self._rpc("append", path=path, content=content)
        # Optimistic patch — add any new links introduced by the fragment.
        if self._is_graph_built and path in self._graph:
            self._add_link_edges(path, content)

    def set_prop(self, ref: NoteRef | str, name: str, value: Any, type_: str = "text") -> None:
        self._rpc("set_prop", path=self._path_arg(ref), name=name, value=value, type=type_)

    def move(self, ref: NoteRef | str, to: str) -> None:
        self._rpc("move", path=self._path_arg(ref), to=to)
        # Obsidian rewrites incoming wikilinks on move; patching every referrer
        # over the wire isn't worth it — reinvalidate, rebuild lazily on next read.
        self._is_graph_built = False

    def delete(self, ref: NoteRef | str) -> None:
        path = self._path_arg(ref)
        self._rpc("delete", path=path)
        self._patch_graph_remove(path)

    def autolink_note(self, path: str, candidates: list[str] | None = None) -> list[str]:
        added = self._rpc("autolink_note", path=path, candidates=candidates) or []
        if added:
            self._is_graph_built = False  # body rewritten plugin-side; cheapest correct patch
        return list(added)

    # ------------------------------------------------------------------
    # In-memory index patching after writes (mirrors cli_backend)
    # ------------------------------------------------------------------

    def _patch_graph_add(self, path: str, ref: NoteRef, content: str) -> None:
        """Patch the index after create/overwrite so same-session reads stay correct.

        No-op if the graph was never built (nothing cached to patch).
        """
        if not self._is_graph_built:
            return
        if path in self._graph:
            self._graph.remove_edges_from(list(self._graph.out_edges(path)))
        self._unresolved_links = {(s, t) for s, t in self._unresolved_links if s != path}
        self._notes[path] = ref
        self._graph.add_node(path, ref=ref)
        by_name = self._notes_by_name.setdefault(ref.name.lower(), [])
        if ref not in by_name:
            by_name.append(ref)
        self._add_link_edges(path, content)
        # Mention index: rescan this body against all known titles. Existing
        # notes that mention the *new* title are not rescanned (same trade-off
        # as cli_backend — the bulk build captured session-start state).
        for paths_set in self._mention_index.values():
            paths_set.discard(path)
        trie = build_title_trie(self._notes_by_name)
        for title_lower in mentions_in(content.lower(), trie):
            self._mention_index.setdefault(title_lower, set()).add(path)

    def _patch_graph_remove(self, path: str) -> None:
        """Patch the index after a delete."""
        if not self._is_graph_built:
            return
        if path in self._graph:
            self._graph.remove_node(path)
        self._notes.pop(path, None)
        name_lower = path.rsplit("/", 1)[-1].removesuffix(".md").lower()
        if name_lower in self._notes_by_name:
            self._notes_by_name[name_lower] = [
                r for r in self._notes_by_name[name_lower] if r.path != path
            ]
        self._unresolved_links = {(s, t) for s, t in self._unresolved_links if s != path}
        for paths_set in self._mention_index.values():
            paths_set.discard(path)

    def _add_link_edges(self, path: str, content: str) -> None:
        for target in extract_links(content):
            resolved = self._resolve_link(target)
            if resolved is not None:
                self._graph.add_edge(path, resolved.path)
            else:
                self._unresolved_links.add((path, target))

    def _resolve_link(self, target: str) -> NoteRef | None:
        """Resolve a wikilink target against the index — by path, else by name."""
        t = target.removesuffix(".md")
        if "/" in t:
            exact = self._notes.get(t + ".md")
            if exact is not None:
                return exact
            suffix = "/" + t.lower() + ".md"
            matches = [r for p, r in self._notes.items() if p.lower().endswith(suffix)]
            return min(matches, key=lambda r: (r.path.count("/"), r.path.lower())) if matches else None
        matched = self._notes_by_name.get(t.lower(), [])
        return matched[0] if matched else None

    # ------------------------------------------------------------------
    # Transactionality — content snapshots (no history over the bridge)
    # ------------------------------------------------------------------

    def snapshot_versions(self, refs: list[NoteRef]) -> Txn:
        """Capture each ref's current body; restore() overwrites it back."""
        inverses = []
        for ref in refs:
            path = ref.path or f"{ref.name}.md"
            try:
                content = self.read_note(ref).content
            except Exception as e:
                logger.warning("snapshot: could not read %s: %s", path, e)
                continue
            inverses.append(InverseOp(
                kind=InverseOpKind.restore_version, path=path, prior_content=content,
            ))
        return Txn(id=f"txn_ws_{int(time.time())}", refs=refs, inverses=inverses)

    def restore(self, txn: Txn) -> None:
        """Rollback: overwrite snapshotted bodies back, delete created notes."""
        if txn.versions:
            logger.warning("WS backend has no note history; ignoring versions, "
                           "restoring from captured content instead.")
        for inv in txn.inverses:
            kind = getattr(inv, "kind", None)
            prior = getattr(inv, "prior_content", None)
            if kind == InverseOpKind.restore_version and prior is not None:
                self.overwrite(inv.path, prior)
        for path in txn.created_paths:
            try:
                self.delete(path)
            except RuntimeError as e:
                if "not found" in str(e).lower():
                    logger.info("restore: created note %s already absent", path)
                else:
                    raise
