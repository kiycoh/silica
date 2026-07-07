"""ObsidianWSBackend read path — driven against a Python stub WS server.

The stub is the machine-checkable contract the TS plugin must satisfy: it
speaks PROTOCOL.md's read RPCs, but its answers are sourced from the FS backend
over the synthetic vault, so it is a faithful oracle. A WS backend consuming it
must therefore match the FS backend read-for-read — that's the parity test.

Topology here is the inverse of production (unit 5: Python hosts, plugin dials);
the RPC channel is symmetric, so backend-as-client is the simplest unit-3 seam.
"""
from __future__ import annotations

import asyncio
import json
import threading

import pytest

pytest.importorskip("websockets")
from websockets.asyncio.server import serve  # noqa: E402

from silica.driver.base import NoteRef  # noqa: E402
from silica.driver.fs_backend import ObsidianFSBackend  # noqa: E402


# ---------------------------------------------------------------------------
# Stub WS server — PROTOCOL.md read RPCs, answered from an FS oracle.
# ---------------------------------------------------------------------------

class StubPluginServer:
    """A loopback WS server that answers read RPCs from a real FS vault.

    Stands in for the Obsidian plugin: validates the token on `hello`, replies
    `welcome`, then serves `rpc` frames from the FS backend so results carry the
    exact PROTOCOL.md shapes a correct plugin would return.
    """

    def __init__(self, vault_path, token: str = "test-token"):
        self.token = token
        self._fs = ObsidianFSBackend(vault_path=str(vault_path))
        self._fs._ensure_index()
        self.port: int | None = None
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(5):
            raise RuntimeError("stub server failed to start")

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)

    # -- server internals --------------------------------------------------

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._boot())
        self._loop.run_forever()

    async def _boot(self) -> None:
        self._server = await serve(self._handler, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        self._ready.set()

    async def _handler(self, ws) -> None:
        hello = json.loads(await ws.recv())
        if hello.get("token") != self.token or hello.get("protocolVersion") != 1:
            await ws.send(json.dumps({"type": "bye", "reason": "bad token or version"}))
            return
        await ws.send(json.dumps({
            "type": "welcome", "vault": "synthetic",
            "obsidianVersion": "stub", "protocolVersion": 1,
        }))
        async for raw in ws:
            frame = json.loads(raw)
            if frame.get("type") != "rpc":
                continue
            rid, method, params = frame.get("id"), frame.get("method"), frame.get("params") or {}
            try:
                result = self._dispatch(method, params)
            except Exception as exc:  # fatal op → rpc_error, driver raises
                await ws.send(json.dumps({"type": "rpc_error", "id": rid, "error": str(exc)}))
            else:
                await ws.send(json.dumps({"type": "rpc_result", "id": rid, "result": result}))

    def _dispatch(self, method: str, params: dict):
        fs = self._fs
        if method == "read":
            path = params["path"]
            nc = fs.read_note(NoteRef(name=path.rsplit("/", 1)[-1].removesuffix(".md"), path=path))
            return {"path": path, "content": nc.content, "size": nc.size}
        if method == "list_files":
            return [{"name": r.name, "path": r.path} for r in fs.list_files(params.get("folder", ""))]
        if method == "props_of":
            return fs.props_of(NoteRef(name="", path=params["path"]))
        if method == "outline":
            return [{"level": h.level, "text": h.text, "position": h.position}
                    for h in fs.outline(NoteRef(name="", path=params["path"]))]
        if method == "search_context":
            return self._grouped_search(params["query"])
        if method == "search_context_batch":
            return {q: self._grouped_search(q) for q in params["queries"]}
        if method == "resolved_links":
            resolved: dict = {}
            for s, t in fs._graph.edges():
                resolved.setdefault(s, {})[t] = 1
            unresolved: dict = {}
            for s, t in fs._unresolved_links:
                unresolved.setdefault(s, {})[t] = 1
            return {"resolved": resolved, "unresolved": unresolved}
        if method == "mention_index":
            wanted = {t.lower() for t in params["titles"]}
            return {tl: sorted(paths) for tl, paths in fs._mention_index.items() if tl in wanted}
        raise RuntimeError(f"unknown method: {method}")

    def _grouped_search(self, query: str) -> list[dict]:
        by_path: dict[str, dict] = {}
        for hit in self._fs.search_context(query):
            entry = by_path.setdefault(hit.ref.path, {"path": hit.ref.path, "name": hit.ref.name, "matches": []})
            entry["matches"].append({"line": hit.line, "content": hit.snippet})
        return list(by_path.values())


@pytest.fixture(scope="module")
def stub(synthetic_vault):
    srv = StubPluginServer(synthetic_vault)
    yield srv
    srv.stop()


@pytest.fixture
def ws_backend(stub):
    from silica.driver.ws_backend import ObsidianWSBackend

    be = ObsidianWSBackend(url=stub.url, token=stub.token)
    yield be
    be.close()


@pytest.fixture
def fs(synthetic_vault):
    """FS backend over the same vault — the parity oracle."""
    return ObsidianFSBackend(vault_path=str(synthetic_vault))


# ---------------------------------------------------------------------------
# Read-path behaviour — measured by parity with the FS backend on one vault.
# ---------------------------------------------------------------------------

def test_read_note_round_trips_content_over_the_socket(ws_backend, synthetic_vault):
    """A `read` RPC returns the note's verbatim content, correlated by id."""
    nc = ws_backend.read_note(NoteRef(name="Concepts", path="Hub/Concepts.md"))
    assert "central hub" in nc.content
    assert nc.size == len(nc.content)


def test_list_files_matches_fs(ws_backend, fs):
    assert {r.path for r in ws_backend.list_files()} == {r.path for r in fs.list_files()}


def test_search_names_matches_fs(ws_backend, fs):
    assert {r.path for r in ws_backend.search_names("cell")} == {r.path for r in fs.search_names("cell")}


def test_props_of_matches_fs(ws_backend, fs):
    assert ws_backend.props_of(NoteRef(name="Concepts", path="Hub/Concepts.md")) == \
        fs.props_of(NoteRef(name="Concepts", path="Hub/Concepts.md"))


def test_outline_matches_fs(ws_backend, fs):
    ref = NoteRef(name="Concepts", path="Hub/Concepts.md")
    assert [(h.level, h.text) for h in ws_backend.outline(ref)] == \
        [(h.level, h.text) for h in fs.outline(ref)]


def test_search_context_matches_fs(ws_backend, fs):
    ws_hits = {(h.ref.path, h.line, h.snippet) for h in ws_backend.search_context("hub")}
    fs_hits = {(h.ref.path, h.line, h.snippet) for h in fs.search_context("hub")}
    assert ws_hits == fs_hits


def test_search_context_batch_matches_fs(ws_backend, fs):
    ws_res = ws_backend.search_context_batch(["hub", "gradient"])
    fs_res = fs.search_context_batch(["hub", "gradient"])
    assert set(ws_res) == set(fs_res)
    for q in ws_res:
        assert {(h.ref.path, h.line) for h in ws_res[q]} == {(h.ref.path, h.line) for h in fs_res[q]}


# ---------------------------------------------------------------------------
# Graph path — _ensure_graph via resolved_links + mention_index, parity w/ FS.
# ---------------------------------------------------------------------------

_HUB = NoteRef(name="Concepts", path="Hub/Concepts.md")


def test_links_match_fs(ws_backend, fs):
    assert {r.name.lower() for r in ws_backend.links(_HUB)} == \
        {r.name.lower() for r in fs.links(_HUB)}


def test_backlinks_match_fs(ws_backend, fs):
    ref = NoteRef(name="Backpropagation", path="Concepts/Backpropagation.md")
    assert {r.path for r in ws_backend.backlinks(ref)} == {r.path for r in fs.backlinks(ref)}


def test_orphans_match_fs(ws_backend, fs):
    assert {r.path for r in ws_backend.orphans()} == {r.path for r in fs.orphans()}


def test_unresolved_match_fs(ws_backend, fs):
    assert {lnk.target.lower() for lnk in ws_backend.unresolved()} == \
        {lnk.target.lower() for lnk in fs.unresolved()}


def test_graph_snapshot_matches_fs(ws_backend, fs):
    ws_snap = ws_backend.graph_snapshot()
    fs_snap = fs.graph_snapshot()
    assert ws_snap.link_counts == fs_snap.link_counts
    assert ws_snap.backlink_counts == fs_snap.backlink_counts
    assert {r.path for r in ws_snap.orphans} == {r.path for r in fs_snap.orphans}


def test_mentions_of_matches_fs(ws_backend, fs):
    # Every title the FS backend indexes must resolve identically over the wire.
    for title in ("gradient", "perceptron", "concepts"):
        assert set(ws_backend.mentions_of(title)) == set(fs.mentions_of(title))


# ---------------------------------------------------------------------------
# Write path — deferred to unit 4; must fail loudly, not silently no-op or dial.
# ---------------------------------------------------------------------------

def test_write_methods_raise_not_implemented_without_touching_the_socket():
    from silica.driver.ws_backend import ObsidianWSBackend

    be = ObsidianWSBackend(url="ws://127.0.0.1:1", token="")  # dead port: must not dial
    writes = [
        lambda: be.create("A.md", "x"),
        lambda: be.overwrite("A.md", "x"),
        lambda: be.append("A.md", "x"),
        lambda: be.set_prop("A.md", "k", "v"),
        lambda: be.move("A.md", "B.md"),
        lambda: be.delete("A.md"),
        lambda: be.autolink_note("A.md"),
        lambda: be.snapshot_versions([]),
        lambda: be.restore(None),
    ]
    for call in writes:
        with pytest.raises(NotImplementedError):
            call()
