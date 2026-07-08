"""`silica connect` — host the WS bridge server the Obsidian plugin dials into.

PROTOCOL.md's server half. One loopback socket carries both channels: the
plugin's chat turns stream through the transport-neutral `run_turn` (framed as
`chat_event*` + one `chat_done`/`chat_error`), while DRIVER `rpc` frames
interleave on the same connection. On handshake the accepted connection is
wrapped in an attached ObsidianWSBackend and installed as the global driver;
on drop the driver falls back to the configured local backend (cli/fs).

Only stdlib at import time — websockets is imported inside start()/run_connect
so the module loads without the [connect] extra.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from silica.config import CONFIG

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1


def _fallback_backend() -> str:
    """Local driver while no plugin is attached (PROTOCOL fallback chain):
    cli if the Obsidian CLI is on PATH, else headless fs."""
    return "cli" if shutil.which("obsidian") else "fs"


def _origin_ok(origin: str) -> bool:
    """Browsers set Origin; the plugin's native WebSocket does not. Any
    non-loopback page is refused (loopback port ≠ same-origin protection)."""
    if not origin:
        return True
    return urlsplit(origin).hostname in ("127.0.0.1", "localhost", "::1")


async def _send(ws: Any, frame: dict) -> None:
    await ws.send(json.dumps(frame))


class BridgeServer:
    """The ws://127.0.0.1 bridge server (PROTOCOL.md, server half)."""

    def __init__(self) -> None:
        self.port: int = CONFIG.ws_port  # 0 → OS picks; real port set by start()
        self.token: str = CONFIG.ws_token or secrets.token_hex(16)
        self._server: Any = None
        self._backend: Any = None  # the attached ws driver, while a plugin is connected
        self._bridge_file: Path | None = None

    async def start(self) -> None:
        from websockets.asyncio.server import serve

        self._server = await serve(self._handler, "127.0.0.1", self.port)
        self.port = self._server.sockets[0].getsockname()[1]
        self._bridge_file = self._write_bridge_file()
        logger.info("bridge: listening on ws://127.0.0.1:%d", self.port)

    async def stop(self) -> None:
        self._uninstall(self._backend)
        if self._bridge_file is not None:
            self._bridge_file.unlink(missing_ok=True)
            self._bridge_file = None
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    def _write_bridge_file(self) -> Path:
        """Discovery file the plugin reads to find port + token (mode 0600)."""
        path = Path(CONFIG.vault_path) / ".obsidian" / "silica-bridge.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "port": self.port, "token": self.token,
            "pid": os.getpid(), "protocolVersion": PROTOCOL_VERSION,
        }), encoding="utf-8")
        path.chmod(0o600)
        return path

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def _handler(self, ws: Any) -> None:
        if not await self._handshake(ws):
            return
        backend = self._install(ws)
        await _send(ws, {
            "type": "welcome", "vault": Path(CONFIG.vault_path).name,
            "obsidianVersion": "", "protocolVersion": PROTOCOL_VERSION,
        })
        try:
            async for raw in ws:
                try:
                    frame = json.loads(raw)
                except ValueError:
                    logger.warning("bridge: non-JSON frame ignored")
                    continue
                await self._route(ws, frame, backend)
        finally:
            self._uninstall(backend)

    async def _handshake(self, ws: Any) -> bool:
        hello = json.loads(await ws.recv())
        origin = ws.request.headers.get("Origin", "") if ws.request is not None else ""
        if not _origin_ok(origin):
            reason = "non-loopback origin refused"
        elif hello.get("token") != self.token:
            reason = "bad token"
        elif hello.get("protocolVersion") != PROTOCOL_VERSION:
            reason = f"protocol mismatch: {hello.get('protocolVersion')!r}"
        else:
            return True
        logger.warning("bridge: handshake refused (%s)", reason)
        await _send(ws, {"type": "bye", "reason": reason})
        await ws.close()
        return False

    def _install(self, ws: Any) -> Any:
        from silica.driver import set_driver
        from silica.driver.ws_backend import ObsidianWSBackend

        backend = ObsidianWSBackend.attached(ws, asyncio.get_running_loop())
        self._backend = backend
        set_driver(backend)
        logger.info("bridge: plugin connected — ws driver installed")
        return backend

    def _uninstall(self, backend: Any) -> None:
        from silica.driver import set_driver

        if backend is None or backend is not self._backend:
            return  # a newer connection already took over
        self._backend = None
        backend.detach("plugin disconnected")
        set_driver(None)
        logger.info("bridge: plugin disconnected — driver falls back to %r", CONFIG.backend)

    # ------------------------------------------------------------------
    # Frame routing
    # ------------------------------------------------------------------

    async def _route(self, ws: Any, frame: dict, backend: Any) -> None:
        from silica.ui.web import server as web

        kind = frame.get("type")
        if kind in ("rpc_result", "rpc_error"):
            backend._on_frame(frame)
        elif kind == "chat":
            tid, text = str(frame.get("turnId", "")), str(frame.get("text", ""))
            if not web._begin_turn():
                await _send(ws, {"type": "chat_error", "turnId": tid,
                                 "error": "a turn is already in progress"})
                return
            asyncio.create_task(self._chat_turn(ws, tid, text))
        elif kind == "chat_cancel":
            if web.current_cancel is not None:
                web.current_cancel.set()
        elif kind == "event":
            logger.debug("bridge: metadata event: %s", frame)  # non-fatal, LINT audits later

    async def _chat_turn(self, ws: Any, tid: str, text: str) -> None:
        from silica.ui.web.server import run_turn

        try:
            async for item in run_turn(text):
                kind = item.get("type")
                if kind == "done":
                    await _send(ws, {"type": "chat_done", "turnId": tid,
                                     "answer": item.get("answer", ""),
                                     "html": item.get("html", "")})
                elif kind == "error":
                    await _send(ws, {"type": "chat_error", "turnId": tid,
                                     "error": item.get("error", "")})
                else:
                    await _send(ws, {"type": "chat_event", "turnId": tid, "event": item})
        except Exception as exc:  # socket died mid-turn; run_turn's finally cleans up
            logger.warning("bridge: chat turn aborted: %s", exc)


def run_connect() -> int:
    """`silica connect` entry: host the bridge until Ctrl-C. Returns exit code."""
    try:
        import websockets  # noqa: F401
    except ImportError:
        print("silica connect requires the [connect] extra: pip install 'silica[connect]'")
        return 1
    if not CONFIG.vault_path:
        print("silica connect needs a vault: set SILICA_VAULT or run inside a repo with .silica/")
        return 1
    if CONFIG.backend == "ws":
        CONFIG.backend = _fallback_backend()  # ws installs on dial-in, not via config
    logger.info("bridge: driver fallback while no plugin is attached: %r", CONFIG.backend)

    async def _main() -> None:
        server = BridgeServer()
        await server.start()
        print(f"silica connect — ws://127.0.0.1:{server.port} (Ctrl-C to stop)")
        try:
            await asyncio.get_running_loop().create_future()  # serve until interrupted
        finally:
            await server.stop()

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
    return 0
