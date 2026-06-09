"""Web Clipper daemon — localhost HTTP server for external content ingestion.

ADR-0009 compliance:
  - Zero-trust ingress: all incoming content is sanitized before writing.
  - RBAC inbox-write only: writes to Inbox/ directory exclusively.
  - Bearer token auth: unauthenticated requests → 401.
  - No GitNexus, no external network calls inside Silica.

Usage:
    python -m silica.driver.clip_server --token <secret> --inbox Inbox --port 7357
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import PurePosixPath
from typing import Any

from silica.driver import DRIVER
from silica.kernel.sanitize import strip_degenerate_runs

logger = logging.getLogger(__name__)

# Filesystem-safe slug: keep letters, digits, spaces, hyphens, underscores
_UNSAFE_RE = re.compile(r'[^\w\s\-]', re.UNICODE)
_MULTI_SPACE_RE = re.compile(r'\s+')


@dataclass
class ClipConfig:
    token: str
    inbox_dir: str = "Inbox"
    port: int = 7357


def _safe_slug(title: str, max_len: int = 80) -> str:
    """Convert a title to a filesystem-safe slug. No path traversal."""
    slug = title.strip()
    slug = _UNSAFE_RE.sub("", slug)
    slug = _MULTI_SPACE_RE.sub(" ", slug).strip()
    slug = slug.replace(" ", "-")
    # Clamp length
    return slug[:max_len] if slug else ""


def _inbox_path(title: str, inbox_dir: str) -> str:
    """Build an inbox-scoped vault path. Ensures no path traversal."""
    slug = _safe_slug(title)
    if not slug:
        ts = int(time.time())
        slug = f"clip-{ts}"

    # Resolve against inbox_dir and ensure the result is inside it
    base = PurePosixPath(inbox_dir)
    candidate = PurePosixPath(inbox_dir) / f"{slug}.md"

    # PurePosixPath("..", "..") won't resolve to an absolute path, but we guard
    # against traversal by checking that candidate starts with inbox_dir.
    if not str(candidate).startswith(str(base)):
        ts = int(time.time())
        candidate = base / f"clip-{ts}.md"

    return str(candidate)


def clip_request(
    *,
    token: str,
    title: str,
    content: str,
    config: ClipConfig,
) -> dict[str, Any]:
    """Process a single clip request. Returns {status, ...}.

    Statuses:
        ok            — clip written to inbox
        unauthorized  — wrong token
        error         — validation failure (empty content, etc.)
    """
    if token != config.token:
        return {"status": "unauthorized"}

    if not content or not content.strip():
        return {"status": "error", "message": "content must not be empty"}

    sanitized = strip_degenerate_runs(content)
    path = _inbox_path(title, config.inbox_dir)

    note_content = f"# {title or 'Web Clip'}\n\n{sanitized}\n"
    DRIVER.create(path, note_content)
    logger.info("clip_server: wrote %s", path)
    return {"status": "ok", "path": path}


class ClipHandler(BaseHTTPRequestHandler):
    """HTTP handler for the clip server.

    Expects POST /clip with JSON body: {token, title, content}.
    """

    config: ClipConfig  # injected by make_server

    def do_POST(self) -> None:
        if self.path != "/clip":
            self._respond(404, {"status": "not_found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(length)
        try:
            body = json.loads(body_bytes)
        except Exception:
            self._respond(400, {"status": "error", "message": "invalid JSON"})
            return

        result = clip_request(
            token=body.get("token", ""),
            title=body.get("title", ""),
            content=body.get("content", ""),
            config=self.config,
        )

        if result["status"] == "unauthorized":
            self._respond(401, result)
        elif result["status"] == "error":
            self._respond(400, result)
        else:
            self._respond(200, result)

    def _respond(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("clip_server: " + fmt, *args)


def make_server(config: ClipConfig) -> ThreadingHTTPServer:
    """Build a ThreadingHTTPServer with the config injected into the handler."""

    class _Handler(ClipHandler):
        pass

    _Handler.config = config
    server = ThreadingHTTPServer(("127.0.0.1", config.port), _Handler)
    return server


def run(config: ClipConfig) -> None:
    server = make_server(config)
    logger.info("clip_server: listening on 127.0.0.1:%d", config.port)
    server.serve_forever()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Silica Web Clipper daemon")
    parser.add_argument("--token", required=True)
    parser.add_argument("--inbox", default="Inbox")
    parser.add_argument("--port", type=int, default=7357)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run(ClipConfig(token=args.token, inbox_dir=args.inbox, port=args.port))
