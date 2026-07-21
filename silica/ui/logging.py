# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

import io
import logging
import sys

from rich.console import Console

from silica.ui.console import CONSOLE
from silica.ui.style import GLYPHS
from silica.ui.theme import SILICA_THEME


class LiveAwareStreamHandler(logging.StreamHandler):
    """StreamHandler that resolves ``sys.stderr`` at emit time instead of caching it.

    A ``rich.live.Live`` redirects ``sys.stderr`` to a proxy that prints above the
    live region; a handler holding the original stream writes raw and tears the
    render. Reading ``sys.stderr`` dynamically lets the log flow through that proxy
    while a Live is active, and through the real stream otherwise.
    """

    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, _value):
        pass  # always dynamic — ignore the value StreamHandler.__init__ assigns

    def emit(self, record: logging.LogRecord) -> None:
        # Serialize the stderr write against CONSOLE's stdout flush. Without this,
        # a background-thread log (e.g. websocket keepalive) can land mid-way
        # through a main-thread `CONSOLE.print` and split the LLM answer panel.
        # rich holds `_lock` (an RLock) around its own buffer flush, so grabbing
        # the same lock forces our line fully before or after — never inside.
        # ponytail: couples to rich's private _lock; if rich drops it, fall back
        # to unsynchronized emit (the pre-existing race), never crash logging.
        lock = getattr(CONSOLE, "_lock", None)
        if lock is None:
            super().emit(record)
            return
        with lock:
            super().emit(record)


class HumanFriendlyFormatter(logging.Formatter):
    """Log formatter: timestamp + level icon in Rich markup, long messages truncated."""

    def __init__(self) -> None:
        super().__init__(datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        time_str = self.formatTime(record, self.datefmt)

        level = record.levelno
        if level == logging.DEBUG:
            icon = f"[muted]{GLYPHS['gear']}[/muted]"
        elif level == logging.INFO:
            icon = f"[tool.ok]{GLYPHS['info']}[/tool.ok]"
        elif level == logging.WARNING:
            icon = f"[warn]{GLYPHS['warn']}[/warn]"
        elif level >= logging.ERROR:
            icon = f"[tool.err]{GLYPHS['err']}[/tool.err]"
        else:
            icon = GLYPHS["bullet"]

        try:
            message = record.getMessage()
        except Exception as e:
            message = f"{record.msg} (args: {record.args}) [formatting error: {e}]"

        lines = message.split("\n")
        if len(lines) > 15:
            head = lines[:5]
            tail = lines[-5:]
            hidden = len(lines) - 10
            message = "\n".join(head + [f"  [dim]... ({hidden} lines truncated) ...[/dim]"] + tail)

        return f"  [muted][{time_str}][/muted] {icon} {message}"


class AnsiHumanFriendlyFormatter(HumanFriendlyFormatter):
    """HumanFriendlyFormatter for worker threads: renders the markup to ANSI here.

    The main-thread RichHandler renders markup through the shared CONSOLE; a
    worker thread can't — concurrent rich rendering onto an active Live tears the
    terminal (see ``_setup_logging``). So we render this one record to its own
    throwaway Console/buffer (nothing shared → thread-safe) and hand plain ANSI
    text to the StreamHandler, which the live-aware stderr proxy prints above the
    Live region intact. The buffer Console mirrors CONSOLE's terminal/colour, so
    piped output stays plain — exactly like the main-thread path.
    """

    def format(self, record: logging.LogRecord) -> str:
        markup = super().format(record)
        buf = io.StringIO()
        Console(
            file=buf,
            theme=SILICA_THEME,
            # highlight stays on: RichHandler applies its ReprHighlighter to
            # main-thread records, so worker records get the same treatment here.
            force_terminal=CONSOLE.is_terminal,
            color_system=CONSOLE.color_system,
            width=CONSOLE.width,
        ).print(markup, end="", soft_wrap=True)
        return buf.getvalue()
