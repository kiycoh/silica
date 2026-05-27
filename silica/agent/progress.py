from __future__ import annotations
import re
import json
import logging
from rich.live import Live
from rich.spinner import Spinner
from rich.panel import Panel
from rich.text import Text
from rich.markup import escape
from silica.agent.events import (
    ToolStartEvent,
    ToolCompleteEvent,
    ToolErrorEvent,
    ReasoningEvent,
    RenderEvent,
    ThinkingStartEvent,
    ThinkingEndEvent,
)
from silica.config import CONFIG
from silica.ui.console import CONSOLE

logger = logging.getLogger(__name__)

_MAX_RESULT_CHARS = 600
_MAX_RESULT_LINES = 12
_MAX_ARGS_PREVIEW_CHARS = 120
_REASONING_MAX_LINES = 20

_REDACT_PATTERNS = [
    re.compile(r'(api_?key|token|secret|password|auth|bearer)\s*[=:]\s*\S+', re.I),
    re.compile(r'"(api_?key|token|secret|password)"\s*:\s*"[^"]*"', re.I),
]


def _redact(text: str) -> str | None:
    """Returns None if redaction fails (fail-closed)."""
    try:
        for pattern in _REDACT_PATTERNS:
            text = pattern.sub(r'\1=[REDACTED]', text)
        return text
    except Exception as exc:
        logger.debug("Redaction failed (omitting detail): %s", exc)
        return None


def _cap(text: str, max_chars: int = _MAX_RESULT_CHARS, max_lines: int = _MAX_RESULT_LINES) -> str:
    lines = text.splitlines()
    if len(lines) > max_lines:
        omitted = len(lines) - max_lines
        tail = "\n".join(lines[-max_lines:])
        text = f"[… {omitted} lines omitted]\n{tail}"
    if len(text) > max_chars:
        omitted_chars = len(text) - max_chars
        text = f"[… {omitted_chars} chars omitted]\n{text[-max_chars:]}"
    return text


def _head_cap(text: str, max_lines: int = _REASONING_MAX_LINES) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    extra = len(lines) - max_lines
    return "\n".join(lines[:max_lines]) + f"\n[… +{extra} lines]"


def _args_preview(args: dict) -> str:
    try:
        s = json.dumps(args, ensure_ascii=False)
        if len(s) > _MAX_ARGS_PREVIEW_CHARS:
            return s[:_MAX_ARGS_PREVIEW_CHARS] + "…"
        return s
    except Exception:
        return "{…}"


class _ProgressRenderer:
    def __init__(self) -> None:
        self._live: Live | None = None
        self._last_tool_name: str = ""

    def _start_spinner(self) -> None:
        if self._live is not None:
            return
        spinner = Spinner("dots", text="  thinking…", style="dim cyan")
        self._live = Live(
            spinner,
            console=CONSOLE,
            refresh_per_second=12,
            transient=True,
        )
        self._live.start()

    def _stop_spinner(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def __call__(self, event: RenderEvent) -> None:
        try:
            self._dispatch(event)
        except Exception as exc:
            logger.debug("tool_progress_callback error (swallowed): %s", exc)

    def _dispatch(self, event: RenderEvent) -> None:
        mode = CONFIG.tool_progress

        if isinstance(event, ThinkingStartEvent):
            if mode != "off":
                self._start_spinner()
            return

        if isinstance(event, ThinkingEndEvent):
            self._stop_spinner()
            return

        if isinstance(event, ReasoningEvent):
            self._stop_spinner()
            if CONFIG.show_thinking or mode == "verbose" or CONFIG.verbose:
                body = Text(_head_cap(event.text).strip(), style="dim")
                CONSOLE.print(Panel(
                    body,
                    title="[dim]✦ thinking[/]",
                    title_align="left",
                    border_style="dim cyan",
                    padding=(0, 1),
                ))
            return

        if isinstance(event, ToolErrorEvent):
            self._stop_spinner()
            CONSOLE.print(f"  [bold red]✗ {escape(event.name)}[/]: [red]{escape(event.error)}[/]")
            return

        if mode == "off":
            return

        if isinstance(event, ToolStartEvent):
            if mode == "new":
                if event.name == self._last_tool_name:
                    return
                self._last_tool_name = event.name
                CONSOLE.print(f"  [dim]⚙ {escape(event.name)}[/]")

            elif mode == "all":
                preview = _args_preview(event.args)
                CONSOLE.print(f"  [cyan]→ [bold]{escape(event.name)}[/bold]({escape(preview)})[/]")

            elif mode == "verbose":
                try:
                    args_json = json.dumps(event.args, indent=2, ensure_ascii=False)
                except Exception:
                    args_json = str(event.args)
                redacted = _redact(args_json)
                if redacted is not None:
                    CONSOLE.print(f"  [cyan]→ [bold]{escape(event.name)}[/bold][/]")
                    CONSOLE.print(f"  [dim]args: {escape(_cap(redacted, max_lines=6))}[/]")
                else:
                    CONSOLE.print(f"  [cyan]→ [bold]{escape(event.name)}[/bold] [dim][redacted args][/][/]")

        elif isinstance(event, ToolCompleteEvent):
            dur = f"{event.duration_s:.3f}s"
            if mode in ("new", "all"):
                CONSOLE.print(f"  [green]✓ {escape(event.name)}[/] [dim]({dur})[/]")

            elif mode == "verbose":
                redacted = _redact(event.result)
                if redacted is not None:
                    capped = _cap(redacted)
                    CONSOLE.print(f"  [green]✓ [bold]{escape(event.name)}[/bold][/] [dim]({dur})[/]")
                    CONSOLE.print(f"  [dim]result: {escape(capped)}[/]")
                else:
                    CONSOLE.print(
                        f"  [green]✓ [bold]{escape(event.name)}[/bold][/]"
                        f" [dim]({dur}) [result redacted][/]"
                    )


def make_progress_callback() -> _ProgressRenderer:
    return _ProgressRenderer()
