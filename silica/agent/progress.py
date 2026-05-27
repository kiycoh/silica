from __future__ import annotations
import re
import json
import logging
from silica.agent.events import ToolStartEvent, ToolCompleteEvent, ToolErrorEvent, ToolProgressEvent, ReasoningEvent, RenderEvent
from silica.config import CONFIG
from silica.ui.console import CONSOLE
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)

# --- Costanti ---
_MAX_RESULT_CHARS = 600
_MAX_RESULT_LINES = 12
_MAX_ARGS_PREVIEW_CHARS = 120

# Redaction: pattern comuni per credenziali/segreti
_REDACT_PATTERNS = [
    re.compile(r'(api_?key|token|secret|password|auth|bearer)\s*[=:]\s*\S+', re.I),
    re.compile(r'"(api_?key|token|secret|password)"\s*:\s*"[^"]*"', re.I),
]

def _redact(text: str) -> str | None:
    """Restituisce None se la redaction fallisce (fail-closed)."""
    try:
        for pattern in _REDACT_PATTERNS:
            text = pattern.sub(r'\1=[REDACTED]', text)
        return text
    except Exception as exc:
        logger.debug("Redaction failed (omitting detail): %s", exc)
        return None

def _cap(text: str, max_chars: int = _MAX_RESULT_CHARS, max_lines: int = _MAX_RESULT_LINES) -> str:
    """Tail-truncation con header esplicito, identico al comportamento hermes."""
    lines = text.splitlines()
    if len(lines) > max_lines:
        omitted = len(lines) - max_lines
        tail = "\n".join(lines[-max_lines:])
        text = f"[… {omitted} righe omesse]\n{tail}"
    if len(text) > max_chars:
        omitted_chars = len(text) - max_chars
        text = f"[… {omitted_chars} chars omessi]\n{text[-max_chars:]}"
    return text

def _args_preview(args: dict) -> str:
    """One-liner preview degli args per mode 'all'."""
    try:
        s = json.dumps(args, ensure_ascii=False)
        if len(s) > _MAX_ARGS_PREVIEW_CHARS:
            return s[:_MAX_ARGS_PREVIEW_CHARS] + "…"
        return s
    except Exception:
        return "{…}"

def _safe_print(text: str) -> None:
    """Print con catch su OSError/ValueError (stdout rotto o tty sparito)."""
    try:
        print(text)
    except (OSError, ValueError):
        pass

# --- ANSI helpers ---
_DIM   = "\033[2m"
_CYAN  = "\033[36m"
_GREEN = "\033[32m"
_YELLOW= "\033[33m"
_RED   = "\033[31m"
_RESET = "\033[0m"
_BOLD  = "\033[1m"

_REASONING_MAX_LINES = 20

def _head_cap(text: str, max_lines: int = _REASONING_MAX_LINES) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    extra = len(lines) - max_lines
    return "\n".join(lines[:max_lines]) + f"\n[… +{extra} righe]"


def make_progress_callback():
    """Costruisce il callback adatto alla mode corrente in CONFIG.tool_progress."""
    # State per mode 'new': evita di ristampare lo stesso tool consecutivamente
    _last_tool_name: list[str] = [""]  # lista mutabile per closure

    def callback(event: RenderEvent) -> None:
        if isinstance(event, ReasoningEvent):
            if CONFIG.show_thinking or CONFIG.tool_progress == "verbose" or CONFIG.verbose:
                body = Text(_head_cap(event.text).strip(), style="dim")
                CONSOLE.print(Panel(body, title="thinking", title_align="left", border_style="dim", padding=(0, 1)))
            return

        if isinstance(event, ToolErrorEvent):
            # Errori sempre visibili indipendentemente dalla mode (force-show)
            _safe_print(f"  {_RED}✗ {event.name}: {event.error}{_RESET}")
            return

        current_mode = CONFIG.tool_progress  # rilegge live: /verbose può cambiare mid-session

        if current_mode == "off":
            return

        if isinstance(event, ToolStartEvent):
            if current_mode == "new":
                if event.name == _last_tool_name[0]:
                    return  # stessa tool, skip
                _last_tool_name[0] = event.name
                _safe_print(f"  {_DIM}⚙ {event.name}{_RESET}")

            elif current_mode == "all":
                preview = _args_preview(event.args)
                _safe_print(f"  {_CYAN}→ {_BOLD}{event.name}{_RESET}{_CYAN}({preview}){_RESET}")

            elif current_mode == "verbose":
                try:
                    args_json = json.dumps(event.args, indent=2, ensure_ascii=False)
                except Exception:
                    args_json = str(event.args)
                redacted = _redact(args_json)
                if redacted is not None:
                    _safe_print(f"  {_CYAN}→ {_BOLD}{event.name}{_RESET}")
                    _safe_print(f"  {_DIM}args: {_cap(redacted, max_lines=6)}{_RESET}")
                else:
                    _safe_print(f"  {_CYAN}→ {_BOLD}{event.name}{_RESET} {_DIM}[args redatti]{_RESET}")

        elif isinstance(event, ToolCompleteEvent):
            if current_mode in ("new", "all"):
                dur = f"{event.duration_s:.3f}s"
                _safe_print(f"  {_GREEN}✓ {event.name}{_RESET} {_DIM}({dur}){_RESET}")

            elif current_mode == "verbose":
                dur = f"{event.duration_s:.3f}s"
                redacted = _redact(event.result)
                if redacted is not None:
                    capped = _cap(redacted)
                    _safe_print(f"  {_GREEN}✓ {_BOLD}{event.name}{_RESET} {_DIM}({dur}){_RESET}")
                    _safe_print(f"  {_DIM}result: {capped}{_RESET}")
                else:
                    _safe_print(f"  {_GREEN}✓ {_BOLD}{event.name}{_RESET} {_DIM}({dur}) [result redatto]{_RESET}")


    return callback
