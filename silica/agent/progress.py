from __future__ import annotations
import re
import json
import logging
import time
from typing import Callable
from rich.live import Live
from rich.spinner import Spinner
from rich.panel import Panel
from rich.text import Text
from rich.markup import escape
from rich.console import Group
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

# Module-level hook for pipeline phase events emitted by InjectorFSM.
# Set by _ProgressRenderer when silica_run_injector starts; cleared on completion.
_pipeline_phase_hook: Callable[[str, str, float | None], None] | None = None


def _set_pipeline_hook(hook: Callable[[str, str, float | None], None] | None) -> None:
    global _pipeline_phase_hook
    _pipeline_phase_hook = hook


def emit_pipeline_phase(phase: str, status: str, elapsed: float | None = None) -> None:
    """Called by InjectorFSM to surface phase transitions into the TUI. No-op if not registered."""
    if _pipeline_phase_hook is not None:
        try:
            _pipeline_phase_hook(phase, status, elapsed)
        except Exception:
            pass

_MAX_RESULT_CHARS = 600
_MAX_RESULT_LINES = 12
_MAX_ARGS_PREVIEW_CHARS = 120
_REASONING_MAX_LINES = 20
_COMPACT_VAL_CHARS = 60
_RESULT_HEAD_LINES = 3
_RESULT_LINE_CHARS = 120

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


def _compact_args(args: dict) -> str:
    """Format args as space-separated key=val pairs with long values truncated."""
    parts = []
    for k, v in args.items():
        if isinstance(v, str):
            val = v if len(v) <= _COMPACT_VAL_CHARS else v[:_COMPACT_VAL_CHARS] + "…"
        elif isinstance(v, (list, dict)):
            try:
                s = json.dumps(v, ensure_ascii=False)
            except Exception:
                s = str(v)
            val = s if len(s) <= _COMPACT_VAL_CHARS else s[:_COMPACT_VAL_CHARS] + "…"
        else:
            val = str(v)
        parts.append(f"{k}={val}")
    return "  ".join(parts)


def _head_result(text: str) -> str:
    """Return first N lines of a result, each line length-capped, with overflow count."""
    lines = text.splitlines()
    extra = max(0, len(lines) - _RESULT_HEAD_LINES)
    shown = [
        ln if len(ln) <= _RESULT_LINE_CHARS else ln[:_RESULT_LINE_CHARS] + "…"
        for ln in lines[:_RESULT_HEAD_LINES]
    ]
    result = "\n".join(shown)
    if extra:
        result += f"\n(+{extra} more lines)"
    return result


def _synthetic_tool_desc(name: str, args: dict) -> str:
    # Human-readable synthetic description
    if name == "silica_search" or name == "silica_search_context":
        return f"Searching notes for [bold]\"{escape(args.get('query', ''))}\"[/bold]"
    elif name == "silica_read_note":
        return f"Reading note [bold]\"{escape(args.get('name', ''))}\"[/bold]"
    elif name == "silica_props":
        return f"Reading properties of [bold]\"{escape(args.get('name', ''))}\"[/bold]"
    elif name == "silica_outline":
        return f"Reading outline of [bold]\"{escape(args.get('name', ''))}\"[/bold]"
    elif name == "silica_links":
        return f"Reading links of [bold]\"{escape(args.get('name', ''))}\"[/bold]"
    elif name == "silica_backlinks":
        return f"Reading backlinks of [bold]\"{escape(args.get('name', ''))}\"[/bold]"
    elif name == "silica_orphans":
        return "Listing orphan notes"
    elif name == "silica_unresolved":
        return "Listing unresolved links"
    elif name == "silica_files":
        folder = args.get("folder", "")
        folder_str = f" in [bold]\"{escape(folder)}\"[/bold]" if folder else ""
        return f"Listing files{folder_str}"
    elif name == "silica_exists":
        return f"Checking if [bold]\"{escape(args.get('path', ''))}\"[/bold] exists"
    elif name == "silica_deferred_list":
        return "Listing deferred operations"
    elif name == "silica_deferred_flush":
        return "Flushing deferred operations"
    elif name == "silica_inbox_ls":
        return "Listing inbox files"
    elif name == "silica_recon":
        return f"Reconciling [bold]\"{escape(args.get('inbox_file', ''))}\"[/bold]"
    elif name == "silica_payload":
        return f"Building concept payload from [bold]\"{escape(args.get('recon_report_path', ''))}\"[/bold]"
    elif name == "silica_sanitize":
        return f"Sanitizing distiller output at [bold]\"{escape(args.get('distiller_output_path', ''))}\"[/bold]"
    elif name == "silica_validate_ops":
        return f"Validating operations in [bold]\"{escape(args.get('ops_json_path', ''))}\"[/bold]"
    elif name == "silica_bulk_write":
        return f"Executing bulk write using [bold]\"{escape(args.get('ops_json_path', ''))}\"[/bold]"
    elif name == "silica_lint":
        return f"Linting note [bold]\"{escape(args.get('note_name', ''))}\"[/bold]"
    elif name == "silica_run_injector":
        return f"Running injector on [bold]\"{escape(args.get('inbox_file', ''))}\"[/bold]"
    elif name == "silica_deferred_retry":
        return "Retrying deferred operations"
    elif name == "silica_move":
        ref = args.get("ref", "")
        to = args.get("to", "")
        return f"Moving [bold]\"{escape(ref)}\"[/bold] to [bold]\"{escape(to)}\"[/bold]"
    elif name == "silica_delete":
        return f"Deleting [bold]\"{escape(args.get('ref', ''))}\"[/bold]"
    elif name == "silica_snapshot":
        return f"Taking snapshot to [bold]\"{escape(args.get('ops_json_path', ''))}\"[/bold]"
    elif name == "silica_restore":
        return f"Restoring transaction [bold]\"{escape(args.get('txn_id', ''))}\"[/bold]"
    elif name == "silica_cleanup":
        return f"Cleaning up [bold]\"{escape(args.get('inbox_file', ''))}\"[/bold]"
        
    # Fallback
    clean_name = name.removeprefix("silica_").replace("_", " ")
    return f"Executing {clean_name}"


_PHASE_LABELS: dict[str, str] = {
    "recon":      "recon",
    "crossdedup": "cross-dedup",
    "payload":    "payload",
    "salience":   "salience",
    "collision":  "collision",
    "distill":    "distill",
    "sanitize":   "sanitize",
    "validate":   "validate",
    "snapshot":   "snapshot",
    "write":      "write",
    "hub_update": "hub-update",
    "autolink":   "autolink",
    "backlink":   "backlink",
    "lint":       "lint",
    "cleanup":    "cleanup",
    "rollback":   "rollback",
}


class _ProgressRenderer:
    def __init__(self) -> None:
        self._live: Live | None = None
        self._last_tool_name: str = ""
        self._active_tools: dict[str, dict] = {}
        # Pipeline phase tracking (populated when silica_run_injector is active)
        self._injector_call_id: str | None = None
        self._injector_desc: str = ""
        self._pipeline_phases: list[dict] = []   # ordered: {phase, status, elapsed}
        self._phase_start_times: dict[str, float] = {}

    def _start_spinner(self) -> None:
        if self._live is not None:
            return
        if not CONSOLE.is_terminal:
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

    def _on_pipeline_phase(self, phase: str, status: str, elapsed: float | None) -> None:
        """Callback registered as the global pipeline hook while injector runs."""
        label = _PHASE_LABELS.get(phase, phase)
        if status == "running":
            self._phase_start_times[phase] = time.monotonic()
            for entry in self._pipeline_phases:
                if entry["phase"] == label:
                    entry["status"] = "running"
                    entry["elapsed"] = None
                    break
            else:
                self._pipeline_phases.append({"phase": label, "status": "running", "elapsed": None})
        elif status in ("done", "failed"):
            start = self._phase_start_times.pop(phase, time.monotonic())
            dur = elapsed if elapsed is not None else (time.monotonic() - start)
            for entry in self._pipeline_phases:
                if entry["phase"] == label:
                    entry["status"] = status
                    entry["elapsed"] = dur
                    break
            else:
                self._pipeline_phases.append({"phase": label, "status": status, "elapsed": dur})
        self._update_live()

    def _update_live(self) -> None:
        mode = CONFIG.tool_progress
        if mode == "off" or not CONSOLE.is_terminal:
            return

        if not self._active_tools:
            if self._live:
                self._live.update(Spinner("dots", text="  thinking…", style="dim cyan"))
            return

        renderables = []
        for cid, tinfo in self._active_tools.items():
            name = tinfo["name"]
            args = tinfo["args"]
            desc = _synthetic_tool_desc(name, args)

            if name == "silica_run_injector" and self._pipeline_phases:
                # Injector: show a nested subprocess panel
                phase_lines: list[str] = []
                for entry in self._pipeline_phases:
                    p_label = entry["phase"]
                    p_status = entry["status"]
                    p_elapsed = entry["elapsed"]
                    if p_status == "running":
                        phase_lines.append(f"    [dim]⠿[/] [cyan]{p_label}[/]…")
                    elif p_status == "done":
                        dur_str = f"[dim]{p_elapsed:.2f}s[/]" if p_elapsed is not None else ""
                        phase_lines.append(f"    [tool.ok]✓[/] [dim]{p_label}[/] {dur_str}")
                    elif p_status == "failed":
                        phase_lines.append(f"    [tool.err]✗[/] [dim red]{p_label}[/]")
                header = Spinner("dots", text=f"  [cyan]Running[/] {desc}…", style="cyan")
                phases_text = Text.from_markup("\n".join(phase_lines)) if phase_lines else None
                if phases_text:
                    renderables.append(Group(header, phases_text))
                else:
                    renderables.append(header)
            else:
                tool_spinner = Spinner("dots", text=f"  [cyan]Running[/] {desc}…", style="cyan")
                renderables.append(tool_spinner)

        if self._live:
            self._live.update(Group(*renderables))
        else:
            self._live = Live(
                Group(*renderables),
                console=CONSOLE,
                refresh_per_second=12,
                transient=True,
            )
            self._live.start()

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
                body = _head_cap(event.text).strip()
                indented = "\n".join(f"  [reasoning.gutter]│[/] [dim]{line}[/]" for line in body.splitlines())
                CONSOLE.print(f"  [reasoning]✦ thinking[/]\n{indented}\n")
            return

        if isinstance(event, ToolErrorEvent):
            self._stop_spinner()
            CONSOLE.print(f"  [tool.err]✗[/] [bold]{escape(event.name)}[/]: [tool.err]{escape(event.error)}[/]")
            self._active_tools.pop(event.call_id, None)
            self._update_live()
            return

        if mode == "off":
            return

        if isinstance(event, ToolStartEvent):
            if CONSOLE.is_terminal:
                self._active_tools[event.call_id] = {"name": event.name, "args": event.args}
                if event.name == "silica_run_injector":
                    self._injector_call_id = event.call_id
                    self._injector_desc = _synthetic_tool_desc(event.name, event.args)
                    self._pipeline_phases = []
                    self._phase_start_times = {}
                    _set_pipeline_hook(self._on_pipeline_phase)
                self._update_live()
            else:
                # Non-interactive fallback: print immediately
                desc = _synthetic_tool_desc(event.name, event.args)
                if mode == "new":
                    if event.name == self._last_tool_name:
                        return
                    self._last_tool_name = event.name
                    CONSOLE.print(f"  [dim]⚙[/] {desc}")
                elif mode == "all":
                    CONSOLE.print(f"  [cyan]→[/] {desc}")
                elif mode == "verbose":
                    CONSOLE.print(f"  [cyan]→[/] {desc}")

        elif isinstance(event, ToolCompleteEvent):
            dur = f"{event.duration_s:.3f}s"
            if CONSOLE.is_terminal:
                self._stop_spinner()
                desc = _synthetic_tool_desc(event.name, event.args)

                if event.name == "silica_run_injector" and self._injector_call_id == event.call_id:
                    # Deregister pipeline hook and print a compact subprocess summary
                    _set_pipeline_hook(None)
                    self._injector_call_id = None
                    CONSOLE.print(f"  [tool.ok]✓[/] {desc} [dim]({dur})[/]")
                    for entry in self._pipeline_phases:
                        p_label = entry["phase"]
                        p_elapsed = entry["elapsed"]
                        p_status = entry["status"]
                        if p_status == "done":
                            dur_str = f"[dim]{p_elapsed:.2f}s[/]" if p_elapsed is not None else ""
                            CONSOLE.print(f"    [tool.ok]✓[/] [dim]{p_label}[/] {dur_str}")
                        elif p_status == "failed":
                            CONSOLE.print(f"    [tool.err]✗[/] [dim red]{p_label}[/]")
                    self._pipeline_phases = []
                else:
                    CONSOLE.print(f"  [tool.ok]✓[/] {desc} [dim]({dur})[/]")
                    if mode == "verbose":
                        redacted = _redact(event.result)
                        if redacted is not None:
                            head = _head_result(redacted.strip())
                            if head:
                                CONSOLE.print(f"    [dim]{escape(head)}[/]")
                        else:
                            CONSOLE.print(f"    [dim][result redacted][/]")

                self._active_tools.pop(event.call_id, None)
                self._update_live()
            else:
                # Non-interactive fallback
                desc = _synthetic_tool_desc(event.name, event.args)
                if mode in ("new", "all"):
                    CONSOLE.print(f"  [tool.ok]✓[/] {desc} [dim]({dur})[/]")
                elif mode == "verbose":
                    redacted = _redact(event.result)
                    if redacted is not None:
                        head = _head_result(redacted.strip())
                        CONSOLE.print(f"  [tool.ok]✓[/] {desc} [dim]({dur})[/]")
                        if head:
                            CONSOLE.print(f"  [dim]{escape(head)}[/]")
                    else:
                        CONSOLE.print(f"  [tool.ok]✓[/] {desc} [dim]({dur}) [result redacted][/]")


    def close(self) -> None:
        """Unconditionally stop the live display and deregister the pipeline hook.

        Called on KeyboardInterrupt / uncaught exceptions so the terminal is
        always restored before the next prompt is printed.
        """
        _set_pipeline_hook(None)
        self._stop_spinner()


def make_progress_callback() -> _ProgressRenderer:
    return _ProgressRenderer()
