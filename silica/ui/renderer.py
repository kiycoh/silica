from __future__ import annotations
import re
import json
import logging
import time
from typing import Callable
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich.markup import escape
from rich.console import Group
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
from silica.agent.events import (
    ToolStartEvent,
    ToolCompleteEvent,
    ToolErrorEvent,
    ReasoningEvent,
    RenderEvent,
    ThinkingStartEvent,
    ThinkingEndEvent,
    LLMStreamEvent,
    BatchRunStartEvent,
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


_batch_run_hook: Callable[[RenderEvent], None] | None = None


def emit_batch_event(event: RenderEvent) -> None:
    """Called from cli.py to signal a batch run start. No-op if renderer not registered."""
    if _batch_run_hook is not None:
        try:
            _batch_run_hook(event)
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


_PHASE_ORDER: list[str] = list(_PHASE_LABELS.values())
_MICRO_PHASE_ORDER: tuple[str, ...] = ("reading", "calling_llm", "committing")


def _stage_track(pipeline_phases: list[dict], console_width: int) -> "Text":
    """Compact horizontal stage track for the injector panel.

    Shows a ±3 window around the running phase. Each phase is rendered as:
      done    → dim  "✓ label"
      running → bold #22d3ee "◉ label"
      failed  → bold red "✗ label"
      pending → dim  "· label"
    Leading/trailing "…" indicate clipped phases.
    """
    status_map: dict[str, str] = {e["phase"]: e["status"] for e in pipeline_phases}

    running_idx = next(
        (i for i, p in enumerate(_PHASE_ORDER) if status_map.get(p) == "running"),
        None,
    )
    if running_idx is None:
        done_indices = [
            i for i, p in enumerate(_PHASE_ORDER)
            if status_map.get(p) in ("done", "failed")
        ]
        center = done_indices[-1] if done_indices else 0
    else:
        center = running_idx

    start = max(0, center - 3)
    end = min(len(_PHASE_ORDER), center + 4)

    t = Text()
    if start > 0:
        t.append("… ", style="dim")
    for i in range(start, end):
        phase = _PHASE_ORDER[i]
        st = status_map.get(phase, "pending")
        if st == "done":
            t.append(f"✓ {phase}", style="dim")
        elif st == "running":
            t.append(f"◉ {phase}", style="bold #22d3ee")
        elif st == "failed":
            t.append(f"✗ {phase}", style="bold red")
        else:
            t.append(f"· {phase}", style="dim")
        if i < end - 1:
            t.append("  ")
    if end < len(_PHASE_ORDER):
        t.append("  …", style="dim")
    # Keep the track on a single line: on a narrow console the ±3 window would wrap,
    # changing the panel height between frames and tearing the Live region. Crop with
    # an ellipsis instead of wrapping so the panel height stays constant at any width.
    t.no_wrap = True
    t.overflow = "ellipsis"
    return t


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
        # Inject progress bar
        self._inject_inbox_label: str = ""
        self._inject_progress: Progress | None = None
        self._inject_task_id = None
        # Batch run progress (refine/enrich)
        self._batch: dict | None = None
        # Register as batch hook so emit_batch_event reaches this renderer
        global _batch_run_hook
        _batch_run_hook = self.__call__
        from silica.agent.bus import BUS
        BUS.subscribe("work/feedback", self._on_work_feedback)

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
        if self._batch is not None:
            return  # Batch Live is managed by _update_batch_panel / _finalize_batch
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
            if self._inject_progress is not None and self._inject_task_id is not None:
                # Count DISTINCT completed phases, not raw done/failed events: a phase
                # can re-fire `done` within one run (retries / deferred reprocessing),
                # and bare advance=1 overflowed the bar (e.g. 68/16). _pipeline_phases
                # is deduped by label, so this is naturally idempotent. Cap belt-and-braces.
                completed = sum(1 for e in self._pipeline_phases if e["status"] in ("done", "failed"))
                self._inject_progress.update(
                    self._inject_task_id, completed=min(completed, len(_PHASE_LABELS))
                )
        self._update_live()

    def _on_work_feedback(self, event) -> None:
        """Called from BUS when a sub-agent publishes a WorkFeedbackEvent."""
        if self._batch is None:
            return
        if event.kind != self._batch["kind"]:
            return
        self._batch["micro_phase"] = event.phase
        self._update_batch_panel()

    def _update_live(self) -> None:
        if self._batch is not None:
            return  # Batch panel managed by _update_batch_panel
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

            if name == "silica_run_injector" and (self._pipeline_phases or self._inject_progress is not None):
                running = next((e for e in self._pipeline_phases if e["status"] == "running"), None)
                phase_name = running["phase"] if running else "…"
                spinner_line = Spinner("dots", text=f"  [dim]⠿[/] [cyan]{escape(phase_name)}[/]…", style="dim")
                title = f"[bold]injector[/] [dim]·[/] {escape(self._inject_inbox_label)}"
                parts: list = [spinner_line, _stage_track(self._pipeline_phases, CONSOLE.width)]
                if self._inject_progress is not None:
                    parts.append(self._inject_progress)
                renderables.append(Panel(Group(*parts), title=title, border_style="dim cyan"))
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

    def _update_batch_panel(self) -> None:
        if self._batch is None or not CONSOLE.is_terminal:
            return
        batch = self._batch
        batch["progress_obj"].update(batch["task_id"], completed=batch["done"])
        cur = batch["current_label"]
        cur_str = f"  [dim]({escape(cur)})[/]" if cur else ""
        spinner_line = Spinner(
            "dots",
            text=f"  [dim]⠿[/] Batch {batch['done']}/{batch['total']}{cur_str}",
            style="dim",
        )
        title = f"[bold]{escape(batch['kind'])}[/] [dim]·[/] {escape(batch['label'])}"
        micro = batch.get("micro_phase", "")
        if micro:
            micro_parts = []
            for mp in _MICRO_PHASE_ORDER:
                display = mp.replace("_", " ")
                if mp == micro:
                    micro_parts.append(f"[bold #22d3ee]◉ {display}[/]")
                else:
                    micro_parts.append(f"[dim]· {display}[/]")
            micro_text = Text.from_markup("   ".join(micro_parts))
            panel_content = Group(spinner_line, batch["progress_obj"], micro_text)
        else:
            panel_content = Group(spinner_line, batch["progress_obj"])
        panel = Panel(
            panel_content,
            title=title,
            border_style="dim cyan",
        )
        if self._live is None:
            self._live = Live(panel, console=CONSOLE, refresh_per_second=12, transient=True)
            self._live.start()
        else:
            self._live.update(panel)

    def _finalize_batch(self) -> None:
        batch = self._batch
        if batch is None:
            return
        self._batch = None
        batch["progress_obj"].stop()
        if self._live is not None:
            self._live.stop()
            self._live = None
        kind = escape(batch["kind"])
        label = escape(batch["label"])
        done = batch["done"]
        total = batch["total"]
        failed = batch["failed"]
        elapsed = f"{time.monotonic() - batch['start_time']:.1f}s"
        if failed > 0:
            CONSOLE.print(
                f"  [tool.err]✗[/] [bold]{kind}[/] [dim]·[/] {label}"
                f"   {done}/{total} batches [dim]·[/] {failed} failed [dim]·[/] {elapsed}"
            )
        else:
            CONSOLE.print(
                f"  [tool.ok]✓[/] [bold]{kind}[/] [dim]·[/] {label}"
                f"   {done}/{total} batches [dim]·[/] {elapsed}"
            )

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

        if isinstance(event, LLMStreamEvent):
            if not CONSOLE.is_terminal:
                return
            import sys
            if event.chunk_type == "reasoning":
                if CONFIG.show_thinking or mode == "verbose" or CONFIG.verbose:
                    sys.stdout.write(f"\033[2m{event.content}\033[0m")
            elif event.chunk_type == "text":
                sys.stdout.write(event.content)
            sys.stdout.flush()
            return

        if isinstance(event, ReasoningEvent):
            self._stop_spinner()
            if CONFIG.show_thinking or mode == "verbose" or CONFIG.verbose:
                body = _head_cap(event.text).strip()
                indented = "\n".join(f"  [reasoning.gutter]│[/] [dim]{line}[/]" for line in body.splitlines())
                CONSOLE.print(f"  [reasoning]✦ thinking[/]\n{indented}\n")
            return

        if isinstance(event, ToolErrorEvent):
            if self._batch is not None:
                if event.name in ("silica_refine_batch", "silica_enrich_batch"):
                    self._batch["failed"] += 1
                CONSOLE.print(f"  [tool.err]✗[/] {escape(event.name)}  {escape(event.error[:80])}")
                return
            self._stop_spinner()
            CONSOLE.print(f"  [tool.err]✗[/] [bold]{escape(event.name)}[/]: [tool.err]{escape(event.error)}[/]")
            self._active_tools.pop(event.call_id, None)
            self._update_live()
            return

        if mode == "off":
            return

        if isinstance(event, BatchRunStartEvent):
            if not CONSOLE.is_terminal:
                return
            self._stop_spinner()
            progress_obj = Progress(
                SpinnerColumn(),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                auto_refresh=False,
            )
            task_id = progress_obj.add_task("", total=event.total)
            # Not started on purpose — embedded in self._live (see injector note); a
            # Progress.start() here would open a second Live and orphan the bar.
            self._batch = {
                "run_id": event.run_id,
                "kind": event.kind,
                "label": event.label,
                "total": event.total,
                "done": 0,
                "failed": 0,
                "start_time": time.monotonic(),
                "progress_obj": progress_obj,
                "task_id": task_id,
                "current_label": "",
                "micro_phase": "",
            }
            self._update_batch_panel()
            return

        if isinstance(event, ToolStartEvent):
            if self._batch is not None:
                return  # All tool spinners suppressed during batch run
            if CONSOLE.is_terminal:
                self._active_tools[event.call_id] = {"name": event.name, "args": event.args}
                if event.name == "silica_run_injector":
                    self._injector_call_id = event.call_id
                    inbox_file = event.args.get("inbox_file", "")
                    if not inbox_file:
                        files = event.args.get("inbox_files", [])
                        inbox_file = files[0] if isinstance(files, list) and files else "?"
                    self._inject_inbox_label = str(inbox_file)
                    self._injector_desc = _synthetic_tool_desc(event.name, event.args)
                    self._pipeline_phases = []
                    self._phase_start_times = {}
                    _set_pipeline_hook(self._on_pipeline_phase)
                    self._inject_progress = Progress(
                        SpinnerColumn(),
                        BarColumn(),
                        MofNCompleteColumn(),
                        TimeElapsedColumn(),
                        auto_refresh=False,
                    )
                    self._inject_task_id = self._inject_progress.add_task("", total=len(_PHASE_LABELS))
                    # Do NOT start() it: Progress.start() spins up its own Live on the
                    # global console, which double-renders the bar (an orphan above the
                    # panel on small consoles). We embed it as a renderable in self._live,
                    # which drives the rendering.
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
            # Batch: track ledger_next completions to advance progress
            if self._batch is not None and event.name == "silica_ledger_next":
                try:
                    result_data = json.loads(event.result)
                except Exception:
                    result_data = {}
                if result_data.get("done"):
                    self._finalize_batch()
                else:
                    payload = result_data.get("payload", {})
                    note_paths = payload.get("note_paths", [])
                    if isinstance(note_paths, list) and note_paths:
                        name0 = note_paths[0].rsplit("/", 1)[-1]
                        extra = f" +{len(note_paths) - 1}" if len(note_paths) > 1 else ""
                        self._batch["current_label"] = f"{name0}{extra}"
                    self._batch["done"] = min(self._batch["done"] + 1, self._batch["total"])
                    self._batch["micro_phase"] = ""
                    self._update_batch_panel()
                return
            # Suppress all other tool completions during batch
            if self._batch is not None:
                return

            dur = f"{event.duration_s:.3f}s"
            if CONSOLE.is_terminal:
                self._stop_spinner()
                desc = _synthetic_tool_desc(event.name, event.args)

                if event.name == "silica_run_injector" and self._injector_call_id == event.call_id:
                    # Deregister pipeline hook; print compact single-line summary
                    _set_pipeline_hook(None)
                    self._injector_call_id = None
                    if self._inject_progress is not None:
                        self._inject_progress.stop()
                        self._inject_progress = None
                        self._inject_task_id = None
                    done_phases = [e for e in self._pipeline_phases if e["status"] == "done"]
                    failed_phases = [e for e in self._pipeline_phases if e["status"] == "failed"]
                    total_count = len(self._pipeline_phases)
                    lbl = escape(self._inject_inbox_label)
                    inject_dur = f"{event.duration_s:.1f}s"
                    if failed_phases:
                        last_phase = self._pipeline_phases[-1]["phase"] if self._pipeline_phases else "?"
                        CONSOLE.print(
                            f"  [tool.err]✗[/] [bold]injector[/] [dim]·[/] {lbl}"
                            f"   {last_phase} [dim]·[/] {inject_dur}"
                        )
                    else:
                        CONSOLE.print(
                            f"  [tool.ok]✓[/] [bold]injector[/] [dim]·[/] {lbl}"
                            f"   {len(done_phases)}/{total_count} phases [dim]·[/] {inject_dur}"
                        )
                    self._pipeline_phases = []
                    self._inject_inbox_label = ""
                else:
                    CONSOLE.print(f"  [tool.ok]✓[/] {desc} [dim]({dur})[/]")
                    if mode == "verbose":
                        redacted = _redact(event.result)
                        if redacted is not None:
                            head = _head_result(redacted.strip())
                            if head:
                                CONSOLE.print(f"    [dim]{escape(head)}[/]")
                        else:
                            CONSOLE.print("    [dim][result redacted][/]")

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
        """Unconditionally stop the live display and deregister all hooks.

        Called on KeyboardInterrupt / uncaught exceptions so the terminal is
        always restored before the next prompt is printed.
        """
        global _batch_run_hook
        _batch_run_hook = None
        _set_pipeline_hook(None)
        from silica.agent.bus import BUS
        BUS.unsubscribe("work/feedback", self._on_work_feedback)
        if self._batch is not None:
            batch = self._batch
            self._batch = None
            batch["progress_obj"].stop()
            if self._live is not None:
                self._live.stop()
                self._live = None
            CONSOLE.print(
                f"  [bold yellow]⚠[/]  {escape(batch['kind'])} [dim]·[/] {escape(batch['label'])}"
                f"   interrupted at {batch['done']}/{batch['total']}"
            )
        if self._inject_progress is not None:
            self._inject_progress.stop()
            self._inject_progress = None
            self._inject_task_id = None
        self._stop_spinner()


def make_progress_callback() -> _ProgressRenderer:
    return _ProgressRenderer()
