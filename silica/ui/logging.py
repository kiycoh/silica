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


# Human-friendly rewording of standard log messages
FRIENDLY_TEMPLATES = {
    # Debug messages
    "Registered tool: %s (class=%s)":
        "Tool registered in system: {0} (implemented by class {1})",
    "Redaction failed (omitting detail): %s":
        "Sensitive data redaction failed (details omitted for safety): {0}",
    "tool_progress_callback error (swallowed): %s":
        "Error in tool progress callback (ignored): {0}",
    "Agent loop iteration %d":
        "Starting agent loop iteration: {0}",
    "Created temporary staging file at: %s":
        "Created a temporary file for staging changes at: {0}",
    "FSM Transition: %s -> executing handler":
        "State machine transition: {0} -> executing associated handler",
    "Rebuilding FS graph index...":
        "Rebuilding File System index...",
    "Skipping indexing for inbox directory: %s":
        "Skipping indexing for the Inbox directory: {0}",
    "Indexed %d notes":
        "Indexing complete: {0} notes found and loaded",
    "CLI exec: %s":
        "Executing CLI command: {0}",
    "CLI exec: %s  (timeout=%.1fs)":
        "Executing CLI command: {0} (timeout={1}s)",
    "CLI stderr: %s":
        "CLI standard error: {0}",

    # LLM call summary (compact)
    "LLM call: model=%s | msg=%d | tools=%d":
        "LLM call → {0}  [{1} msg, {2} tools]",
    "LLM resp: finish=%s | tool_calls=%d | text=%r":
        "LLM response → finish={0}  {1} tool call(s)  text={2}",

    # Info messages
    "Refiner phase: %s":
        "Refinement phase: {0}",
    "Skipping already processed note: %s":
        "Skipping note already processed: {0}",
    "Injector phase: %s":
        "Injection phase: {0}",
    "VALIDATE: no actionable ops (all skip) — short-circuit to CLEANUP":
        "Validation complete: no updates needed, short-circuiting to cleanup",
    "Calling Distiller LLM (payload checksum %s)":
        "Calling Distiller model to extract changes (checksum: {0})",
    "Distiller produced %d updates":
        "Distiller produced {0} updates",
    "Validation: hub '%s' does not exist. Injected creation operation at %s":
        "Hub note '{0}' does not exist. Injected creation operation at {1}",
    "Restored %s to version %d":
        "Restored file {0} to version {1}",
    "Rolled back created note: %s":
        "Undid note creation: {0} (removed during rollback)",
    "Rolled back created note %s (already absent)":
        "Undoing note creation of {0} not needed (already absent)",
    "Rollback complete for txn %s":
        "Rollback successfully completed for transaction {0}",

    # Warning messages
    "restore_version with no version number for %s — skipped":
        "Requested restoration of {0} without specific version number (ignored)",
    "Failed to load recipe 'injector', using defaults: %s":
        "Failed to load recipe for injector, using default values: {0}",
    "Failed to fetch pre-write links for %s: %s":
        "Failed to fetch pre-write links for {0}: {1}",
    "Failed to write ledger: %s":
        "Failed to write transaction log (ledger): {0}",
    "Failed to mark rollback in ledger: %s":
        "Failed to record rollback in ledger: {0}",
    "Failed to load recipe 'refiner', using defaults: %s":
        "Failed to load recipe for refiner, using default values: {0}",
    "Distiller provider call failed, falling back to litellm: %s":
        "Distiller provider call failed, falling back to standard litellm: {0}",
    "Failed to index %s: %s":
        "Failed to index note {0}: {1}",
    "No history available for %s: %s":
        "No version history available for {0} (reason: {1})",
    "Convergence guard: tool '%s' with args %s failed consecutively. Injecting warning message.":
        "Loop detected: tool '{0}' with args {1} failed consecutively. Injected warning message.",
    "Agent loop hit max iterations (%d)":
        "Agent loop reached the maximum limit of iterations ({0})",

    # Error messages
    "Rollback error: %s":
        "Error during rollback of changes: {0}",
    "FSM Error in state %s: %s":
        "Error in the state machine in state {0}: {1}",
    "Failed to take pre-write graph snapshot: %s":
        "Failed to save initial graph snapshot: {0}",
    "Failed to perform graph-diff check: %s":
        "Failed to compare graph differences: {0}",
    "Rollback partially failed: %s":
        "Rollback partially failed: {0}",
    "Rollback failed: %s":
        "Annulling changes (rollback) failed: {0}",
    "Enricher failed for task %d: %s":
        "Enrichment phase failed for task {0}: {1}",
    "Distiller output truncated/malformed (%s); salvaged %d complete update(s) from the valid prefix — batch continues with partial set":
        "Distiller output was cut off ({0}); recovered {1} complete note(s) and continued the batch",
    "Transient LLM error, retries exhausted: %s":
        "Transient LLM error, retries exhausted: {0}",
    "Permanent LLM or execution error: %s":
        "Permanent LLM or execution error: {0}",
    "LLM call failed: %s":
        "LLM call failed: {0}",
    "Failed to execute or parse eval search: %s":
        "Failed to execute or parse evaluation search: {0}",
    "Failed to restore %s: %s":
        "Failed to restore note {0}: {1}",
    "Failed to delete created note %s during rollback: %s":
        "Failed to remove created note {0} during rollback: {1}",
    "Convergence guard: tool '%s' with args %s failed %d times consecutively. Aborting agent run.":
        "Potential infinite loop detected: tool '{0}' with args {1} failed {2} consecutive times. Aborting execution.",
}


class HumanFriendlyFormatter(logging.Formatter):
    """Logging formatter that maps technical messages to human-readable Rich markup."""

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

        friendly_message = None

        if record.name.startswith("silica"):
            template = record.msg
            if isinstance(template, str) and template in FRIENDLY_TEMPLATES:
                friendly_template = FRIENDLY_TEMPLATES[template]
                try:
                    if record.args:
                        if isinstance(record.args, dict):
                            friendly_message = friendly_template.format(**record.args)
                        elif isinstance(record.args, tuple):
                            friendly_message = friendly_template.format(*record.args)
                        else:
                            friendly_message = friendly_template.format(record.args)
                    else:
                        friendly_message = friendly_template
                except Exception:
                    pass

        if not friendly_message:
            friendly_message = message

        lines = friendly_message.split("\n")
        if len(lines) > 15:
            head = lines[:5]
            tail = lines[-5:]
            hidden = len(lines) - 10
            friendly_message = "\n".join(head + [f"  [dim]... ({hidden} lines truncated) ...[/dim]"] + tail)

        return f"  [muted][{time_str}][/muted] {icon} {friendly_message}"


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
            highlight=False,
            force_terminal=CONSOLE.is_terminal,
            color_system=CONSOLE.color_system,
            width=CONSOLE.width,
        ).print(markup, end="", soft_wrap=True)
        return buf.getvalue()
