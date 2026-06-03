from __future__ import annotations

from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion, FuzzyCompleter
from prompt_toolkit.formatted_text import HTML

from silica.config import CONFIG
from silica.ui.commands import command_names, COMMANDS

SLASH_COMMANDS = list(command_names())

_METER_WIDTH = 10


def _history_path() -> Path:
    p = Path.home() / ".silica"
    p.mkdir(parents=True, exist_ok=True)
    return p / "history"


def _context_meter() -> str:
    """Return a prompt_toolkit HTML snippet with a █/░ fill bar for context usage."""
    if CONFIG.max_context_tokens <= 0:
        return ""
    ratio = min(1.0, max(0.0, CONFIG.context_tokens / CONFIG.max_context_tokens))
    filled = round(ratio * _METER_WIDTH)
    bar = "█" * filled + "░" * (_METER_WIDTH - filled)
    pct = round(ratio * 100)
    return f" <ansicyan>{bar}</ansicyan> <ansigray>{pct}%</ansigray> "


def bottom_toolbar() -> HTML:
    vault = CONFIG.vault_name or "—"
    think = "thinking:on" if CONFIG.show_thinking else "thinking:off"
    progress = f"progress:{CONFIG.tool_progress}"
    meter = _context_meter()
    return HTML(
        f" <ansicyan><b>{CONFIG.model}</b></ansicyan>  "
        f"vault:<b>{vault}</b>  "
        f"{progress}  "
        f"<b>{think}</b>"
        f"{meter}"
    )


class SlashCommandCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        if not text.startswith("/"):
            return
        try:
            for cmd in COMMANDS:
                if cmd.name.startswith(text):
                    yield Completion(
                        cmd.name,
                        start_position=-len(text),
                        display=cmd.name,
                        display_meta=cmd.summary,
                    )
                for ex in cmd.examples:
                    if ex.startswith(text):
                        yield Completion(
                            ex,
                            start_position=-len(text),
                            display=ex,
                            display_meta=cmd.summary,
                        )
        except Exception:
            return


def build_session() -> PromptSession:
    return PromptSession(
        history=FileHistory(str(_history_path())),
        auto_suggest=AutoSuggestFromHistory(),
        completer=FuzzyCompleter(SlashCommandCompleter()),
    )


def prompt_text() -> HTML:
    if CONFIG.vault_name:
        return HTML(f"<ansicyan><b>silica</b></ansicyan> <ansigray>[{CONFIG.vault_name}]</ansigray> › ")
    return HTML("<ansicyan><b>silica</b></ansicyan> › ")
