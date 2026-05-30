from __future__ import annotations

from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML

from silica.config import CONFIG

SLASH_COMMANDS = ["/exit", "/model", "/tools", "/clear", "/verbose", "/thinking", "/help"]

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


def build_session() -> PromptSession:
    return PromptSession(
        history=FileHistory(str(_history_path())),
        auto_suggest=AutoSuggestFromHistory(),
        completer=WordCompleter(SLASH_COMMANDS, sentence=True),
    )


def prompt_text() -> HTML:
    if CONFIG.vault_name:
        return HTML(f"<ansicyan><b>silica</b></ansicyan> <ansigray>[{CONFIG.vault_name}]</ansigray> › ")
    return HTML("<ansicyan><b>silica</b></ansicyan> › ")
