from __future__ import annotations

from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML

from silica.config import CONFIG

SLASH_COMMANDS = ["/exit", "/model", "/tools", "/clear", "/verbose", "/thinking", "/help"]


def _history_path() -> Path:
    p = Path.home() / ".silica"
    p.mkdir(parents=True, exist_ok=True)
    return p / "history"


def bottom_toolbar() -> HTML:
    vault = CONFIG.vault_name or "—"
    think = "on" if CONFIG.show_thinking else "off"
    return HTML(f" <b>{CONFIG.model}</b>  vault:{vault}  progress:{CONFIG.tool_progress}  thinking:{think} ")


def build_session() -> PromptSession:
    return PromptSession(
        history=FileHistory(str(_history_path())),
        auto_suggest=AutoSuggestFromHistory(),
        completer=WordCompleter(SLASH_COMMANDS, sentence=True),
    )


def prompt_text() -> HTML:
    return HTML("<ansicyan><b>silica</b></ansicyan> › ")
