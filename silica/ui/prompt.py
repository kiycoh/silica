# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

from __future__ import annotations

from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion, FuzzyCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style

from silica.config import CONFIG
from silica.ui.commands import COMMANDS


def _vault_display() -> str:
    if CONFIG.vault_path:
        return Path(CONFIG.vault_path).name
    return CONFIG.vault_name or "—"

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
    color = "ansicyan" if ratio < 0.7 else ("ansiyellow" if ratio < 0.9 else "ansired")
    pct = round(ratio * 100)
    return f" <{color}>{bar}</{color}> <ansigray>{pct}%</ansigray> "


def bottom_toolbar() -> HTML:
    # Vault lives in the prompt line — not repeated here. Model shown as slug.
    model_slug = (CONFIG.model or "—").rsplit("/", 1)[-1]
    think = "thinking:on" if CONFIG.show_thinking else "thinking:off"
    progress = f"progress:{CONFIG.tool_progress}"
    meter = _context_meter()
    return HTML(
        f" <ansicyan><b>{model_slug}</b></ansicyan>  "
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
        # Kill the default reverse-video strip — the toolbar renders flat on the
        # terminal background, colors come from the HTML fragments themselves.
        style=Style.from_dict({"bottom-toolbar": "noreverse"}),
    )


def prompt_text() -> HTML:
    vault = _vault_display()
    if vault != "—":
        return HTML(f"<ansicyan><b>silica</b></ansicyan> <ansigray>[{vault}]</ansigray> › ")
    return HTML("<ansicyan><b>silica</b></ansicyan> › ")
