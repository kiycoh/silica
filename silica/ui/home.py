from __future__ import annotations

from pathlib import Path

from rich.console import Group
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from silica.config import CONFIG
from silica.ui.banner import banner_group, banner_height, print_banner
from silica.ui.console import CONSOLE

_MIN_SIDE_BY_SIDE_WIDTH = 100


def _model_vault_line(model_slug: str, worker_slug: str, vault: str) -> Text:
    from silica.ui.style import GLYPHS
    t = Text("  ")
    t.append(GLYPHS["model"], style="dim")
    t.append(f" {model_slug}", style="bold")
    t.append("  ·  ", style="dim")
    t.append(GLYPHS["worker"], style="dim")
    t.append(f" {worker_slug}", style="bold")
    t.append("  ·  ", style="dim")
    t.append(GLYPHS["vault"], style="dim")
    t.append(f" vault: ", style="")
    t.append(vault, style="bold")
    return t


def print_home() -> None:
    """Banner + model/vault + commands overview. Shown at launch and after /clear."""
    from silica.ui.commands import COMMANDS
    from silica.ui.style import command_table

    vault = Path(CONFIG.vault_path).name if CONFIG.vault_path else (CONFIG.vault_name or "—")
    model_slug = (CONFIG.model or "(not configured)").rsplit("/", 1)[-1]
    worker_model = CONFIG.worker_model or CONFIG.model or "(not configured)"
    worker_slug = worker_model.rsplit("/", 1)[-1]
    pinned = [c for c in COMMANDS if c.home_pin]
    help_cmd = next(c for c in COMMANDS if c.name == "/help")
    exit_cmd = next(c for c in COMMANDS if c.name == "/exit")
    all_cmds = pinned + [help_cmd, exit_cmd]

    bg = banner_group()

    if bg is not None and CONSOLE.width >= _MIN_SIDE_BY_SIDE_WIDTH:
        left_height = banner_height()
        right_height = 1 + 1 + 1 + len(pinned) + 2  # model/vault + blank + heading + pinned + /help + /exit
        sep_height = max(left_height, right_height)

        if sep_height < 2:
            sep_lines: list = [Text("│", style="dim")]
        else:
            sep_lines = (
                [Text("╷", style="dim")]
                + [Text("│", style="dim")] * (sep_height - 2)
                + [Text("╵", style="dim")]
            )
        separator = Group(*sep_lines)

        right = Group(
            _model_vault_line(model_slug, worker_slug, vault),
            Text(""),
            Text("Commands overview", style="bold"),
            command_table(all_cmds, show_summary=False, compact=True),
        )

        outer = Table(show_header=False, box=None, show_edge=False, pad_edge=False, padding=(0, 1))
        outer.add_column(no_wrap=False)
        outer.add_column(no_wrap=False)
        outer.add_column(no_wrap=False)
        outer.add_row(bg, separator, right)

        CONSOLE.print(outer)
    else:
        print_banner()
        CONSOLE.print()
        CONSOLE.print(_model_vault_line(model_slug, worker_slug, vault))
        CONSOLE.print()
        CONSOLE.print("  [bold]Commands overview[/]")
        CONSOLE.print()
        CONSOLE.print(Padding(command_table(all_cmds, show_summary=False), (0, 0, 0, 4)))
        CONSOLE.print()
