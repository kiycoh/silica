from __future__ import annotations

from rich import box
from rich.console import Group
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from silica.config import CONFIG
from silica.ui.banner import banner_group, print_banner
from silica.ui.console import CONSOLE

_MIN_SIDE_BY_SIDE_WIDTH = 100


def _model_vault_line(model_slug: str, vault: str) -> Text:
    from silica.ui.style import GLYPHS
    t = Text("  ")
    t.append(GLYPHS["model"], style="dim")
    t.append(f" {model_slug}", style="bold")
    t.append("  ·  ", style="dim")
    t.append(GLYPHS["vault"], style="dim")
    t.append(f" vault: ", style="")
    t.append(vault, style="bold")
    return t


def print_home() -> None:
    """Banner + model/vault + pinned commands + footer. Shown at launch and after /clear."""
    from silica.ui.commands import COMMANDS
    from silica.ui.style import command_table

    vault = CONFIG.vault_name or "—"
    model_slug = CONFIG.model.rsplit("/", 1)[-1]
    pinned = [c for c in COMMANDS if c.home_pin]

    bg = banner_group()

    if bg is not None and CONSOLE.width >= _MIN_SIDE_BY_SIDE_WIDTH:
        left = Group(bg, Text(""), _model_vault_line(model_slug, vault))

        right = Group(
            Text("Overview", style="bold"),
            Text(""),
            command_table(pinned, show_summary=False),
        )

        outer = Table(show_header=False, box=box.ROUNDED, padding=(0, 1), pad_edge=True, border_style="dim")
        outer.add_column(no_wrap=False)
        outer.add_column(no_wrap=False)
        outer.add_row(left, Padding(right, (0, 0, 0, 6)))

        CONSOLE.print(outer)
    else:
        print_banner()
        CONSOLE.print()
        CONSOLE.print(_model_vault_line(model_slug, vault))
        CONSOLE.print()
        CONSOLE.print("  [bold]Overview[/]")
        CONSOLE.print()
        CONSOLE.print(Padding(command_table(pinned, show_summary=False), (0, 0, 0, 4)))
        CONSOLE.print()

    CONSOLE.print()
    CONSOLE.rule(style="dim", characters="▀▄")
    CONSOLE.print()
    CONSOLE.print("  [dim]/  commands   ·   /help  all   ·   /exit  quit[/]")
    CONSOLE.print()
