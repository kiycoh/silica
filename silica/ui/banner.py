from __future__ import annotations

from rich.text import Text

from silica.config import CONFIG
from silica.ui.console import CONSOLE

_VERSION = "0.1.0"
_SHADES = ["#22d3ee", "#1fc6e0", "#1bb8d2", "#17abc4", "#139eb6"]  # cyan → teal


def print_banner() -> None:
    use_art = CONSOLE.width >= 50
    art_lines: list[str] = []
    if use_art:
        try:
            from pyfiglet import Figlet
            art_lines = Figlet(font=CONFIG.banner_font).renderText("silica").rstrip("\n").splitlines()
        except Exception:
            art_lines = []  # font mancante o pyfiglet assente → fallback

    if art_lines:
        for idx, line in enumerate(art_lines):
            shade = _SHADES[min(idx, len(_SHADES) - 1)]
            CONSOLE.print(Text(line, style=f"bold {shade}"))
        CONSOLE.print(f"  [dim]v{_VERSION} · agente Obsidian-nativo[/]")
    else:
        CONSOLE.print(f"  [bold cyan]silica[/] [dim]v{_VERSION} · agente Obsidian-nativo[/]")
