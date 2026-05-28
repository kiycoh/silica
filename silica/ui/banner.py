from __future__ import annotations

from importlib.resources import files

from rich.text import Text

from silica.config import CONFIG
from silica.ui.console import CONSOLE

_VERSION = "0.2.1"
_CAPTION = f"  [dim]v{_VERSION} · Obsidian-native agent[/]"


def _load_wordmark() -> list[str]:
    # Load the static wordmark from the asset
    text = (files("silica.ui") / "assets" / "ascii-art-font.txt").read_text(encoding="utf-8")
    return text.rstrip("\n").split("\n") + [""]


def _gradient(n: int, c0=(0x22, 0xd3, 0xee), c1=(0x63, 0x66, 0xf1)) -> list[str]:
    if n <= 1:
        return [f"#{c0[0]:02x}{c0[1]:02x}{c0[2]:02x}"]
    out = []
    for i in range(n):
        t = i / (n - 1)
        r, g, b = (round(a + (b - a) * t) for a, b in zip(c0, c1))
        out.append(f"#{r:02x}{g:02x}{b:02x}")
    return out  # cyan → indigo


def _print_wordmark() -> bool:
    if CONSOLE.width < 60:
        return False
    try:
        art = _load_wordmark()
    except Exception:
        return False
    for line, color in zip(art, _gradient(len(art))):
        CONSOLE.print(Text(line, style=f"bold {color}"))
    CONSOLE.print(_CAPTION)
    return True


def print_banner() -> None:
    style = CONFIG.banner_style
    if style in ("crystal", "wordmark") and _print_wordmark():
        return
    # minimal or fallback from failed guard
    CONSOLE.print(f"  [bold cyan]silica[/] [dim]v{_VERSION} · Obsidian-native agent[/]")
