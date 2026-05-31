from __future__ import annotations

import pyfiglet
from rich.text import Text

from silica.config import CONFIG
from silica.ui.console import CONSOLE
from silica.ui.theme import BRAND_CYAN, BRAND_INDIGO

_VERSION = "0.2.1"
_CAPTION = f"  [dim]v{_VERSION} · Your personal note curator agent[/]"


def _gradient(n: int, c0: tuple[int, int, int] = BRAND_CYAN, c1: tuple[int, int, int] = BRAND_INDIGO) -> list[str]:
    if n <= 1:
        return [f"#{c0[0]:02x}{c0[1]:02x}{c0[2]:02x}"]
    out = []
    for i in range(n):
        t = i / (n - 1)
        r, g, b = (round(a + (bb - a) * t) for a, bb in zip(c0, c1))
        out.append(f"#{r:02x}{g:02x}{b:02x}")
    return out  # cyan → indigo


def _print_wordmark() -> bool:
    try:
        raw = pyfiglet.figlet_format("silica", font=CONFIG.banner_font)
        art = [ln for ln in raw.rstrip("\n").split("\n")]
        # Drop trailing blank lines
        while art and not art[-1].strip():
            art.pop()
    except Exception:
        return False

    min_width = max((len(ln) for ln in art), default=0) + 2
    if CONSOLE.width < min_width:
        return False

    for line, color in zip(art, _gradient(len(art))):
        CONSOLE.print(Text(line, style=f"bold {color}"))
    CONSOLE.print(_CAPTION)
    return True


def print_banner() -> None:
    style = CONFIG.banner_style
    if style == "wordmark" and _print_wordmark():
        return
    # minimal or fallback from failed guard
    CONSOLE.print(f"  [bold cyan]silica[/] [dim]v{_VERSION} · Your personal note curator agent[/]")
