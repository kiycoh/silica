from __future__ import annotations

from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from silica import __version__ as _VERSION
from silica.config import CONFIG
from silica.ui.console import CONSOLE
from silica.ui.theme import BRAND_CYAN, BRAND_INDIGO

_CAPTION = f"v{_VERSION.split('+')[0]} · Your personal note curator agent"

# Inlined from docs/assets/sili_compressed.txt (gitignored — a disk read would
# come up empty on any machine but the author's).
_MASCOT = (
    "    ██▄█▄█",
    "    █▄██▄█",
    "   ▀ ▀██▀ ▀",
    "       ▀",
)

# Hand-drawn wordmark — thin rounded line-art, deliberately not a figlet font
# (generator fonts like ANSI Shadow are everywhere; bespoke glyphs are not).
_ART = (
    "╭─╴ ╷ ╷   ╷ ╭─╴ ╭─╮",
    "╰─╮ │ │   │ │   ├─┤",
    "╶─╯ ╵ ╰─╴ ╵ ╰─╴ ╵ ╵",
)


def _gradient(n: int, c0: tuple[int, int, int] = BRAND_CYAN, c1: tuple[int, int, int] = BRAND_INDIGO) -> list[str]:
    if n <= 1:
        return [f"#{c0[0]:02x}{c0[1]:02x}{c0[2]:02x}"]
    out = []
    for i in range(n):
        t = i / (n - 1)
        r, g, b = (round(a + (bb - a) * t) for a, bb in zip(c0, c1))
        out.append(f"#{r:02x}{g:02x}{b:02x}")
    return out  # cyan → indigo


def _painted(lines: tuple[str, ...] | list[str]) -> Text:
    """Shared theme: a multi-line Text with a per-column cyan→indigo gradient.

    Same column → same color, so the hue sweeps horizontally, not line by line.
    """
    colors = _gradient(max(len(ln) for ln in lines))
    t = Text()
    for i, line in enumerate(lines):
        if i:
            t.append("\n")
        for ch, color in zip(line, colors):
            t.append(ch, style=f"bold {color}")
    return t


def print_banner() -> None:
    if not CONFIG.show_banner:
        CONSOLE.print(f"  [bold cyan]silica[/] [dim]{_CAPTION}[/]")
        return
    wordmark = _painted(_ART)
    wordmark.append("\n")
    wordmark.append(_CAPTION, style="dim")  # caption rides under the wordmark, right column
    mascot_w = max(len(ln) for ln in _MASCOT)
    if CONSOLE.width >= mascot_w + 3 + len(_CAPTION):
        grid = Table.grid(padding=(0, 3))  # logo left column, wordmark right column
        grid.add_row(_painted(_MASCOT), wordmark)
        CONSOLE.print(Padding(grid, (0, 0, 0, 2)))
    else:
        CONSOLE.print(Padding(wordmark, (0, 0, 0, 2)))
