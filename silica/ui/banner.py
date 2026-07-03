from __future__ import annotations

from rich.console import Group as RichGroup
from rich.text import Text

from silica import __version__ as _VERSION
from silica.config import CONFIG
from silica.ui.console import CONSOLE
from silica.ui.theme import BRAND_CYAN, BRAND_INDIGO

_CAPTION = f"  [dim]v{_VERSION} · Your personal note curator agent[/]"

# Pre-rendered art.text2art("SILICA", font="tarty1") — hardcoded to drop the dep.
_ART = """\
░██████╗██╗██╗░░░░░██╗░█████╗░░█████╗░
██╔════╝██║██║░░░░░██║██╔══██╗██╔══██╗
╚█████╗░██║██║░░░░░██║██║░░╚═╝███████║
░╚═══██╗██║██║░░░░░██║██║░░██╗██╔══██║
██████╔╝██║███████╗██║╚█████╔╝██║░░██║
╚═════╝░╚═╝╚══════╝╚═╝░╚════╝░╚═╝░░╚═╝""".split("\n")


def _gradient(n: int, c0: tuple[int, int, int] = BRAND_CYAN, c1: tuple[int, int, int] = BRAND_INDIGO) -> list[str]:
    if n <= 1:
        return [f"#{c0[0]:02x}{c0[1]:02x}{c0[2]:02x}"]
    out = []
    for i in range(n):
        t = i / (n - 1)
        r, g, b = (round(a + (bb - a) * t) for a, bb in zip(c0, c1))
        out.append(f"#{r:02x}{g:02x}{b:02x}")
    return out  # cyan → indigo


def _compute_art() -> list[str] | None:
    """Return art lines if banner would render, else None."""
    if not CONFIG.show_banner:
        return None
    if CONSOLE.width < max(len(ln) for ln in _ART) + 2:
        return None
    return _ART


def banner_group() -> RichGroup | None:
    """Banner art + caption as a renderable Group, or None if font unavailable or terminal too narrow."""
    art = _compute_art()
    if art is None:
        return None
    items: list = [Text(line, style=f"bold {color}") for line, color in zip(art, _gradient(len(art)))]
    items.append(Text.from_markup(_CAPTION))
    return RichGroup(*items)


def banner_height() -> int:
    """Number of art lines banner_group() would render, or 0 if it would return None."""
    art = _compute_art()
    return len(art) if art is not None else 0


def print_banner() -> None:
    bg = banner_group()
    if bg is not None:
        CONSOLE.print(bg)
        return
    CONSOLE.print(f"  [bold cyan]silica[/] [dim]v{_VERSION} · Your personal note curator agent[/]")
