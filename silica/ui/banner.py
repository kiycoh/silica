from __future__ import annotations

from importlib.resources import files

from rich.text import Text

from silica.config import CONFIG
from silica.ui.console import CONSOLE

_VERSION = "0.1.0"
_CAPTION = f"  [dim]v{_VERSION} · agente Obsidian-nativo[/]"


def _load_crystal() -> list[str]:
    # Carica l'asset ad alta risoluzione e lo rimpicciolisce (downsample 2x2)
    text = (files("silica.ui") / "assets" / "silica_crystal.txt").read_text(encoding="utf-8")
    lines = text.rstrip("\n").split("\n")

    weights = {" ": 0, "░": 1, "▒": 2, "▓": 3, "█": 4}
    chars = [" ", "░", "▒", "▓", "█"]
    max_w = max(len(l) for l in lines)
    grid = [[weights.get(c, 0) for c in line.ljust(max_w)] for line in lines]

    h = len(grid)
    w = max_w
    downsampled = []
    for r in range(0, h, 2):
        row_out = []
        for c in range(0, w, 2):
            vals = []
            for dr in (0, 1):
                for dc in (0, 1):
                    if r + dr < h and c + dc < w:
                        vals.append(grid[r + dr][c + dc])
            avg = sum(vals) / len(vals) if vals else 0
            idx = min(4, max(0, round(avg)))
            row_out.append(chars[idx])
        downsampled.append("".join(row_out).rstrip())
    return downsampled


def _load_wordmark() -> list[str]:
    # Carica la scritta d'arte statica dall'asset
    text = (files("silica.ui") / "assets" / "ascii-art-font.txt").read_text(encoding="utf-8")
    return text.rstrip("\n").split("\n")


def _gradient(n: int, c0=(0x22, 0xd3, 0xee), c1=(0x63, 0x66, 0xf1)) -> list[str]:
    if n <= 1:
        return [f"#{c0[0]:02x}{c0[1]:02x}{c0[2]:02x}"]
    out = []
    for i in range(n):
        t = i / (n - 1)
        r, g, b = (round(a + (b - a) * t) for a, b in zip(c0, c1))
        out.append(f"#{r:02x}{g:02x}{b:02x}")
    return out  # cyan → indigo


def _print_crystal() -> bool:
    # Guard di larghezza (cristallo ~17 col + padding 8 col + wordmark 56 col = ~81 col)
    # e altezza (~ 18 righe)
    if CONSOLE.width < 82 or CONSOLE.size.height < 18:
        return False
    try:
        crystal_lines = _load_crystal()
    except Exception:
        return False

    # Carica la scritta SILICA dall'asset statico
    try:
        wordmark_lines = _load_wordmark()
    except Exception:
        wordmark_lines = []

    # Combina cristallo (sinistra) e scritta (destra)
    combined_lines: list[str] = []
    for idx, c_line in enumerate(crystal_lines):
        if wordmark_lines:
            c_line_padded = c_line.ljust(25)
            w_idx = idx - 4  # Allinea verticalmente la scritta a partire dalla riga 5
            if 0 <= w_idx < len(wordmark_lines):
                w_line = wordmark_lines[w_idx]
            else:
                w_line = ""
            combined_lines.append(c_line_padded + w_line)
        else:
            combined_lines.append(c_line)

    # Stampa con gradiente
    for line, color in zip(combined_lines, _gradient(len(combined_lines))):
        CONSOLE.print(Text(line, style=f"bold {color}"))

    # Allinea il sottotitolo/versione sotto la scritta SILICA se presente
    caption_indent = " " * 25 if wordmark_lines else "  "
    CONSOLE.print(f"{caption_indent}[dim]v{_VERSION} · agente Obsidian-nativo[/]")
    return True


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
    if style == "crystal" and _print_crystal():
        return
    if style == "wordmark" and _print_wordmark():
        return
    # minimal o fallback da guard fallita
    CONSOLE.print(f"  [bold cyan]silica[/] [dim]v{_VERSION} · agente Obsidian-nativo[/]")
