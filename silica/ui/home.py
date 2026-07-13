# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

from __future__ import annotations

from pathlib import Path

from rich.text import Text

from silica.config import CONFIG
from silica.ui.banner import print_banner
from silica.ui.console import CONSOLE


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
    t.append(" vault: ", style="")
    t.append(vault, style="bold")
    return t


def print_home() -> None:
    """Banner + model/vault line. Shown at launch and after /clear."""
    vault = Path(CONFIG.vault_path).name if CONFIG.vault_path else (CONFIG.vault_name or "—")
    model_slug = (CONFIG.model or "(not configured)").rsplit("/", 1)[-1]
    worker_model = CONFIG.worker_model or CONFIG.model or "(not configured)"
    worker_slug = worker_model.rsplit("/", 1)[-1]

    CONSOLE.print()
    print_banner()
    from silica.update import behind_count
    if n := behind_count():
        CONSOLE.print(f"  [yellow]⚠ {n} update(s) behind — run[/] [bold yellow]silica update[/]")
    CONSOLE.print()
    CONSOLE.print(_model_vault_line(model_slug, worker_slug, vault))
    CONSOLE.print()
