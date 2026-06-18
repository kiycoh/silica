from __future__ import annotations

from rich.markup import escape
from rich.table import Table

GLYPHS: dict[str, str] = {
    "ok": "✓",
    "err": "✗",
    "run": "⏺",
    "bullet": "·",
    "arrow": "→",
    "model": "◆",
    "worker": "◇",
    "vault": "⬡",
}

GROUP_STYLE: dict[str, str] = {
    "workflow": "#22d3ee",  # BRAND_CYAN — works in both markup and Table column style
    "direct": "cyan",
    "system": "dim",
}


def aligned_columns(rows: list[tuple[str, str]], indent: int = 4) -> list[str]:
    if not rows:
        return []
    col0_width = max(len(r[0]) for r in rows)
    prefix = " " * indent
    return [f"{prefix}{left:<{col0_width}}   {right}" for left, right in rows]


def command_table(
    commands: list,
    *,
    name_style: str = "bold #22d3ee",
    usage_style: str = "dim",
    show_summary: bool = True,
    compact: bool = False,
) -> Table:
    """Rich Table (no borders): name | usage | [summary].

    With ``compact=True`` the name column is pinned to its content width and the
    usage column flexes (ellipsised when truncated), so a narrow panel can never
    crush the command names — used by the side-by-side home overview.
    """
    table = Table(
        show_header=False, box=None, padding=(0, 3, 0, 0), pad_edge=False, expand=compact
    )
    name_width = max((len(c.name) for c in commands), default=0) if compact else None
    table.add_column(style=name_style, no_wrap=True, width=name_width)
    if compact:
        table.add_column(style=usage_style, no_wrap=True, overflow="ellipsis", ratio=1)
    else:
        # Wrappable so an outlier usage (e.g. /organize) reflows instead of
        # forcing Rich to crush the no_wrap name column down to "/o…".
        table.add_column(style=usage_style, no_wrap=False)
    if show_summary:
        table.add_column(no_wrap=False)
    for cmd in commands:
        row = (cmd.name, escape(cmd.usage), escape(cmd.summary)) if show_summary else (cmd.name, escape(cmd.usage))
        table.add_row(*row)
    return table
