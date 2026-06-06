from __future__ import annotations

from rich.theme import Theme

BRAND_CYAN = (0x22, 0xD3, 0xEE)
BRAND_INDIGO = (0x63, 0x66, 0xF1)

SILICA_THEME = Theme(
    {
        "brand.cyan": "#22d3ee",
        "brand.indigo": "#6366f1",
        "reasoning": "#22d3ee",
        "reasoning.gutter": "#6366f1",
        "role.assistant": "bold #22d3ee",
        "tool.ok": "green",
        "tool.err": "red",
        "muted": "dim",
    }
)
