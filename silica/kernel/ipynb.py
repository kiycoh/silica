# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""ipynb — defensive nbformat-v4 cell extraction (spec-code-lane §3).

Stdlib json, no nbformat dependency: only the `cells` list and the
kernelspec language are read; outputs are ignored (noise, often huge
base64). IPython magic lines (%, !, ?) are stripped from code cells —
left in, a single magic yields a tree-sitter ERROR node that can swallow
adjacent import statements (a silent drop, against spec §7).

Lives in kernel so codegraph can index notebooks without importing from
silica/sources/ (layering: sources import kernel, never the reverse).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

# kernelspec language → codeast language (anything else: narrative-only note)
CODEAST_LANGUAGE: dict[str, str] = {
    "python": "python",
    "typescript": "typescript",
    "javascript": "javascript",
}


@dataclass(frozen=True)
class NotebookCells:
    markdown: list[str]   # markdown cell sources, cell order
    code: str             # code cells concatenated, magic lines stripped
    language: str         # kernelspec language, default "python"


def parse_cells(text: str) -> NotebookCells:
    """Parse a notebook JSON string. Raises ValueError on malformed JSON or
    a missing `cells` list (the adapter read() contract, same as code.py)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"malformed notebook JSON: {e}") from e
    cells = data.get("cells") if isinstance(data, dict) else None
    if not isinstance(cells, list):
        raise ValueError("notebook has no cells list")

    language = "python"
    meta = data.get("metadata")
    if isinstance(meta, dict):
        ks = meta.get("kernelspec")
        if isinstance(ks, dict) and isinstance(ks.get("language"), str):
            language = ks["language"]

    markdown: list[str] = []
    code_parts: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        src = cell.get("source")
        if isinstance(src, list):
            src = "".join(s for s in src if isinstance(s, str))
        if not isinstance(src, str) or not src.strip():
            continue
        kind = cell.get("cell_type")
        if kind == "markdown":
            markdown.append(src)
        elif kind == "code":
            kept = [ln for ln in src.splitlines()
                    if not ln.lstrip().startswith(("%", "!", "?"))]
            if any(ln.strip() for ln in kept):
                code_parts.append("\n".join(kept))
    return NotebookCells(markdown=markdown, code="\n\n".join(code_parts), language=language)
