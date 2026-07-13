# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Notebook source adapter — .ipynb first-class citizen (spec-code-lane §3).

Hybrid: markdown cells → '## Narrative' (sanitized, zero-trust ADR-0009);
code cells → magic-stripped concatenation → codeast skeleton (same rendering
as code.py); cell outputs ignored. Terminal lane like CODE (UC4): the
mechanical stub lands vault-terminal, enrichment is deferred refine.
"""
from __future__ import annotations

from pathlib import Path

from silica.config import CONFIG
from silica.kernel import codeast, gitstate, ipynb, paths
from silica.kernel.sanitize import strip_degenerate_runs
from silica.sources.base import GroundedStub, RawItem
from silica.sources.code import _repo_files, code_note_name, render_skeleton


class NotebookAdapter:
    name = "notebook"

    def matches(self, target: str) -> bool:
        return target.lower().endswith(".ipynb")

    def read(self, target: str) -> RawItem:
        vault = (CONFIG.vault_path or "").strip()
        if not vault:
            raise ValueError("no vault configured")
        root = paths.repo_root_for(vault)
        if root is None:
            raise ValueError("no code-lane repo (vault is not inside its git repo)")
        try:
            src = (Path(root) / target).resolve()
            src.relative_to(Path(root).resolve())
        except (ValueError, OSError):
            raise ValueError("path escapes the repository")
        if not src.is_file():
            raise ValueError(f"not a file: {target}")
        try:
            raw = src.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise ValueError(f"read failed: {e}")
        cells = ipynb.parse_cells(raw)  # ValueError on malformed nb — read() contract
        return RawItem(
            target=target,
            text=raw,
            meta={
                "code_ref": gitstate.head_ref(root) or "",
                "repo_root": str(root),
                "markdown_cells": cells.markdown,
                "code_source": cells.code,
                "language": ipynb.CODEAST_LANGUAGE.get(cells.language),
            },
        )

    def to_stub(self, item: RawItem) -> GroundedStub:
        path = item.target
        root = Path(item.meta["repo_root"])
        name = code_note_name(path)
        language = item.meta.get("language")
        code_source = item.meta.get("code_source") or ""

        sections: list[str] = []
        narrative = strip_degenerate_runs("\n\n".join(item.meta.get("markdown_cells") or []).strip())
        if narrative:
            sections.append(f"## Narrative\n\n{narrative}\n")
        if language is not None and code_source:
            sk = codeast.extract_skeleton(code_source, language, path=path)
            files = _repo_files(str(root), item.meta.get("code_ref", ""))
            sections.append(render_skeleton(sk, root, path, language, files))
        elif code_source:
            sections.append(
                "> Skeleton unavailable: unsupported kernel language. "
                "Narrative preserved; document the code manually.\n"
            )

        yaml_path = path.replace('"', '\\"')
        body = (
            f"---\n"
            f'documents:\n  - "{yaml_path}"\n'
            f"code_ref: {item.meta.get('code_ref', '')}\n"
            f"tags:\n  - codebase\n"
            f"---\n\n"
            f"# {name}\n\n"
            + "\n".join(sections)
        )
        inbox = (CONFIG.inbox_dir or "Inbox").strip("/")
        return GroundedStub(lane="terminal", note_path=f"{inbox}/{name}.md", body=body)


NOTEBOOK = NotebookAdapter()
