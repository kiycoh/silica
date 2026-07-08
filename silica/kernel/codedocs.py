# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""codedocs — doc↔source staleness for codebase mode.

A note documents source files via frontmatter:
    documents: [src/m.py, src/n.py]   # repo-relative paths
    code_ref: <sha>                   # HEAD when last verified

A note is stale if any referenced path's newest commit differs from code_ref.
Staleness state lives in per-note frontmatter (ADR decision in the spec), not a
central index. All git access goes through gitstate and degrades soft.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from silica.kernel import frontmatter, gitstate
from silica.kernel.gitstate import CommitInfo


@dataclass(frozen=True)
class StaleDoc:
    note_path: str          # vault-relative note path
    code_path: str          # repo-relative source path that changed
    recorded_ref: str       # code_ref stored in the note
    current_ref: str        # newest commit sha for code_path
    intervening: list[CommitInfo] = field(default_factory=list)


def _documents_of(data: dict) -> list[str]:
    raw = (data or {}).get("documents")
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(x) for x in raw if x]


def iter_documenting_notes(vault: Path | str):
    """Yield (note_path, data, body) for every note carrying `documents:`."""
    vault = Path(vault)
    for md in sorted(vault.rglob("*.md")):
        try:
            content = md.read_text(encoding="utf-8")
        except OSError:
            continue
        data, _, body = frontmatter.split(content)
        if not data or not _documents_of(data):
            continue
        yield md.relative_to(vault).as_posix(), data, body


def stale_docs(vault: Path | str, repo_root: Path | str | None = None) -> list[StaleDoc]:
    """Return one StaleDoc per (note, changed path). Empty when git is absent."""
    vault = Path(vault)
    root = Path(repo_root) if repo_root else gitstate.find_repo_root(vault)
    if root is None:
        return []

    out: list[StaleDoc] = []
    for note_path, data, _ in iter_documenting_notes(vault):
        recorded = str(data.get("code_ref") or "").strip()
        if not recorded:
            continue  # unknown → not stale
        for code_path in _documents_of(data):
            latest = gitstate.log_for_path(root, code_path, limit=1)
            if not latest:
                continue  # path has no history → unknown, not stale
            current = latest[0].sha
            if current != recorded:
                out.append(
                    StaleDoc(
                        note_path=note_path,
                        code_path=code_path,
                        recorded_ref=recorded,
                        current_ref=current,
                        intervening=gitstate.commits_since(root, recorded, code_path),
                    )
                )
    return out


def stale_count(vault: Path | str) -> int:
    """Count of stale (note, path) pairs. Soft-zero on any failure / no git."""
    try:
        return len(stale_docs(vault))
    except Exception:
        return 0
