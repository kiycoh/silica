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

from silica.kernel import codeast, frontmatter, gitstate
from silica.kernel.gitstate import CommitInfo

CHANGE_COSMETIC = "cosmetic"
CHANGE_STRUCTURAL = "structural"


@dataclass(frozen=True)
class StaleDoc:
    note_path: str          # vault-relative note path
    code_path: str          # repo-relative source path that changed
    recorded_ref: str       # code_ref stored in the note
    current_ref: str        # newest commit sha for code_path
    intervening: list[CommitInfo] = field(default_factory=list)
    change_level: str = CHANGE_STRUCTURAL   # conservative default (floor, not ceiling)
    details: list[str] = field(default_factory=list)


def documents_of(data: dict) -> list[str]:
    raw = (data or {}).get("documents")
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(x) for x in raw if x]


_documents_of = documents_of  # internal alias (pre-rename call sites)


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


def _skeleton_of(src: str, path: str, language):
    if path.lower().endswith(".ipynb"):
        from silica.kernel import ipynb
        cells = ipynb.parse_cells(src)          # ValueError → caller's fallback
        lang = ipynb.CODEAST_LANGUAGE.get(cells.language)
        if lang is None:
            raise ValueError("unsupported kernel language")
        return codeast.extract_skeleton(cells.code, lang, path=path)
    return codeast.extract_skeleton(src, language, path=path)


def classify_change(
    root: Path, base_ref: str, path: str, new_ref: str | None = None
) -> tuple[str, list[str]]:
    """Per-path verdict: skeleton of `path` at base_ref vs the working tree
    (or vs new_ref when given). Single conservative fallback branch: anything
    preventing structural analysis → STRUCTURAL with the named reason."""
    language = codeast.language_for(path)
    if language is None and not path.lower().endswith(".ipynb"):
        return CHANGE_STRUCTURAL, [f"{path}: no structural analysis (unsupported language)"]
    old_src = gitstate.show_file(root, base_ref, path)
    if old_src is None:
        return CHANGE_STRUCTURAL, [f"{path}: no structural analysis (ref {base_ref[:8]} unavailable)"]
    if new_ref is not None:
        new_src = gitstate.show_file(root, new_ref, path)
        if new_src is None:
            return CHANGE_STRUCTURAL, [f"{path}: deleted"]
    else:
        target = Path(root) / path
        if not target.is_file():
            return CHANGE_STRUCTURAL, [f"{path}: deleted"]
        try:
            new_src = target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return CHANGE_STRUCTURAL, [f"{path}: no structural analysis (read failed)"]
    try:
        old_sk = _skeleton_of(old_src, path, language)
        new_sk = _skeleton_of(new_src, path, language)
    except ValueError as e:
        return CHANGE_STRUCTURAL, [f"{path}: no structural analysis ({e})"]
    if old_sk.parse_error or new_sk.parse_error:
        return CHANGE_STRUCTURAL, [f"{path}: no structural analysis (parse failed)"]
    diff = codeast.diff_skeletons(old_sk, new_sk)
    if not diff:
        return CHANGE_COSMETIC, []
    return CHANGE_STRUCTURAL, [f"{path}: {d}" for d in diff]


def note_verdict(docs: list[StaleDoc]) -> tuple[str, list[str]]:
    """Aggregate per-path verdicts for one note (spec §2): a single
    STRUCTURAL path makes the note structural; details concatenate."""
    level = (CHANGE_STRUCTURAL
             if any(d.change_level == CHANGE_STRUCTURAL for d in docs)
             else CHANGE_COSMETIC)
    return level, [line for d in docs for line in d.details]


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
                level, details = classify_change(root, recorded, code_path)
                out.append(
                    StaleDoc(
                        note_path=note_path,
                        code_path=code_path,
                        recorded_ref=recorded,
                        current_ref=current,
                        intervening=gitstate.commits_since(root, recorded, code_path),
                        change_level=level,
                        details=details,
                    )
                )
    return out


def stale_count(vault: Path | str) -> int:
    """Count of stale (note, path) pairs. Soft-zero on any failure / no git."""
    try:
        return len(stale_docs(vault))
    except Exception:
        return 0
