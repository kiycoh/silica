# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""gitstate — a deterministic, soft-degrading wrapper over the `git` CLI.

No git library: plain `git` via subprocess (ADR-0009 — provider-free).
Every function degrades soft: git binary missing, not a repo, repo with no
commits, or detached/broken state all yield None/empty and never raise toward
callers. Silica without git behaves exactly as it does today.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

_TIMEOUT_S = 10

# Field separator unlikely to appear in a commit subject (ASCII unit separator).
_FS = "\x1f"


def _run(args: list[str], cwd: Path | str) -> subprocess.CompletedProcess | None:
    """Run a git command, returning the CompletedProcess or None on any failure."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None


def find_repo_root(path: Path | str) -> Path | None:
    """Return the repo top-level for `path`, or None if not inside a git repo."""
    p = Path(path)
    cwd = p if p.is_dir() else p.parent
    proc = _run(["rev-parse", "--show-toplevel"], cwd)
    if proc is None or proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return Path(out).resolve() if out else None


def head_ref(root: Path | str) -> str | None:
    """Return the 40-char HEAD sha, or None on an empty/headless repo."""
    proc = _run(["rev-parse", "HEAD"], root)
    if proc is None or proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return out if len(out) == 40 else None


@dataclass(frozen=True)
class CommitInfo:
    """A single commit touching a path. `subject` is UNTRUSTED text — it is
    never interpolated into a worker prompt (history-as-knowledge is deferred);
    it appears only in CLI reports."""

    sha: str
    committed_at: str  # ISO 8601 (committer date)
    subject: str


def _parse_log(stdout: str) -> list[CommitInfo]:
    out: list[CommitInfo] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split(_FS)
        if len(parts) == 3:
            out.append(CommitInfo(sha=parts[0], committed_at=parts[1], subject=parts[2]))
    return out


def is_ignored(root: Path | str, paths: list[Path]) -> set[Path]:
    """Return the subset of `paths` ignored by git. Empty set on any failure.

    Paths are interpreted relative to `root`. Works for any vault (an Obsidian
    vault under git benefits too), not just codebase mode. `check-ignore` needs
    stdin, which the shared `_run` helper does not provide, so this calls
    subprocess directly.
    """
    if not paths:
        return set()
    stdin = "\n".join(str(p) for p in paths)
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "--stdin"],
            cwd=str(root),
            input=stdin,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return set()
    # exit 0 = some ignored, 1 = none ignored, >1 = error (treat as none).
    if proc.returncode not in (0, 1):
        return set()
    return {Path(line) for line in proc.stdout.splitlines() if line.strip()}


def log_for_path(root: Path | str, path: str, limit: int = 1) -> list[CommitInfo]:
    """Latest `limit` commits touching `path` (rename-following). Empty on failure."""
    proc = _run(
        ["log", f"-{limit}", "--follow", f"--format=%H{_FS}%cI{_FS}%s", "--", path],
        root,
    )
    if proc is None or proc.returncode != 0:
        return []
    return _parse_log(proc.stdout)


def commits_since(root: Path | str, since_ref: str, path: str) -> list[CommitInfo]:
    """Commits touching `path` after `since_ref` (newest-first, `since_ref`
    excluded). Empty on failure or when `since_ref` is unknown."""
    if not since_ref:
        return []
    proc = _run(
        ["log", f"--format=%H{_FS}%cI{_FS}%s", f"{since_ref}..HEAD", "--", path],
        root,
    )
    if proc is None or proc.returncode != 0:
        return []
    return _parse_log(proc.stdout)


def commit_docs(
    root: Path | str,
    vault: Path | str,
    paths: list[Path],
    message: str,
) -> str | None:
    """Commit `paths` (which must all live under `vault`) with `message`.

    Hard-refuses any path outside `vault` — a bug must never commit source
    files. Returns the new HEAD sha, or None if there is nothing to commit,
    a path is out of bounds, or git is unavailable.
    """
    vault_resolved = Path(vault).resolve()
    rel_args: list[str] = []
    for p in paths:
        try:
            resolved = Path(p).resolve()
            resolved.relative_to(vault_resolved)  # raises if outside vault
        except (ValueError, OSError):
            return None
        rel_args.append(str(resolved))

    if not rel_args:
        return None

    add = _run(["add", "--", *rel_args], root)
    if add is None or add.returncode != 0:
        return None
    commit = _run(["commit", "-q", "-m", message, "--", *rel_args], root)
    if commit is None or commit.returncode != 0:
        return None  # e.g. nothing staged → non-zero, treated as no-op
    return head_ref(root)
