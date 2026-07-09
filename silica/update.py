# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""`silica update` — pull the latest source, refusing to land code that
doesn't even byte-compile, and refresh deps only when they changed.

Silica installs as an editable git checkout (`uv pip install -e .`), so the
working tree *is* the running code: updating is a `git pull`, and there is no
reinstall for pure-Python changes. The one corruption risk — pulling code with
a syntax error, which would brick the CLI on next launch — is guarded by a
post-pull `compileall` that rolls back to the pre-pull commit on failure.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root of the editable install


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def update(check_only: bool = False) -> int:
    if not (ROOT / ".git").is_dir():
        print("✗ Not a git checkout — reinstall from source to update.")
        return 1
    if _git("fetch", "--quiet").returncode != 0:
        print("✗ Fetch failed — check your network.")
        return 1

    counted = _git("rev-list", "--count", "HEAD..@{u}")
    if counted.returncode != 0:
        print("✗ No upstream branch to compare against.")
        print("  Set one with: git branch --set-upstream-to=origin/main")
        return 1
    ahead = counted.stdout.strip()
    if ahead in ("", "0"):
        print("✓ Already up to date.")
        return 0
    print(f"→ {ahead} new commit(s) available.")
    if check_only:  # pure query — a dirty tree is fine, we touch nothing
        return 0
    if _git("status", "--porcelain").stdout.strip():
        # ponytail: abort, don't stash. A shipped checkout is clean; add
        # auto-stash only if users routinely edit their install in place.
        print("✗ Uncommitted local changes — commit or stash them, then retry.")
        return 1

    old = _git("rev-parse", "HEAD").stdout.strip()
    changed = _git("diff", "--name-only", "HEAD", "@{u}").stdout
    if _git("merge", "--ff-only", "@{u}").returncode != 0:
        print("✗ Fast-forward not possible (history diverged) — resolve manually.")
        return 1

    # Corruption guard: never keep code that doesn't byte-compile.
    if subprocess.run(
        [sys.executable, "-m", "compileall", "-q", str(ROOT / "silica")]
    ).returncode != 0:
        print("✗ Pulled code has a syntax error — rolling back.")
        _git("reset", "--hard", old)
        return 1

    if "pyproject.toml" in changed:
        print("→ Dependencies changed, reinstalling…")
        if subprocess.run(["uv", "pip", "install", "-e", "."], cwd=ROOT).returncode != 0:
            print("✗ Reinstall failed — code is updated but deps may be stale.")
            print("  Retry manually: uv pip install -e .")
            return 1

    print("✓ Updated. Restart silica to load the new version.")
    return 0


def behind_count() -> int:
    """Commits the local checkout is behind upstream (0 if unknown/current).

    Reads the local tracking ref — no network. Fires a background fetch at most
    once/day so the ref stays roughly fresh; its result shows on the next launch.
    """
    git = ROOT / ".git"
    if not git.is_dir():
        return 0
    try:
        fetch_head = git / "FETCH_HEAD"
        stale = not fetch_head.exists() or time.time() - fetch_head.stat().st_mtime > 86_400
        if stale:
            # ponytail: fire-and-forget — never blocks startup. Leaves one
            # short-lived zombie per session until exit; add a reaper if it matters.
            subprocess.Popen(
                ["git", "fetch", "--quiet"],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        out = _git("rev-list", "--count", "HEAD..@{u}").stdout.strip()
        return int(out) if out.isdigit() else 0
    except Exception:
        return 0  # offline / detached HEAD / no upstream → no nudge
