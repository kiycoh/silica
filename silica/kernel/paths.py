# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Path canonicalization for vault notes and Silica runtime directories.

The CLI backend resolves notes by their vault-relative POSIX path. Any code
path that accepts a user- or agent-supplied note path MUST canonicalize it
through ``to_vault_relative`` before handing it to the driver — otherwise
absolute filesystem paths reach the Obsidian CLI verbatim and surface as a
misleading "No matches found" / "File not found", because the CLI indexes
by vault-relative path.

This module is the single source of truth for that normalization.
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

from silica.config import CONFIG

logger = logging.getLogger(__name__)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Torn-write-proof write: tmp file in the same dir, fsync, os.replace.

    For derived indexes and bundles rewritten in place — a crash or full
    disk mid-write must leave the previous file intact, not a truncated one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    # ponytail: no directory fsync — post-power-loss rename durability is
    # filesystem-dependent, and every caller's file is rebuildable or
    # re-produceable; upgrade if a real loss is ever traced here.


def quarantine(path: Path) -> Path | None:
    """Rename a corrupt state file aside — never clobbered, never deleted.

    Derived stores rebuild from empty afterwards; authoritative stores keep
    the bytes here for manual inspection. `silica doctor` surfaces any
    `*.corrupt.*` file it finds. Returns the quarantine path, or None if the
    rename itself failed (callers treat that as "proceed anyway": read paths
    must not raise).
    """
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    dest = path.with_name(f"{path.name}.corrupt.{stamp}")
    n = 0
    while dest.exists():  # same-second collision: bump, never overwrite
        n += 1
        dest = path.with_name(f"{path.name}.corrupt.{stamp}.{n}")
    try:
        path.rename(dest)
        return dest
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Silica runtime directory helpers
# ---------------------------------------------------------------------------

_SILICA_HOME = Path.home() / ".silica"


def silica_tmp_dir() -> Path:
    """Return the pipeline staging directory (~/.silica/tmp/), creating it if needed.

    All FSM temporary files (ops JSON, payload chunks, distiller output) live
    here instead of the system temp directory so they survive the pipeline run
    and are inspectable for debugging.  The FSM removes them on successful
    completion via _cleanup_tmp().
    """
    d = _SILICA_HOME / "tmp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def index_file(name: str) -> Path:
    """Resolve one derived-index file (``<name>.json``) under the current vault's
    index dir. The per-store ``_index_path`` shims delegate here (kept as module
    functions so tests can still monkeypatch them per store)."""
    return index_dir() / f"{name}.json"


def build_postings(docs: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    """Invert ``{doc: {term: count}}`` into the postings index ``{term: {doc: count}}``.
    The shared inner loop of every term/stem postings build (co-occurrence, lexical)."""
    idx: dict[str, dict[str, int]] = {}
    for doc, terms in docs.items():
        for term, count in terms.items():
            idx.setdefault(term, {})[doc] = count
    return idx


def path_keyed_singleton(cache: dict, key: str, factory):
    """Return cache[key], building it via factory() on first access.

    The shared shape behind every per-index-path store singleton (embed,
    co-occurrence, lexical): keying by resolved index path follows a /vault
    switch automatically. Callers own the cache dict (and its clear())."""
    inst = cache.get(key)
    if inst is None:
        inst = factory()
        cache[key] = inst
    return inst


def is_obsidian_vault(path) -> bool:
    """True when `path` is an Obsidian vault (carries a `.obsidian/` dir).

    This — not git presence — is the single signal that decides vault layout:
    an Obsidian vault is adopted verbatim (notes in its root), anything else is
    Silica repo mode (notes under `docs/silica`). Non-existent paths are False.
    """
    return (Path(path) / ".obsidian").is_dir()


def repo_mode_vault(root) -> Path:
    """Silica's notes location for a non-Obsidian target: `<root>/docs/silica`.

    Visible and committable next to the code it documents, unlike the old
    hidden `<root>/.silica`. ponytail: a plain dir with neither `.obsidian` nor
    git also lands here; telling a real repo from a bare dir would need a
    stronger signal (source files? a marker?) — not worth it until it bites.
    """
    return Path(root) / "docs" / "silica"


def resolve_repo_root(vault: str | Path) -> tuple[Path | None, str | None]:
    """Code-lane repo root for `vault`, validating the vault⊂repo invariant (ADR-0019).

    Returns (root, warning). Valid layouts: a repo-mode vault (`<root>/docs/silica`
    or any plain dir — git discovers the target repo above it) and an Obsidian
    vault that is itself the repo root. An Obsidian vault nested inside a
    FOREIGN git repo yields (None, warning): code lane disabled, never grounded
    on the wrong repo. (None, None) when git is absent or no repo contains the
    vault. Pure resolution, no caching — see `repo_root_for`.
    """
    from silica.kernel import gitstate

    v = Path(vault).resolve()
    root = gitstate.find_repo_root(v)
    if root is None:
        return None, None
    if is_obsidian_vault(v) and root != v:
        return None, (
            f"code lane disabled: Obsidian vault {v} is nested inside "
            f"foreign git repo {root}"
        )
    return root, None


# Resolved-once storage for the code-lane root (ADR-0019), keyed by resolved
# vault path so it follows /vault switches and stays correct for entry points
# that never run the CLI startup (GUI, MCP). "No repo" results are NOT cached,
# so a `git init` after first resolution is still picked up.
_REPO_ROOT_CACHE: dict[str, tuple[Path | None, str | None]] = {}


def _repo_root_resolved(vault: str | Path) -> tuple[Path | None, str | None]:
    raw = str(vault or "").strip()
    if not raw:
        return None, None
    key = str(Path(raw).resolve())
    hit = _REPO_ROOT_CACHE.get(key)
    if hit is not None:
        return hit
    root, warn = resolve_repo_root(raw)
    if warn:
        logger.warning(warn)
    if root is not None or warn is not None:
        _REPO_ROOT_CACHE[key] = (root, warn)
    return root, warn


def repo_root_for(vault: str | Path) -> Path | None:
    """The single choke point every code-lane consumer derives its repo root
    through (ADR-0019): resolved once per vault, invariant-validated, warning
    logged once. None ⇒ code lane disabled for this vault."""
    return _repo_root_resolved(vault)[0]


def repo_root_warning(vault: str | Path) -> str | None:
    """The invariant-violation message for `vault`, if any — for the CLI to
    surface loudly at startup and /vault switch."""
    return _repo_root_resolved(vault)[1]


def clear_repo_root_cache() -> None:
    """Drop cached repo-root resolutions (test isolation)."""
    _REPO_ROOT_CACHE.clear()


def index_dir_for(vault: str) -> Path:
    """Per-vault index namespace for an explicit `vault` path, independent of
    the global CONFIG singleton. Same digest scheme as `index_dir()` — the
    two agree whenever `vault == CONFIG.vault_path`.

    Callers that need to resolve a *specific* vault's on-disk index (e.g. a
    diagnostic comparing a passed-in config's vault against whatever vault
    the live global CONFIG currently points at) MUST use this rather than
    `index_dir()`, which only ever resolves the global singleton and would
    silently compare the wrong vault's state.
    """
    base = _SILICA_HOME / "index"
    vault = (vault or "").strip()
    if not vault:
        return base
    digest = hashlib.sha1(str(Path(vault).resolve()).encode("utf-8")).hexdigest()[:12]
    return base / digest


def index_dir() -> Path:
    """Per-vault index namespace: ~/.silica/index/<digest12>/ keyed by the
    resolved vault path; legacy global ~/.silica/index/ when no vault is
    configured. Per-vault state follows the vault (ADR-0014), so /vault
    switch no longer serves another vault's entries."""
    return index_dir_for(getattr(CONFIG, "vault_path", "") or "")


def to_vault_relative(path: str, *, ensure_md: bool = True) -> str:
    """Normalize an arbitrary note path to POSIX vault-relative form.

    Rules:
      - already-relative paths pass through (POSIX-normalized, leading
        slashes stripped);
      - absolute paths *under* the configured vault root are relativized;
      - absolute paths *outside* the vault raise ``ValueError`` with a
        clear diagnostic — they would otherwise become a silent
        "File not found" when the CLI fails to resolve them;
      - if ``ensure_md`` is True (default) and the result does not end in
        ``.md``, the extension is appended.

    The vault root is read at call time from ``CONFIG.vault_path`` so test
    fixtures that mutate the singleton continue to work.
    """
    if not path:
        raise ValueError("Empty path is not a valid vault reference")

    p = Path(path)
    if p.is_absolute():
        vault_str = getattr(CONFIG, "vault_path", None) or ""
        if not vault_str:
            raise ValueError(
                f"Absolute path {path!r} provided but SILICA_VAULT is not configured"
            )
        vault = Path(vault_str)
        try:
            p = p.relative_to(vault)
        except ValueError as exc:
            raise ValueError(
                f"Path {path!r} is outside the configured vault "
                f"{vault.as_posix()!r}"
            ) from exc

    rel = p.as_posix().strip("/")
    if ensure_md and not rel.endswith(".md"):
        rel += ".md"
    return rel


def is_inbox_path(path: str) -> bool:
    """True when a vault-relative path sits anywhere under the configured
    inbox root (case-insensitive). The inbox is staging, never a write or
    merge target — callers use this to filter candidates and reject ops.
    """
    inbox = getattr(CONFIG, "inbox_dir", None)
    root = inbox.strip("/") if isinstance(inbox, str) and inbox.strip("/") else "Inbox"
    return path.replace("\\", "/").lstrip("/").casefold().startswith(root.casefold() + "/")


def resolve_target_dir(target_dir: str) -> str:
    """Fold a user-typed vault folder onto the existing tree, case-insensitively.

    'Informatica/Intelligenza Artificiale' typed against a vault holding
    'Informatica/Intelligenza artificiale' silently forks the tree on a
    case-sensitive filesystem: new-note writes ENOENT through the Obsidian
    bridge and patch paths mismatch their expected collisions. Each segment
    adopts the casing of an existing folder when one matches case-insensitively;
    unmatched segments keep the typed casing (a genuinely new folder).
    Absolute paths and unconfigured vaults pass through untouched.
    """
    vault_str = getattr(CONFIG, "vault_path", None) or ""
    if not target_dir or not vault_str or Path(target_dir).is_absolute():
        return target_dir
    base = Path(vault_str)
    resolved: list[str] = []
    for seg in Path(target_dir.strip("/")).parts:
        cur = base.joinpath(*resolved)
        if not (cur / seg).is_dir() and cur.is_dir():
            seg = next(
                (e.name for e in cur.iterdir()
                 if e.is_dir() and e.name.casefold() == seg.casefold()),
                seg,
            )
        resolved.append(seg)
    return "/".join(resolved)
