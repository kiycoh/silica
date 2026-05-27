"""Path canonicalization for vault notes.

The CLI backend resolves notes by their vault-relative POSIX path. Any code
path that accepts a user- or agent-supplied note path MUST canonicalize it
through ``to_vault_relative`` before handing it to the driver — otherwise
absolute filesystem paths reach the Obsidian CLI verbatim and surface as a
misleading "No matches found" / "File not found", because the CLI indexes
by vault-relative path.

This module is the single source of truth for that normalization.
"""
from __future__ import annotations

from pathlib import Path

from silica.config import CONFIG


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
