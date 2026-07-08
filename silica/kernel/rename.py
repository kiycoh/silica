# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Pure link-rewriting kernel for note move/rename operations (Phase 1a).

Mirrors Obsidian's "automatically update internal links" behaviour.

Public API
----------
    rewrite_links(content, old_path, new_path, *, rewrite_name_links=True)
        → (new_content, n_rewritten)

All logic is pure (no I/O, no driver imports).  Wire-up to the FS backend
is Task 2 and lives in silica/driver/fs_backend.py.

Skip regions (never modified):
    - YAML frontmatter  (--- block at the very top of the note)
    - Fenced code       (``` or ~~~ blocks)
    - Inline code       (`...`)
    - Display math      ($$...$$)
    - Inline math       ($...$)

See silica/kernel/autolink.py for the masking idiom this module mirrors.
"""
from __future__ import annotations

import os
import re
from urllib.parse import unquote, quote


# ---------------------------------------------------------------------------
# Skip-region mask (same idiom as autolink.py)
# ---------------------------------------------------------------------------

# YAML frontmatter at the very top of the note (OFM convention)
_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n?", re.DOTALL)

# Fenced code blocks (``` or ~~~, any info string)
_FENCED_CODE_RE = re.compile(r"(?:```|~~~)[^\n]*\n.*?(?:```|~~~)", re.DOTALL)

# Inline code (`...`)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

# Display math ($$...$$) — must come before single-$ match
_DISPLAY_MATH_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)

# Inline math ($...$) — single-line only
_INLINE_MATH_RE = re.compile(r"\$[^$\n]+\$")


def _build_skip_mask(text: str) -> list[bool]:
    """Return a per-character boolean mask: True = inside a skip region.

    Matches the contract from autolink._build_skip_mask but without the
    existing-wikilink and HTML-comment entries (we *want* to scan inside
    wikilinks for rewriting).
    """
    mask = [False] * len(text)

    def _mark(m: re.Match) -> None:
        for i in range(m.start(), m.end()):
            mask[i] = True

    for pattern in (
        _FRONTMATTER_RE,
        _FENCED_CODE_RE,
        _INLINE_CODE_RE,
        _DISPLAY_MATH_RE,
        _INLINE_MATH_RE,
    ):
        for m in pattern.finditer(text):
            _mark(m)

    return mask


def _span_is_clear(mask: list[bool], start: int, end: int) -> bool:
    """Return True iff no character in [start, end) is masked."""
    return not any(mask[i] for i in range(start, end))


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _stem(vault_path: str) -> str:
    """Basename without extension (e.g. 'Folder/Note.md' → 'Note')."""
    return os.path.splitext(os.path.basename(vault_path))[0]


def _same_basename(old: str, new: str) -> bool:
    """True when old and new share the same (case-insensitive) stem."""
    return _stem(old).lower() == _stem(new).lower()


def _path_without_ext(vault_path: str) -> str:
    """Vault-relative path with extension stripped ('Folder/Note.md' → 'Folder/Note')."""
    return os.path.splitext(vault_path)[0]


# ---------------------------------------------------------------------------
# Wikilink rewriter
# ---------------------------------------------------------------------------

# Matches optional embed bang + full wikilink: !?[[target(#^suffix)?(|alias)?]]
# Group 1: "!" or ""
# Group 2: the target (everything before # or ^ or |)
# Group 3: suffix (#... or ^...) — may be empty
# Group 4: alias part (|...) — may be empty
_WIKILINK_FULL_RE = re.compile(
    r"(!?)"                         # optional embed
    r"\[\["
    r"([^\]#^|]*)"                  # target (no suffix, no alias delimiters)
    r"([#^][^\]|]*)?"               # optional heading/block suffix
    r"(\|[^\]]*)?"                  # optional alias
    r"\]\]",
    re.DOTALL,
)


def _rewrite_wikilinks(
    content: str,
    mask: list[bool],
    old_path: str,
    new_path: str,
    *,
    rewrite_name_links: bool,
) -> tuple[str, int]:
    """Rewrite wikilinks in *content* that point to old_path.

    Handles:
    - Name-based links (target == old basename, no path separator)
    - Path-based links (target contains a '/')

    Returns (new_content, n_rewritten).
    """
    old_stem = _stem(old_path)
    new_stem = _stem(new_path)
    old_without_ext = _path_without_ext(old_path)  # e.g. "Folder/Old"
    new_without_ext = _path_without_ext(new_path)  # e.g. "Folder/New"
    basename_changed = not _same_basename(old_path, new_path)

    # Build list of (match, replacement) pairs so we can apply back-to-front
    replacements: list[tuple[re.Match, str]] = []

    for m in _WIKILINK_FULL_RE.finditer(content):
        if not _span_is_clear(mask, m.start(), m.end()):
            continue

        bang = m.group(1)        # "!" or ""
        target = m.group(2)      # raw target string (may have leading space)
        suffix = m.group(3) or ""   # "#Section" or "^blockid" or ""
        alias = m.group(4) or ""    # "|alias text" or ""

        target_stripped = target.strip()

        # ---------------------------------------------------------------
        # Path-based link: target contains a "/" → match by full path
        # ---------------------------------------------------------------
        if "/" in target_stripped:
            # Decode the stored target to a vault-relative path
            # Obsidian stores paths without extension sometimes, so check both
            target_lower = target_stripped.lower()
            old_without_ext_lower = old_without_ext.lower()
            old_path_lower = old_path.lower()

            if (
                target_lower == old_without_ext_lower
                or target_lower == old_path_lower
            ):
                # Preserve presence/absence of .md extension as written
                has_ext = target_stripped.lower().endswith(".md")
                new_target = new_without_ext if not has_ext else new_path
                # Preserve the canonical casing from new_path
                replacement = f"{bang}[[{new_target}{suffix}{alias}]]"
                replacements.append((m, replacement))

            continue  # path-based — no fallthrough to name-based

        # ---------------------------------------------------------------
        # Name-based link: no "/" in target — match by basename
        # ---------------------------------------------------------------
        if not rewrite_name_links:
            continue
        if not basename_changed:
            continue  # fast-path: identical basename → nothing to do

        # The target might include a heading suffix already split out,
        # but target itself is the raw basename ref (e.g. "Old Note").
        # Case-insensitive whole-target comparison (target_stripped vs old_stem).
        if target_stripped.lower() == old_stem.lower():
            replacement = f"{bang}[[{new_stem}{suffix}{alias}]]"
            replacements.append((m, replacement))

    # Apply replacements back-to-front to keep earlier offsets valid
    result = content
    for m, repl in reversed(replacements):
        result = result[: m.start()] + repl + result[m.end() :]

    return result, len(replacements)


# ---------------------------------------------------------------------------
# Markdown link rewriter
# ---------------------------------------------------------------------------

# Matches Markdown links: [text](href)  or  [text](<href>)
# Group 1: link text
# Group 2: "<" if angle-bracket form, else ""
# Group 3: the raw href (inside brackets or angle brackets)
# Group 4: ">" if angle-bracket form, else ""
_MD_LINK_RE = re.compile(
    r"\[([^\]]*)\]"         # [link text]
    r"\((<?)"               # opening paren + optional <
    r"([^)>]*)"             # href
    r"(>?)\)",              # optional > + closing paren
)


def _rewrite_markdown_links(
    content: str,
    mask: list[bool],
    old_path: str,
    new_path: str,
) -> tuple[str, int]:
    """Rewrite [text](href) links in *content* that resolve to old_path.

    - Only rewrites relative hrefs (skips http/https/mailto/#...).
    - Handles %20-encoded spaces: decodes to compare, re-encodes same way.
    - Handles angle-bracket form: [text](<href>) — brackets preserved.
    - Case-insensitive path comparison.

    Returns (new_content, n_rewritten).
    """
    old_path_lower = old_path.lower()

    replacements: list[tuple[re.Match, str]] = []

    for m in _MD_LINK_RE.finditer(content):
        if not _span_is_clear(mask, m.start(), m.end()):
            continue

        text = m.group(1)
        open_angle = m.group(2)   # "<" or ""
        raw_href = m.group(3)
        close_angle = m.group(4)  # ">" or ""

        # Skip non-relative hrefs
        if raw_href.startswith(("http://", "https://", "mailto:", "#")):
            continue

        # Split off any URL fragment (#...) — preserved verbatim on rewrite
        if "#" in raw_href:
            hash_pos = raw_href.index("#")
            href_path = raw_href[:hash_pos]    # portion before the first #
            fragment = raw_href[hash_pos:]     # "#Section" or "#^blockid" etc.
        else:
            href_path = raw_href
            fragment = ""

        # Decode %20 to compare against old_path (fragment excluded)
        decoded_href = unquote(href_path)

        if decoded_href.lower() != old_path_lower:
            continue

        # Decide how to re-encode the new path
        # If original href used %20 encoding, re-encode spaces the same way
        original_had_percent = "%20" in href_path
        if original_had_percent:
            new_href = quote(new_path, safe="/")
        else:
            new_href = new_path

        replacement = f"[{text}]({open_angle}{new_href}{fragment}{close_angle})"
        replacements.append((m, replacement))

    result = content
    for m, repl in reversed(replacements):
        result = result[: m.start()] + repl + result[m.end() :]

    return result, len(replacements)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rewrite_links(
    content: str,
    old_path: str,
    new_path: str,
    *,
    rewrite_name_links: bool = True,
) -> tuple[str, int]:
    """Rewrite links in *content* that point to old_path so they point to
    new_path.  Paths are vault-relative (e.g. ``"Folder/Old Note.md"``).

    Returns ``(new_content, n_rewritten)``.  Pure — no I/O.

    Args:
        content:            Full note text (may include frontmatter).
        old_path:           Vault-relative path before move/rename.
        new_path:           Vault-relative path after move/rename.
        rewrite_name_links: When False, skip name-based wikilink rewriting
                            (the driver sets this for ambiguous basenames).
                            Path-based wikilinks and Markdown links are always
                            rewritten regardless of this flag.

    Behaviours
    ----------
    1. Name-based wikilink, basename UNCHANGED (pure folder move):
       ``[[Note]]`` is NOT rewritten — name resolution still holds.
       Fast-path: when old and new stems are identical, name-based pass is
       skipped entirely; path-based links are still processed.

    2. Name-based wikilink, basename CHANGED:
       ``[[Old]]`` → ``[[New]]``; alias preserved: ``[[Old|alias]]`` →
       ``[[New|alias]]``.  Embeds too: ``![[Old]]`` → ``![[New]]``.

    3. Heading / block suffixes preserved verbatim:
       ``[[Old#Section]]`` → ``[[New#Section]]``.

    4. Path-based wikilinks (target contains ``/``):
       Matched case-insensitively by full vault path; extension presence/
       absence in the link is preserved.  Always rewritten, even when
       ``rewrite_name_links=False``.

    5. Markdown links [text](href):
       Only relative hrefs that resolve to old_path are rewritten.
       %20 and angle-bracket forms are handled and round-tripped.

    6. Skip regions — never rewritten:
       frontmatter, fenced code, inline code, display math, inline math.

    7. Case-insensitive matching; whole-target only (``[[Older]]`` is safe).
    """
    if not content:
        return content, 0

    mask = _build_skip_mask(content)

    result, n_wiki = _rewrite_wikilinks(
        content, mask, old_path, new_path,
        rewrite_name_links=rewrite_name_links,
    )

    # Rebuild mask after wikilink rewrites (positions shift)
    if n_wiki:
        mask = _build_skip_mask(result)

    result, n_md = _rewrite_markdown_links(result, mask, old_path, new_path)

    return result, n_wiki + n_md
