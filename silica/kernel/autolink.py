"""Deterministic wikilink injector for touched notes (Phase 4).

Rule (from the plan):
  "embeddings PROPOSE, graph DISPOSES"
  - `candidates`  — optional list of titles prioritized by embedding similarity.
  - `title_index` — authoritative list of titles that exist in the vault graph.
  A link is emitted ONLY when the title exists in `title_index`.  If `candidates`
  is given, only titles in candidates∩title_index are considered, which keeps
  the autolink pass focused and fast.

Skip regions (never modified):
  - YAML frontmatter  (--- block at the very top of the note)
  - Fenced code       (``` or ~~~ blocks)
  - Inline code       (`...`)
  - LaTeX math        ($...$  and  $$...$$)
  - Existing wikilinks ([[...]])
  - HTML comments     (<!-- ... -->)

Disambiguation rule:
  If `title_index` contains two entries that differ only in path but share the
  same display name, the caller must deduplicate them before passing — this
  function works on display names only and will happily link an ambiguous title.
  Use `build_title_index` (below) to get a pre-disambiguated index from the
  driver.

Idempotency: calling autolink twice on the same body is a no-op — any already-
linked title is in a skip region on the second pass.
"""
from __future__ import annotations

import re
from typing import Sequence

# ---------------------------------------------------------------------------
# Skip-region detection
# ---------------------------------------------------------------------------

# Matches the YAML frontmatter at the very top of a note (OFM convention)
_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n?", re.DOTALL)

# Fenced code blocks (``` or ~~~, any info string)
_FENCED_CODE_RE = re.compile(r"(?:```|~~~)[^\n]*\n.*?(?:```|~~~)", re.DOTALL)

# Inline code (`...`)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

# Display math ($$...$$) — must come before single-$ match
_DISPLAY_MATH_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)

# Inline math ($...$) — single-line only
_INLINE_MATH_RE = re.compile(r"\$[^$\n]+\$")

# Existing wikilinks [[...]]
_WIKILINK_RE = re.compile(r"\[\[[^\]]+\]\]")

# HTML comments
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# Heading lines (# ... at line start)
_HEADING_RE = re.compile(r"^#{1,6} .+$", re.MULTILINE)


def _build_skip_mask(text: str) -> list[bool]:
    """Return a per-character boolean mask: True = inside a skip region."""
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
        _WIKILINK_RE,
        _HTML_COMMENT_RE,
        _HEADING_RE,
    ):
        for m in pattern.finditer(text):
            _mark(m)

    return mask


# ---------------------------------------------------------------------------
# Main autolink function
# ---------------------------------------------------------------------------

def autolink(
    body: str,
    title_index: Sequence[str],
    candidates: Sequence[str] | None = None,
    self_title: str | None = None,
) -> tuple[str, list[str]]:
    """Wrap the first occurrence of each vault title in `body` with a wikilink.

    Args:
        body:        The full note text (including frontmatter if present).
        title_index: All vault titles that may be linked (pre-disambiguated).
        candidates:  Optional prioritized subset from embeddings.  If given,
                     only titles in candidates∩title_index are processed.
        self_title:  The title/basename of the note being processed.  When
                     provided, this title is excluded from linking — a note must
                     never contain a wikilink to itself.

    Returns:
        (new_body, added_links) — modified body and list of linked titles.

    Guarantees:
        - Never modifies text inside skip regions.
        - Only links titles that already exist in `title_index` (graph-safe).
        - At most one wikilink per title per call (first occurrence only).
        - Never creates a self-referential wikilink (self_title excluded).
        - Idempotent: running twice produces the same result.
    """
    if not body or not title_index:
        return body, []

    # Determine which titles to consider
    if candidates is not None:
        title_set = {t.lower(): t for t in title_index}
        work_titles = [t for t in candidates if t.lower() in title_set]
        # Canonicalise to the title_index spelling
        work_titles = [title_set[t.lower()] for t in work_titles]
    else:
        work_titles = list(title_index)

    # Exclude the note's own title to prevent self-referential wikilinks
    if self_title:
        _self_lower = self_title.lower()
        work_titles = [t for t in work_titles if t.lower() != _self_lower]

    if not work_titles:
        return body, []

    # Sort longest-first: prevents short titles from shadowing longer ones
    # ("Deep Learning" before "Learning")
    work_titles = sorted(work_titles, key=len, reverse=True)

    # Pre-scan: collect titles that are already wikilinked in the body.
    # A note should have at most one [[title]] link — if it's already there,
    # skip that title entirely regardless of where in the body it appears.
    existing_links: set[str] = {
        m.group(1).split("|")[0].strip().lower()   # handle [[Title|alias]] form
        for m in re.finditer(r"\[\[([^\]]+)\]\]", body)
    }

    added: list[str] = []
    current = body

    for title in work_titles:
        if len(title) < 2:
            continue  # single-character titles are too noisy

        if title.lower() in existing_links:
            continue  # already linked elsewhere in the note — skip

        # Build case-insensitive whole-word pattern
        escaped = re.escape(title)
        pattern = re.compile(
            r"(?<!\[)(?<!\w)" + escaped + r"(?!\w)(?!\])",
            re.IGNORECASE,
        )

        # Rebuild skip mask on the current body (changes after each substitution)
        mask = _build_skip_mask(current)

        # Find the first match that is NOT inside a skip region
        match = None
        for m in pattern.finditer(current):
            if not any(mask[i] for i in range(m.start(), m.end())):
                match = m
                break

        if match is None:
            continue

        # Preserve the exact casing from the body but link to the title
        link = f"[[{title}]]"
        current = current[: match.start()] + link + current[match.end() :]
        added.append(title)
        existing_links.add(title.lower())  # prevent duplicates within this call

    return current, added


# ---------------------------------------------------------------------------
# Reverse-link pass — inject links to newly created notes into pre-existing ones
# ---------------------------------------------------------------------------

def backlink_pass(
    new_titles: list[str],
    *,
    title_index: list[str],
    neighbourhood: list[str],
) -> dict[str, list[str]]:
    """For each note in `neighbourhood`, autolink only the `new_titles`.

    Runs `autolink(body, title_index, candidates=new_titles)` on every neighbour,
    wrapping mentions of newly-created notes with wikilinks in pre-existing content.
    Returns {path: titles_added}. Inherits all autolink() guarantees (graph-safe,
    skip-region aware, idempotent). Best-effort: per-note failures are logged and
    skipped.
    """
    import os as _os
    from silica.driver import DRIVER

    result: dict[str, list[str]] = {}
    for path in neighbourhood:
        try:
            nc = DRIVER.read_note(path)
            body = nc.content or ""
            if not body.strip():
                continue
            stem = _os.path.splitext(_os.path.basename(path))[0]
            new_body, added = autolink(body, title_index, candidates=new_titles, self_title=stem)
            if added:
                DRIVER.overwrite(path, new_body)
                result[path] = added
                import logging as _l
                _l.getLogger(__name__).info("BACKLINK: %s ← %s", path, added)
        except Exception as _e:
            import logging as _l
            _l.getLogger(__name__).debug("BACKLINK: skipped '%s' (non-fatal): %s", path, _e)
    return result


# ---------------------------------------------------------------------------
# Title index helpers
# ---------------------------------------------------------------------------

def build_title_index(refs: list) -> list[str]:
    """Build a disambiguated title list from driver NoteRef objects.

    Drops any title that appears more than once (basename conflict) — such
    titles cannot be safely linked without an explicit path qualifier.

    Args:
        refs: list of NoteRef objects with `.name` attribute.

    Returns:
        Sorted list of unique, unambiguous display names.
    """
    from collections import Counter

    name_counts: Counter[str] = Counter()
    for ref in refs:
        name = ref if isinstance(ref, str) else (getattr(ref, "name", None) or "")
        if name:
            name_counts[name] += 1

    return sorted(name for name, count in name_counts.items() if count == 1)
