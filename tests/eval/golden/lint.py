# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""OFM well-formedness lint for the golden harness (probe_integrity only).

Two uses:
  * absolute ``scan(text)`` — informational ground-truth counts on the human vault
    (measures the human, not the pipeline).
  * ``new_violations(before, after)`` — the *gated* differential: violations a
    write-path transform INTRODUCES. Stable false positives cancel under the
    Counter subtraction, so absolute lint precision is only cosmetic.

Catalog = 16 structural + 6 style checks (spec §probe_integrity), capped at v1 —
tune a firing check, don't expand. Not ``ofm.ofm_lint`` (manifest-dependent
conventions, out of scope here).

Code-region handling: markdown_it (``ast.get_non_code_text``) strips the exact
tokens the math/latex/emphasis checks need (``$``, ``\\``, ``**``), so a local
regex removes code instead — see ``_strip_code``.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable, NamedTuple

from silica.kernel import frontmatter
from silica.kernel.ast import extract_callouts, parse_headings
from silica.kernel.ofm import CALLOUT_TYPES
from silica.kernel.title import title_key

# ---------------------------------------------------------------------------
# Code stripping — regex, NOT markdown_it (preserves $, \\, ** for the checks)
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")


def _strip_code(body: str) -> str:
    """Body with fenced + inline code removed (but $, \\, ** untouched)."""
    return _INLINE_CODE.sub("", _FENCE.sub("", body))


class Ctx(NamedTuple):
    """Precomputed once per scan — probe_integrity does 9 scans/note, so the
    markdown_it parses (headings, callouts) must not be redone per check."""

    text: str                 # full note (frontmatter + body)
    data: object              # parsed frontmatter dict, or None on YAML error / no FM
    raw: str | None           # raw frontmatter block, or None when absent
    body: str                 # note body (frontmatter stripped)
    noncode: str              # body with code regions stripped (regex, not markdown_it)
    headings: list            # parse_headings(body) — text INCLUDES ** markers
    callouts: list            # extract_callouts(body)
    stem: str | None          # note basename (for h1-title-mismatch)


def _build_ctx(text: str, stem: str | None) -> Ctx:
    data, raw, body = frontmatter.split(text)
    return Ctx(
        text=text,
        data=data,
        raw=raw,
        body=body,
        noncode=_strip_code(body),
        headings=parse_headings(body),
        callouts=extract_callouts(body),
        stem=stem,
    )


# ---------------------------------------------------------------------------
# Structural checks — routed per handoff: math → noncode; latex/emphasis/
# wikilink-patterns → raw body; bracket-balance → noncode.
# ---------------------------------------------------------------------------

_DOUBLE_LATEX = re.compile(r"(?<!\\)\\\\[a-zA-Z]{2,}")        # literal \\frac (LLM artifact)
_LITERAL_BACKSLASH_N = re.compile(r"\\n(?![a-zA-Z])")        # \n artifact, not \nabla/\neq
_EMPTY_DISPLAY_MATH = re.compile(r"\$\$\s*\$\$")
_EMPTY_INLINE_MATH = re.compile(r"(?<!\$)\$[ \t]*\$(?!\$)")
_EMPTY_WIKILINK = re.compile(r"!?\[\[\s*(?:#[^\]]*)?\]\]")   # [[]], ![[]], [[#anchor-only]]
_BROKEN_ALIAS = re.compile(r"\[\[[^\]|]*\|\s*\]\]")          # [[x|]]
_ZW_IN_LINK = re.compile(r"\[\[[^\]]*[​‌‍﻿ ][^\]]*\]\]")
_FENCE_WRAP = re.compile(r"\A\s*```(?:markdown|md)\b", re.IGNORECASE)
_MERMAID_BLOCK = re.compile(r"```mermaid[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_MERMAID_KEYWORDS = (
    "graph", "flowchart", "sequencediagram", "classdiagram", "statediagram",
    "erdiagram", "gantt", "pie", "mindmap",
)
_PLACEHOLDERS = ("(da espandere)", "(to be expanded)", "(da completare)", "lorem ipsum")


def _unclosed_code_fence(c: Ctx) -> int:
    return c.body.count("```") % 2


def _unbalanced_display_math(c: Ctx) -> int:
    return c.noncode.count("$$") % 2


def _unbalanced_inline_math(c: Ctx) -> int:
    # single $ left after removing $$ pairs
    return c.noncode.replace("$$", "").count("$") % 2


def _empty_math_block(c: Ctx) -> int:
    return len(_EMPTY_DISPLAY_MATH.findall(c.noncode)) + len(_EMPTY_INLINE_MATH.findall(c.noncode))


def _double_escaped_latex(c: Ctx) -> int:
    return len(_DOUBLE_LATEX.findall(c.body))


def _frontmatter_yaml_error(c: Ctx) -> int:
    return int(c.data is None and c.raw is not None)


def _unclosed_frontmatter(c: Ctx) -> int:
    # opens with a `---` line at the very top but FM_RE found no closing delimiter
    if re.match(r"\A---[ \t]*\r?\n", c.text) and frontmatter.FM_RE.match(c.text) is None:
        return 1
    return 0


def _duplicate_frontmatter(c: Ctx) -> int:
    # a second frontmatter block at the top of the body
    return int(frontmatter.FM_RE.match(c.body) is not None)


def _malformed_wikilink(c: Ctx) -> int:
    n = (
        len(_EMPTY_WIKILINK.findall(c.body))
        + len(_BROKEN_ALIAS.findall(c.body))
        + len(_ZW_IN_LINK.findall(c.body))
    )
    # bracket-balance on noncode (mirrors ast._balanced)
    if c.noncode.count("[[") != c.noncode.count("]]"):
        n += 1
    return n


def _bad_chars(c: Ctx) -> int:
    return c.text.count("�") + c.text.count("\x00")


def _literal_backslash_n(c: Ctx) -> int:
    return len(_LITERAL_BACKSLASH_N.findall(c.body))


def _markdown_fence_wrapper(c: Ctx) -> int:
    return int(_FENCE_WRAP.match(c.body) is not None)


def _unclosed_html_comment(c: Ctx) -> int:
    return int(c.body.count("<!--") != c.body.count("-->"))


def _placeholder_text(c: Ctx) -> int:
    low = c.body.lower()
    return sum(low.count(p) for p in _PLACEHOLDERS)


def _mermaid_bad(c: Ctx) -> int:
    bad = 0
    for block in _MERMAID_BLOCK.findall(c.body):
        first = next((ln.strip() for ln in block.splitlines() if ln.strip()), "")
        if not first or not first.lower().startswith(_MERMAID_KEYWORDS):
            bad += 1
    return bad


def _table_column_mismatch(c: Ctx) -> int:
    """Count body rows whose column count diverges from their table's header."""
    mismatches = 0
    header_cols: int | None = None
    for line in c.body.splitlines():
        s = line.strip()
        if not (s.startswith("|") or "|" in s and s.count("|") >= 2):
            header_cols = None
            continue
        if not s.startswith("|"):
            header_cols = None
            continue
        cols = s.strip("|").count("|") + 1
        if header_cols is None:
            header_cols = cols
            continue
        if set(s) <= set("|-: "):  # separator row (|---|---|)
            continue
        if cols != header_cols:
            mismatches += 1
    return mismatches


# ---------------------------------------------------------------------------
# Style checks — informational on the vault; counted in the differential only
# when a transform introduces them.
# ---------------------------------------------------------------------------

_HEADING_EMPHASIS = re.compile(r"\*\*.+?\*\*|(?<!\*)\*[^*\s].*?\*|_[^_\s].*?_")


def _heading_emphasis(c: Ctx) -> int:
    return sum(1 for h in c.headings if _HEADING_EMPHASIS.search(h["text"]))


def _heading_level_jump(c: Ctx) -> int:
    jumps = 0
    prev = 0
    for h in c.headings:
        lvl = h["level"]
        if prev and lvl > prev + 1:
            jumps += 1
        prev = lvl
    return jumps


def _duplicate_h1(c: Ctx) -> int:
    return max(0, sum(1 for h in c.headings if h["level"] == 1) - 1)


def _h1_title_mismatch(c: Ctx) -> int:
    if c.stem is None:
        return 0
    h1 = next((h["text"] for h in c.headings if h["level"] == 1), None)
    if h1 is None:
        return 0
    return int(title_key(h1) != title_key(c.stem))


def _malformed_callout(c: Ctx) -> int:
    return sum(1 for t in c.callouts if t.lower() not in CALLOUT_TYPES)


def _unclosed_emphasis(c: Ctx) -> int:
    # odd ** per paragraph (route to raw body — markdown_it strips the markers)
    return sum(1 for para in c.body.split("\n\n") if para.count("**") % 2)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

CHECKS: list[tuple[str, str, Callable[[Ctx], int]]] = [
    # --- structural (16) ---
    ("unclosed-code-fence", "structural", _unclosed_code_fence),
    ("unbalanced-display-math", "structural", _unbalanced_display_math),
    ("unbalanced-inline-math", "structural", _unbalanced_inline_math),
    ("empty-math-block", "structural", _empty_math_block),
    ("double-escaped-latex", "structural", _double_escaped_latex),
    ("frontmatter-yaml-error", "structural", _frontmatter_yaml_error),
    ("unclosed-frontmatter", "structural", _unclosed_frontmatter),
    ("duplicate-frontmatter", "structural", _duplicate_frontmatter),
    ("malformed-wikilink", "structural", _malformed_wikilink),
    ("bad-chars", "structural", _bad_chars),
    ("literal-backslash-n", "structural", _literal_backslash_n),
    ("markdown-fence-wrapper", "structural", _markdown_fence_wrapper),
    ("unclosed-html-comment", "structural", _unclosed_html_comment),
    ("placeholder-text", "structural", _placeholder_text),
    ("mermaid-bad", "structural", _mermaid_bad),
    ("table-column-mismatch", "structural", _table_column_mismatch),
    # --- style (6) ---
    ("heading-emphasis", "style", _heading_emphasis),
    ("heading-level-jump", "style", _heading_level_jump),
    ("duplicate-h1", "style", _duplicate_h1),
    ("h1-title-mismatch", "style", _h1_title_mismatch),
    ("malformed-callout", "style", _malformed_callout),
    ("unclosed-emphasis", "style", _unclosed_emphasis),
]

SEVERITY: dict[str, str] = {name: sev for name, sev, _fn in CHECKS}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan(text: str, stem: str | None = None) -> Counter:
    """Return a Counter of {check_name: count} for one note (zero counts dropped)."""
    ctx = _build_ctx(text, stem)
    out: Counter = Counter()
    for name, _sev, fn in CHECKS:
        n = fn(ctx)
        if n:
            out[name] = n
    return out


def new_violations(before: str, after: str, stem: str | None = None) -> dict:
    """Violations INTRODUCED by a transform: positives of scan(after) − scan(before).

    Counter subtraction keeps only increases, so stable pre-existing violations
    (the human's own) cancel — the gate fires only on what the pipeline added.
    """
    return dict(scan(after, stem) - scan(before, stem))


def totals(counts: Counter | dict) -> tuple[int, int]:
    """Split a scan() result into (structural_total, style_total)."""
    structural = sum(v for k, v in counts.items() if SEVERITY.get(k) == "structural")
    style = sum(v for k, v in counts.items() if SEVERITY.get(k) == "style")
    return structural, style
