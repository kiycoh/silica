import re

LIMITS = {"max_lines": 60, "max_chars": 6000, "lean_chars": 600}

def metrics(content):
    return {"char_count": len(content), "line_count": len(content.splitlines())}

def is_lean(content):
    return len(content.strip()) < LIMITS["lean_chars"]

def wikilink(name):
    return f"[[{name}]]"

def has_wikilink(content, name):
    return f"[[{name}]]" in content

HEADING_RE = re.compile(r'^(#{1,6})\s+(.*?)\s*$', re.MULTILINE)
FENCE_RE = re.compile(r'^(`{3,}|~{3,}).*?\n.*?^\1\s*$', re.MULTILINE | re.DOTALL)

def parse_headings(body):
    """Parse headings, ignoring any inside fenced code blocks."""
    # Build set of character ranges covered by fences
    fenced = set()
    for m in FENCE_RE.finditer(body):
        fenced.update(range(m.start(), m.end()))
    return [{"level": len(m.group(1)), "text": m.group(2), "pos": m.start()}
            for m in HEADING_RE.finditer(body) if m.start() not in fenced]

def sections_by_h2(body):
    """Split body at H2 boundaries. Each section's content includes nested H3+.
    Returns [{'title': str, 'content': str}]."""
    heads = [h for h in parse_headings(body) if h["level"] == 2]
    out = []
    for i, h in enumerate(heads):
        start = h["pos"]
        end = heads[i + 1]["pos"] if i + 1 < len(heads) else len(body)
        block = body[start:end]
        section_body = block.split("\n", 1)[1] if "\n" in block else ""
        out.append({"title": h["text"], "content": section_body.strip()})
    return out


# ---------------------------------------------------------------------------
# OFM structural linter (calibrated against golden notes)
# ---------------------------------------------------------------------------

from . import frontmatter as _fm

# Obsidian callout types (canonical + aliases), matched case-insensitively
CALLOUT_TYPES = frozenset({
    "note", "abstract", "summary", "tldr", "info", "todo",
    "tip", "hint", "important",
    "success", "check", "done",
    "question", "help", "faq",
    "warning", "caution", "attention",
    "failure", "fail", "missing",
    "danger", "error",
    "bug", "example", "quote", "cite",
})

# Matches the YYYY, MM, DD date prefix (allows optional time suffix)
DATE_PREFIX_RE = re.compile(r'^\s*\d{4},\s*\d{1,2},\s*\d{1,2}')
# Matches callout opening lines (case-insensitive type)
CALLOUT_RE = re.compile(r'^>\s*\[!([A-Za-z]+)\][+-]?', re.MULTILINE)


def _balanced(body):
    """Check for unbalanced OFM structural delimiters (fence-aware)."""
    issues = []
    # Strip fenced code blocks to avoid false positives on $$ / [[ inside code
    naked = FENCE_RE.sub("", body)
    # Strip inline code spans (`...`) — e.g. `if x == y` would false-positive ==
    naked = re.sub(r'`[^`\n]*`', '', naked)
    if naked.count("$$") % 2:
        issues.append("unbalanced $$ block")
    if naked.count("==") % 2:
        issues.append("unbalanced == highlight")
    if naked.count("[[") != naked.count("]]"):
        issues.append("unbalanced [[wikilink]]")
    if body.count("```") % 2:
        issues.append("unclosed code fence")
    return issues


def ofm_lint(content, stem=None):
    """Pure structural lint for a single note.

    Returns {"violations": [...], "flags": [...]}.
    - violations  → hard errors, should block the pipeline (exit code 2).
    - flags       → soft warnings, auditable but do NOT block.

    Calibration source: golden notes (Connessionismo (IA), Sistema Esperto, KRR).
    Design: H1 position/text unconstrained, callout types case-insensitive,
    date prefix tolerates time suffix, connectivity via any of parent/related/body links.
    """
    data, _, body = _fm.split(content)
    V, F = [], []  # violations, flags

    if data is None:
        V.append("missing/invalid frontmatter")
        data = {}

    # --- frontmatter schema (calibrated on golden notes) ---

    # Tags: detect inline-CSV scalar vs empty vs per-item issues
    raw_tags = data.get("tags")
    if isinstance(raw_tags, str) and "," in raw_tags:
        F.append(
            f"tags is inline-CSV scalar; split into a YAML list "
            f"(will be mangled by normalizer): {raw_tags!r}"
        )
    elif not raw_tags:
        F.append("tags empty")
    else:
        F += _fm.lint_tags(data)  # per-item normalization issues

    # AI field: must be explicitly boolean
    if not isinstance(data.get("AI"), bool):
        V.append("frontmatter 'AI' missing or not boolean")

    # last modified: date prefix required, time suffix tolerated
    lm = data.get("last modified")
    if not (lm and DATE_PREFIX_RE.match(str(lm))):
        F.append("'last modified' missing or malformed date prefix")

    # --- connectivity floor (any one of: parent note / related / body wikilinks) ---
    body_links = re.findall(r'\[\[([^\]|#]+)', body)
    if not (data.get("parent note") or data.get("related") or body_links):
        F.append("orphan note: no parent note / related / wikilinks")

    # --- OFM structural integrity ---
    V += _balanced(body)

    # Detect literal '\n' character sequence in non-code body
    naked = FENCE_RE.sub("", body)
    naked = re.sub(r'`[^`\n]*`', '', naked)
    if "\\n" in naked:
        V.append("literal '\\n' character sequence detected in body")

    for t in CALLOUT_RE.findall(body):
        if t.lower() not in CALLOUT_TYPES:
            V.append(f"unknown callout type [!{t}]")

    heads = parse_headings(body)
    if not any(h["level"] == 1 for h in heads):
        F.append("no H1 heading")

    prev = 0
    for h in heads:
        if prev and h["level"] - prev > 1:
            F.append(f"heading level jump H{prev}->H{h['level']} ({h['text']!r})")
        prev = h["level"]

    return {"violations": V, "flags": F}
