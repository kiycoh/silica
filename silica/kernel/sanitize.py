# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

import json
import logging
import re

logger = logging.getLogger(__name__)

# Inline ($...$) must not cross newlines; block ($$...$$) may.
_MATH = re.compile(r"\$\$.*?\$\$|\$[^\n$]+?\$", re.DOTALL)


def replace_outside_math(text: str, old: str, new: str) -> str:
    """`text.replace(old, new)` everywhere EXCEPT inside `$...$` / `$$...$$` spans.

    Lets the distiller post-processor turn double-escaped prose newlines into real
    ones without shredding `\\nabla`/`\\neq` or splitting inline math.
    """
    out: list[str] = []
    last = 0
    for m in _MATH.finditer(text):
        out.append(text[last:m.start()].replace(old, new))
        out.append(m.group(0))  # math span: verbatim
        last = m.end()
    out.append(text[last:].replace(old, new))
    return "".join(out)


# Matches [[any/path/to/Note.md]] or [[Note.md]] (with optional #anchor and |alias)
_MD_EXT_WIKILINK_RE = re.compile(
    r'\[\[([^\]#|]+?)\.md((?:#[^\]#|]*)?)(\|[^\]]*)?\]\]',
    re.IGNORECASE,
)

# Characters illegal in filesystem filenames
_ILLEGAL_FILENAME_CHARS_RE = re.compile(r'[/\\:*?"<>|]')

# Degenerate run: 5+ consecutive identical characters (LLM output garbage).
# Excludes markdown-structural chars (# = - * _ ` ~): they legitimately repeat
# — ATX headings up to ######, thematic breaks / setext underlines, emphasis,
# code fences — and collapsing them corrupts real document structure (the golden
# integrity probe caught `##### Heading` → `# Heading` on nucleate).
# Also excludes digits (\d): they are data, not garbage — "100000" is a number,
# not a degenerate run, and collapsing it silently corrupts the value.
_DEGENERATE_RUN_RE = re.compile(r'([^\n\d#*_=~`-])\1{4,}')

# Nested wikilink brackets: 3+ consecutive '[' or ']'. A valid wikilink uses
# exactly two; 3+ only appears when an already-bracketed name was wrapped again
# (e.g. the distiller emits "[[X]]" and a renderer makes "[[[[X]]]]"). Single and
# double brackets — including code like x[[1]] — are left untouched.
_NESTED_WIKILINK_RE = re.compile(r'\[{3,}|\]{3,}')


def collapse_nested_wikilinks(text: str) -> str:
    """Collapse [[[[X]]]] (and deeper) down to a single [[X]] wikilink."""
    return _NESTED_WIKILINK_RE.sub(lambda m: m.group(0)[:2], text)


def strip_degenerate_runs(text: str) -> str:
    """Collapse runs of 5+ identical characters to a single instance.

    Lines are preserved; only in-line repetitions are collapsed.
    """
    return _DEGENERATE_RUN_RE.sub(r'\1', text)


def _strip_md_ext(text: str) -> str:
    """Remove .md extension from inside wikilinks: [[Note.md]] → [[Note]]."""
    return _MD_EXT_WIKILINK_RE.sub(
        lambda m: f"[[{m.group(1)}{m.group(2)}{m.group(3) or ''}]]",
        text,
    )


def normalize_ops(ops: list) -> list:
    """Post-process a list of op dicts to fix common distiller output errors.

    Applied normalizations:
    1. Strip .md extension from wikilinks in `snippet`, `content`, and `related`.
    2. Strip filesystem-illegal characters from `title` when present.
    """
    if not isinstance(ops, list):
        return ops

    cleaned: list = []
    for op in ops:
        if not isinstance(op, dict):
            cleaned.append(op)
            continue
        op = dict(op)  # shallow copy — don't mutate in place

        for field in ("snippet", "content"):
            if isinstance(op.get(field), str):
                val = op[field]
                val = val.rstrip()
                while val.endswith("\\n"):
                    val = val[:-2].rstrip()
                if "\\n" in val:
                    parts = val.split("```")
                    for i in range(len(parts)):
                        if i % 2 == 0:  # prose part — but never inside math spans
                            parts[i] = replace_outside_math(parts[i], "\\n", "\n")
                    val = "```".join(parts)
                val = strip_degenerate_runs(val)
                val = collapse_nested_wikilinks(val)
                op[field] = _strip_md_ext(val)

        if isinstance(op.get("related"), list):
            op["related"] = [
                _strip_md_ext(r) if isinstance(r, str) else r
                for r in op["related"]
            ]

        if isinstance(op.get("title"), str):
            op["title"] = _ILLEGAL_FILENAME_CHARS_RE.sub("", op["title"]).strip()
            if not op["title"]:
                op["title"] = None

        cleaned.append(op)

    return cleaned


# Bodies carried outside the JSON string, keyed by integer ref. Line-anchored,
# non-prose sentinel so distilled markdown/LaTeX won't collide with it.
# ponytail: collides only if a body literally contains a `===SILICA-BODY N===`
# line — vanishingly rare; upgrade the sentinel if it ever surfaces.
_BODY_MARKER = re.compile(r"^===SILICA-BODY (\d+)===$", re.MULTILINE)


def extract_body_appendix(raw: str) -> tuple[str, dict[int, str]]:
    """Split a `<json>\\n===SILICA-BODY N===\\n<body>...` payload.

    Returns the JSON text (everything before the first marker) and a {ref: body}
    map. Bodies are verbatim — no JSON unescaping ever touches them, so LaTeX
    backslashes survive (`\\top` stays `\\top`, never decodes to a TAB). No
    markers → (raw, {}), i.e. legacy single-blob JSON output is untouched.
    """
    markers = list(_BODY_MARKER.finditer(raw))
    if not markers:
        return raw, {}
    json_text = raw[: markers[0].start()]
    bodies: dict[int, str] = {}
    for i, m in enumerate(markers):
        start = m.end()
        if raw[start : start + 1] == "\n":
            start += 1  # drop the newline ending the marker line
        end = markers[i + 1].start() if i + 1 < len(markers) else len(raw)
        body = raw[start:end]
        if body.endswith("\n"):
            body = body[:-1]  # drop the newline preceding the next marker
        bodies[int(m.group(1))] = body
    return json_text, bodies


def _resolve_op_refs(op, bodies: dict[int, str]) -> None:
    if not isinstance(op, dict):
        return
    for ref_key, field in (("snippet_ref", "snippet"), ("content_ref", "content")):
        if ref_key in op:
            ref = op.pop(ref_key)
            if isinstance(ref, int) and ref in bodies:
                op[field] = bodies[ref]
            else:
                # Dangling ref: the model emitted `snippet_ref: N` but never wrote
                # the matching `===SILICA-BODY N===` block. Leaving `field` empty
                # here silently produces a 0-char snippet that validate later
                # rejects as "too short" with no clue why — surface it instead.
                logger.warning(
                    "sanitize: dangling %s=%r for op %r (available bodies: %s) — "
                    "%s left empty",
                    ref_key, ref, op.get("path") or op.get("heading") or "?",
                    sorted(bodies), field,
                )


def _inject_external_bodies(parsed, bodies: dict[int, str]) -> None:
    """Replace `snippet_ref`/`content_ref` ints with their external body string."""
    if isinstance(parsed, list):
        ops = parsed
    elif isinstance(parsed, dict) and isinstance(parsed.get("updates"), list):
        ops = parsed["updates"]
    elif isinstance(parsed, dict):
        ops = [parsed]
    else:
        return
    for op in ops:
        _resolve_op_refs(op, bodies)


def parse_json(raw: str, strict: bool = False):
    raw, _bodies = extract_body_appendix(raw)
    cleaned = raw.strip()
    if cleaned.startswith('\ufeff'):
        cleaned = cleaned[1:]
    
    fence_pattern = re.compile(r'^```(?:json)?\s*\n(.*?)\n```$', re.DOTALL | re.IGNORECASE)
    inner_fence_pattern = re.compile(r'```(?:json)?\s*\n(.*?)\n```', re.DOTALL | re.IGNORECASE)
    
    was_strict_clean = True
    processed = cleaned
    
    m = fence_pattern.match(cleaned)
    if m:
        processed = m.group(1).strip()
        was_strict_clean = False
    else:
        m = inner_fence_pattern.search(cleaned)
        if m:
            processed = m.group(1).strip()
            was_strict_clean = False
            
    parsed = None
    parse_err = None
    try:
        parsed = json.loads(processed)
    except json.JSONDecodeError as e:
        start_idx = -1
        for idx, ch in enumerate(raw):
            if ch in '{[':
                start_idx = idx
                break
        end_idx = -1
        for idx in range(len(raw) - 1, -1, -1):
            if raw[idx] in '}]':
                end_idx = idx
                break
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = raw[start_idx:end_idx+1]
            try:
                parsed = json.loads(candidate)
                was_strict_clean = False
            except json.JSONDecodeError as inner_e:
                parse_err = inner_e
        else:
            parse_err = e

    if parsed is None:
        if parse_err is not None:
            raise parse_err
        raise ValueError("JSON Parse Error")

    if strict and not was_strict_clean:
        raise ValueError("Strict mode violation: markdown fences, preambles, or postambles were stripped from the output.")

    if _bodies:
        _inject_external_bodies(parsed, _bodies)

    return parsed, was_strict_clean
