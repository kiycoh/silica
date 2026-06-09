import json
import re

# Matches [[any/path/to/Note.md]] or [[Note.md]] (with optional #anchor and |alias)
_MD_EXT_WIKILINK_RE = re.compile(
    r'\[\[([^\]#|]+?)\.md((?:#[^\]#|]*)?)(\|[^\]]*)?\]\]',
    re.IGNORECASE,
)

# Characters illegal in filesystem filenames
_ILLEGAL_FILENAME_CHARS_RE = re.compile(r'[/\\:*?"<>|]')


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
                        if i % 2 == 0:  # prose part
                            parts[i] = parts[i].replace("\\n", "\n")
                    val = "```".join(parts)
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


def parse_json(raw: str, strict: bool = False):
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

    return parsed, was_strict_clean
