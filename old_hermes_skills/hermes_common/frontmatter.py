import re
import yaml

FM_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)

def split(content):
    """Return (data_dict_or_None, raw_fm_or_None, body). data is None on YAML error."""
    m = FM_RE.match(content)
    if not m:
        return None, None, content
    raw = m.group(1)
    body = content[m.end():]
    try:
        data = yaml.safe_load(raw) or {}
    except Exception:
        data = None
    return data, raw, body

def clean_tag(t):
    """Canonical tag normalizer (moved from templates.py — single source of truth)."""
    t = re.sub(r'^[\d\.]+\s*', '', str(t))
    t = t.lower()
    t = re.sub(r'[^a-z0-9\s-]', '', t)
    t = re.sub(r'[\s_]+', '-', t)
    return t.strip('-')

def _ensure_tag_list(raw):
    """Coerce raw tags value into a list, splitting CSV scalars."""
    if not raw:
        return []
    if isinstance(raw, str):
        # Detect inline-CSV scalar: "a, b, c" → ["a", "b", "c"]
        if "," in raw:
            return [s.strip() for s in raw.split(",") if s.strip()]
        return [raw]
    return list(raw)

def lint_tags(data):
    issues = []
    tags = _ensure_tag_list((data or {}).get('tags'))
    for t in tags:
        ct = clean_tag(t)
        if ct != str(t):
            issues.append(f"tag '{t}' not normalized -> '{ct}'")
        if not ct:
            issues.append(f"tag '{t}' is empty after normalization")
    return issues

def normalize_tags(data):
    data = dict(data or {})
    tags = _ensure_tag_list(data.get('tags'))
    seen, out = set(), []
    for t in tags:
        ct = clean_tag(t)
        if ct and ct not in seen:
            seen.add(ct); out.append(ct)
    data['tags'] = out
    return data

def dump(data, body):
    """Re-emit a full note: --- yaml --- + blank line + body."""
    fm = yaml.safe_dump(data, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{fm}\n---\n\n{body.lstrip()}"
