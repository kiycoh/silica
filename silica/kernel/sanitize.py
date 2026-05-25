import json
import re

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
