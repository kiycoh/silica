#!/usr/bin/env python3
"""Mechanical Refiner inspector: decides decouple vs reformat + flags FM issues, token-free."""
# --- hermes_common bootstrap (uniform across all hermes skills) ---
import os, sys
_p = os.path.dirname(os.path.abspath(__file__))
while _p != os.path.dirname(_p) and not os.path.isdir(os.path.join(_p, "hermes_common")):
    _p = os.path.dirname(_p)
if _p not in sys.path:
    sys.path.insert(0, _p)
# --- end bootstrap ---

import argparse, json, sys
from hermes_common import frontmatter, ofm

def inspect(path):
    with open(path, encoding="utf-8") as f:
        content = f.read()
    data, _, body = frontmatter.split(content)
    m = ofm.metrics(content)
    headings = ofm.parse_headings(body)
    h2 = [h for h in headings if h["level"] == 2]
    over_limit = m["char_count"] > ofm.LIMITS["max_chars"] or m["line_count"] > ofm.LIMITS["max_lines"]
    return {
        "path": path,
        "char_count": m["char_count"],
        "line_count": m["line_count"],
        "is_empty": len(body.strip()) == 0,
        "is_lean": ofm.is_lean(body),
        "frontmatter_present": data is not None,
        "frontmatter_issues": frontmatter.lint_tags(data) if data is not None else ["missing/invalid frontmatter"],
        "headings": [{"level": h["level"], "text": h["text"]} for h in headings],
        "mode_hint": "decouple" if (over_limit and len(h2) >= 2) else "reformat",
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    rep = inspect(a.note)
    out = json.dumps(rep, ensure_ascii=False, indent=2)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f: f.write(out)
        print(f"[INSPECT] {a.note}: mode={rep['mode_hint']} chars={rep['char_count']} "
              f"issues={len(rep['frontmatter_issues'])}", file=sys.stderr)
    else:
        print(out)
