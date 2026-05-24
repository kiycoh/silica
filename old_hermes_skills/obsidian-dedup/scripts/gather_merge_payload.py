#!/usr/bin/env python3
"""Bundle all duplicate variants into ONE payload (mirrors distiller_payload) so the
Router does the semantic merge in a single shot instead of N read_file round-trips."""
# --- hermes_common bootstrap (uniform across all hermes skills) ---
import os, sys
_p = os.path.dirname(os.path.abspath(__file__))
while _p != os.path.dirname(_p) and not os.path.isdir(os.path.join(_p, "hermes_common")):
    _p = os.path.dirname(_p)
if _p not in sys.path:
    sys.path.insert(0, _p)
# --- end bootstrap ---

import argparse, json, sys
from hermes_common import frontmatter

def build(dupes):
    groups = []
    for basename, paths in dupes.items():
        variants = []
        for p in paths:
            try:
                with open(p, encoding="utf-8") as f: content = f.read()
            except OSError:
                continue
            data, _, body = frontmatter.split(content)
            variants.append({"path": p, "char_count": len(content), "frontmatter": data, "body": body})
        if not variants:
            continue
        canonical = max(variants, key=lambda v: v["char_count"])["path"]
        groups.append({"basename": basename, "canonical_hint": canonical, "variants": variants})
    return {"schema_version": 1, "groups": groups}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--duplicates", required=True, help="find_duplicates.py JSON output")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.duplicates, encoding="utf-8") as f:
        dupes = json.load(f)
    if isinstance(dupes, dict) and "error" in dupes:
        print(json.dumps(dupes)); sys.exit(1)
    payload = build(dupes)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"[MERGE-PAYLOAD] {len(payload['groups'])} duplicate groups -> {a.out}", file=sys.stderr)
