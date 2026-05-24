#!/usr/bin/env python3
"""Deterministically lowercase+hyphenate YAML tags. --write applies in place;
otherwise emits a bulk_writer overwrite-op JSON."""
# --- hermes_common bootstrap (uniform across all hermes skills) ---
import os, sys
_p = os.path.dirname(os.path.abspath(__file__))
while _p != os.path.dirname(_p) and not os.path.isdir(os.path.join(_p, "hermes_common")):
    _p = os.path.dirname(_p)
if _p not in sys.path:
    sys.path.insert(0, _p)
# --- end bootstrap ---

import argparse, json
from hermes_common import frontmatter


def normalize(path):
    """Normalize YAML tags for a single note. Returns a list of bulk_writer
    overwrite ops (empty list if no changes needed or frontmatter is missing)."""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    data, _, body = frontmatter.split(content)
    if data is None:
        return []
    before = data.get("tags")
    norm = frontmatter.normalize_tags(data)
    if (norm.get("tags") or []) == (before if isinstance(before, list) else ([before] if before else [])):
        return []
    new_content = frontmatter.dump(norm, body)
    return [{"op": "overwrite", "path": path, "content": new_content}]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", required=True)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    ops = normalize(a.note)
    if not ops:
        # Check if it was a missing-frontmatter or a no-change situation
        with open(a.note, encoding="utf-8") as f:
            data, _, _ = frontmatter.split(f.read())
        if data is None:
            print(json.dumps({"path": a.note, "changed": False, "error": "missing/invalid frontmatter"})); raise SystemExit(1)
        print(json.dumps({"path": a.note, "changed": False})); raise SystemExit(0)
    new_content = ops[0]["content"]
    norm_data, _, _ = frontmatter.split(new_content)
    if a.write:
        with open(a.note, "w", encoding="utf-8") as f: f.write(new_content)
        print(json.dumps({"path": a.note, "changed": True, "applied": True, "tags": norm_data.get("tags", [])}))
    else:
        target = a.out or (a.note + ".ops.json")
        with open(target, "w", encoding="utf-8") as f:
            json.dump(ops, f, ensure_ascii=False, indent=2)
        print(json.dumps({"path": a.note, "changed": True, "applied": False, "ops": target, "tags": norm_data.get("tags", [])}))
