#!/usr/bin/env python3
"""Decouple-mode planner: H1->Hub, each H2->Spoke write op (raw section as snippet),
plus one overwrite op turning the monolith into a Hub index. Emits bulk_writer ops JSON."""
# --- hermes_common bootstrap (uniform across all hermes skills) ---
import os, sys
_p = os.path.dirname(os.path.abspath(__file__))
while _p != os.path.dirname(_p) and not os.path.isdir(os.path.join(_p, "hermes_common")):
    _p = os.path.dirname(_p)
if _p not in sys.path:
    sys.path.insert(0, _p)
# --- end bootstrap ---

import argparse, datetime, json, os, sys
from hermes_common import frontmatter, ofm, templates

def build_ops(note, parent_folder, hub):
    with open(note, encoding="utf-8") as f:
        content = f.read()
    _, _, body = frontmatter.split(content)

    # C2 guard: abort early if the note lacks enough H2s to decouple
    heads = ofm.parse_headings(body)
    h2 = [h for h in heads if h["level"] == 2]
    if len(h2) < 2:
        sys.exit(f"[SPLIT] abort: need >=2 H2 to decouple, found {len(h2)} in {note}")

    # C1: capture any content (intro/abstract/H1 body) sitting above the first H2
    preamble = body[:h2[0]["pos"]].strip()
    # Strip leading H1 line from preamble so we don't duplicate it in the hub index
    if preamble:
        lines = preamble.splitlines()
        if lines and lines[0].startswith("# "):
            preamble = "\n".join(lines[1:]).strip()

    sections = ofm.sections_by_h2(body)
    ops, titles = [], []
    seen = {}  # C3: track slug usage to disambiguate collisions
    for s in sections:
        slug = templates.slugify(s["title"])
        seen[slug] = seen.get(slug, 0) + 1
        fname = slug if seen[slug] == 1 else f"{slug} ({seen[slug]})"
        titles.append(s["title"])
        spoke_path = os.path.join(parent_folder, f"{fname}.md")
        ops.append({"op": "write", "path": spoke_path, "heading": s["title"],
                    "snippet": s["content"], "hub": hub})
    hub_fm = {
        "related": [],
        "tags": [frontmatter.clean_tag(hub)],
        "last modified": datetime.date.today().strftime("%Y, %m, %d"),
        "AI": True,
    }
    links = "\n".join(f"- [[{t}]]" for t in titles)
    index_body = f"# {hub}\n\n" + (f"{preamble}\n\n" if preamble else "") + links + "\n"
    ops.append({"op": "overwrite", "path": note, "content": frontmatter.dump(hub_fm, index_body)})
    return ops, titles

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", required=True)
    ap.add_argument("--parent-folder", required=True)
    ap.add_argument("--hub", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ops, titles = build_ops(a.note, a.parent_folder, a.hub)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(ops, f, ensure_ascii=False, indent=2)
    print(f"[SPLIT] {len(titles)} spokes + 1 hub index -> {a.out}", file=sys.stderr)
