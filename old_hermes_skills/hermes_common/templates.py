# --- hermes_common bootstrap (uniform across all hermes skills) ---
import os, sys
_p = os.path.dirname(os.path.abspath(__file__))
while _p != os.path.dirname(_p) and not os.path.isdir(os.path.join(_p, "hermes_common")):
    _p = os.path.dirname(_p)
if _p not in sys.path:
    sys.path.insert(0, _p)
# --- end bootstrap ---

import datetime, re
from hermes_common.frontmatter import clean_tag  # canonical; do not redefine

def slugify(s: str) -> str:
    s = re.sub(r'[\\/:*?"<>|]', '', s)
    return s.strip().replace('  ', ' ')  # keep spaces, Obsidian likes them

def template_spoke(heading: str, snippet: str, hub: str, tags: list[str] = None, related: list[str] = None) -> str:
    today = datetime.date.today().strftime("%Y, %m, %d")
    body = snippet.strip() or "(da espandere)"
    
    # parent note link
    parent_note = f'"[[{hub}]]"'
    
    # related list
    related_items = [f'"[[{hub}]]"']
    if related:
        for r in related:
            r_link = f'"[[{r}]]"'
            if r_link not in related_items:
                related_items.append(r_link)
    
    # tags list
    tag_list = []
    if tags:
        for t in tags:
            ct = clean_tag(t)
            if ct and ct not in tag_list:
                tag_list.append(ct)
    else:
        # default tag derived from hub
        ch = clean_tag(hub)
        if ch:
            tag_list.append(ch)
            
    # Format YAML components
    related_yaml = "\n".join(f"  - {item}" for item in related_items)
    tags_yaml = "\n".join(f"  - {tag}" for tag in tag_list)
    
    frontmatter = f"""---
parent note: {parent_note}
related:
{related_yaml}
tags:
{tags_yaml}
last modified: {today}
AI: true
---"""

    return f"""{frontmatter}

# {heading}

{body}
"""

def patch_snippet(heading: str, snippet: str, source_basename: str, hub: str = None, existing_content: str = None) -> str:
    patch_text = f"""

## Note aggiuntive — {heading} (da {source_basename})

{snippet.strip()}
"""
    if existing_content is not None:
        if hub and f"[[{hub}]]" not in existing_content:
            if existing_content.startswith("---\n"):
                end_idx = existing_content.find("\n---\n", 4)
                if end_idx != -1:
                    if "\nrelated:\n" in existing_content[:end_idx]:
                        parts = existing_content.split("\nrelated:\n", 1)
                        existing_content = parts[0] + f"\nrelated:\n  - \"[[{hub}]]\"\n" + parts[1]
                    else:
                        existing_content = existing_content[:end_idx] + f"\nrelated:\n  - \"[[{hub}]]\"" + existing_content[end_idx:]
            else:
                today = datetime.date.today().strftime("%Y, %m, %d")
                frontmatter = f"""---
parent note: "[[{hub}]]"
related:
  - "[[{hub}]]"
last modified: {today}
AI: true
---
"""
                existing_content = frontmatter + existing_content

        return existing_content.rstrip() + "\n" + patch_text
    
    return patch_text

