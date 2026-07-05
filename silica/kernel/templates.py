"""Note templates — template_spoke and patch_snippet.

Migrated AS-IS from hermes_common/templates.py, with the bootstrap path
hack removed (no longer needed — this is a proper Python package now).
"""
import datetime
import re

from silica.kernel.frontmatter import clean_tag  # canonical; do not redefine


def slugify(s: str) -> str:
    s = re.sub(r'[\\/:*?"<>|]', '', s)
    s = re.sub(r'\s+', ' ', s)  # normalise newlines, tabs, and multiple spaces to a single space
    return s.strip()


def _link_name(name: str) -> str:
    """Bare note name for a wikilink target — strips brackets the distiller may
    already have wrapped around it, so f'[[{name}]]' never becomes '[[[[X]]]]'
    (a quadruple-bracket frontmatter link Obsidian reads as unresolved)."""
    return name.strip().strip("[]").strip()


def template_spoke(heading: str, snippet: str, hub: str, title: str | None = None, tags: list[str] | None = None, related: list[str] | None = None, parent: str | None = None) -> str:
    today = datetime.date.today().isoformat()
    body = snippet.strip() or "(da espandere)"
    h1 = title or heading  # title wins: filename and H1 stay in sync

    hub_link = _link_name(hub)
    # parent note link — specific parent overrides hub when provided
    if parent:
        parent_link = _link_name(parent)
        parent_note = f'"[[{parent_link}]]"'
        related_items = [f'"[[{parent_link}]]"', f'"[[{hub_link}]]"']
    else:
        parent_note = f'"[[{hub_link}]]"'
        related_items = [f'"[[{hub_link}]]"']

    # related list
    if related:
        for r in related:
            r_link = f'"[[{_link_name(r)}]]"'
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

# {h1}

{body}
"""


def patch_snippet(heading: str, snippet: str, source_basename: str, hub: str | None = None, existing_content: str | None = None) -> str:
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
                        existing_content = parts[0] + f'\nrelated:\n  - "[[{hub}]]"\n' + parts[1]
                    else:
                        existing_content = existing_content[:end_idx] + f'\nrelated:\n  - "[[{hub}]]"' + existing_content[end_idx:]
            else:
                today = datetime.date.today().isoformat()
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


def provenance_header(heading: str, source_basename: str) -> str:
    """The exact header line patch_snippet emits for a (heading, source) block.

    Single source of truth so the patch executor can detect an already-injected
    block and stay idempotent on re-injection.
    """
    return f"## Note aggiuntive — {heading} (da {source_basename})"


def block_present(existing_content: str | None, heading: str, source_basename: str) -> bool:
    """True if a provenance block for (heading, source_basename) is already present."""
    if not existing_content:
        return False
    return provenance_header(heading, source_basename) in existing_content
