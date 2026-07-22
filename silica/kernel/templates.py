# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Note templates — template_spoke and patch_snippet.

Migrated AS-IS from hermes_common/templates.py, with the bootstrap path
hack removed (no longer needed — this is a proper Python package now).
"""
import datetime
import logging
import os
import re

from silica.kernel import frontmatter
from silica.kernel.frontmatter import clean_tag  # canonical; do not redefine

logger = logging.getLogger(__name__)

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")

# Template-scoped fence split: unlike frontmatter.FM_RE (whose trailing \s*
# swallows every blank line after the closing fence), this stops at the first
# newline so the template author's body spacing passes through unchanged.
_TEMPLATE_FM_RE = re.compile(r"^---[ \t]*\n(.*?)\n---[ \t]*\n", re.DOTALL)


def slugify(s: str) -> str:
    s = re.sub(r'[\\/:*?"<>|]', '', s)
    s = re.sub(r'\s+', ' ', s)  # normalise newlines, tabs, and multiple spaces to a single space
    return s.strip()


def _link_name(name: str) -> str:
    """Bare note name for a wikilink target — strips brackets the distiller may
    already have wrapped around it, so f'[[{name}]]' never becomes '[[[[X]]]]'
    (a quadruple-bracket frontmatter link Obsidian reads as unresolved), and
    un-escapes a markdown-table pipe (``\\|`` → ``|``): the distiller emits
    aliases like ``[[Target\\|Alias]]``, and ``\\|`` is an invalid escape inside
    the double-quoted YAML scalar we wrap related/hub links in — it breaks
    parsing of the WHOLE frontmatter block (real incident: DQL note, run of
    2026-07-22, rejected as 'Missing or invalid frontmatter')."""
    return name.strip().strip("[]").strip().replace("\\|", "|")


def close_unbalanced_fences(text: str) -> str:
    """Append a closing code fence when the ``` count is odd, mirroring the
    lint's ``body.count('```') % 2`` check (ast._balanced). A distilled snippet
    that opens a fence and never closes it fails post-write lint and gets
    deferred — closing it mechanically lands the note instead of losing it.
    ponytail: balances only the top-level fence count; a snippet that nests
    fences pathologically still needs a real fix."""
    if text.count("```") % 2:
        return text.rstrip() + "\n```\n"
    return text


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


# The historical template_spoke layout as a template string. A vault with no
# templates/ dir and no config renders bit-identically to the old code
# (guarded by the golden parity test).
BUILTIN_TEMPLATE = """---
parent note: {{parent}}
related: {{related}}
tags: {{tags}}
last modified: {{date}}
AI: true
---

# {{title}}

{{body}}
"""


def prepare_fields(*, title: str, body: str, hub: str | None = None,
                   tags: list[str] | None = None,
                   related: list[str] | None = None,
                   parent: str | None = None) -> dict:
    """Encode template_spoke's conditional fallbacks once, for every template.

    Substitution in render_note is pure, so ALL conditional behavior lives
    here: parent falls back to the hub, the hub is merged into related
    (deduplicated), tags default to clean_tag(hub) when empty, date is
    today's ISO date. Values come back ready to substitute — wikilinks
    quoted, tags cleaned. Templates never re-implement these rules.

    Also normalizes body: models drift toward emitting frontmatter at the
    top of markdown regardless of instructions, so a leading YAML block is
    stripped with a warning rather than landing inside the rendered note.
    """
    m = frontmatter.FM_RE.match(body)
    if m:
        logger.warning("prepare_fields: stripped leading YAML block from body")
        body = body[m.end():]

    hub_link = _link_name(hub) if hub else ""
    parent_link = _link_name(parent) if parent else hub_link

    related_items: list[str] = []
    if parent_link:
        related_items.append(f'"[[{parent_link}]]"')
    if hub_link and f'"[[{hub_link}]]"' not in related_items:
        related_items.append(f'"[[{hub_link}]]"')
    for r in related or []:
        r_link = f'"[[{_link_name(r)}]]"'
        if r_link not in related_items:
            related_items.append(r_link)

    tag_list: list[str] = []
    for t in tags or []:
        ct = clean_tag(t)
        if ct and ct not in tag_list:
            tag_list.append(ct)
    if not tag_list and hub_link:
        ch = clean_tag(hub_link)
        if ch:
            tag_list.append(ch)

    return {
        "title": title,
        "body": close_unbalanced_fences(body.strip()) or "(da espandere)",
        "tags": tag_list,
        "related": related_items,
        "parent": f'"[[{parent_link}]]"' if parent_link else "",
        "hub": f'"[[{hub_link}]]"' if hub_link else "",
        "date": datetime.date.today().isoformat(),
    }


def render_note(template_source: str, fields: dict) -> str:
    """Logic-free {{placeholder}} substitution over a whole-note skeleton.

    Line-aware over the frontmatter block only: a frontmatter line whose
    placeholder resolves empty is dropped, and a list value expands to a
    YAML block sequence at its key (empty list drops the key line). In the
    body, placeholders substitute in place. Unknown placeholders are
    removed with a warning — they never block the write.
    """
    def _lookup(name: str):
        if name not in fields:
            logger.warning("render_note: unknown placeholder {{%s}} — removed", name)
            return None
        return fields[name]

    def _sub_all(text: str) -> str:
        return _PLACEHOLDER_RE.sub(lambda m: str(_lookup(m.group(1)) or ""), text)

    m = _TEMPLATE_FM_RE.match(template_source)
    if not m:
        return _sub_all(template_source)

    out: list[str] = []
    for line in m.group(1).split("\n"):
        ph = _PLACEHOLDER_RE.search(line)
        if not ph:
            out.append(line)
            continue
        val = _lookup(ph.group(1))
        if isinstance(val, list):
            if not val:
                continue
            out.append(line[:ph.start()].rstrip())
            out.extend(f"  - {item}" for item in val)
        elif val is None or str(val) == "":
            continue
        else:
            out.append(_sub_all(line))
    return "---\n" + "\n".join(out) + "\n---\n" + _sub_all(template_source[m.end():])


def _bad_template_name(name: str) -> bool:
    """True if a template name contains path separators, traversal sequences, or drive letters.

    Rejects: path separators ("/", "\\"), parent-dir traversal (".."), and drive-relative
    names (":") which would re-anchor pathlib operations to escape the templates dir.
    """
    return "/" in name or "\\" in name or ".." in name or ":" in name


class TemplateNotFoundError(ValueError):
    """Explicit template name that does not resolve — fails loudly."""


def _read_template(path) -> str | None:
    """Template source, or None when missing or malformed (a file that opens
    a frontmatter fence and never closes it)."""
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None
    source = source.replace("\r\n", "\n")
    if source.startswith("---") and not _TEMPLATE_FM_RE.match(source):
        return None
    return source


def resolve_template(name: str | None = None) -> str:
    """Resolution order: explicit name > vault.yaml default_template > built-in.

    An explicit name that is missing/malformed raises TemplateNotFoundError
    listing the available templates; a broken vault default degrades to the
    built-in with a warning — ingestion never stops for a broken template.
    """
    from pathlib import Path

    from silica.config import CONFIG
    from silica.kernel.vault_manifest import get_active_manifest

    conv = get_active_manifest().conventions
    tdir = Path((getattr(CONFIG, "vault_path", "") or "").strip()) / conv.templates_dir
    if name:
        if _bad_template_name(name):
            raise TemplateNotFoundError(
                f"invalid template name {name!r} — names must not contain path separators, '..' or ':'")
        source = _read_template(tdir / f"{name}.md")
        if source is None:
            available = sorted(p.stem for p in tdir.glob("*.md")) if tdir.is_dir() else []
            raise TemplateNotFoundError(
                f"template '{name}' not found or malformed in '{tdir}' — "
                f"available: {', '.join(available) or 'none'}")
        return source
    if conv.default_template:
        if _bad_template_name(conv.default_template):
            logger.warning("invalid vault default template name %r — names must not contain path separators, '..' or ':' — using built-in",
                           conv.default_template)
            return BUILTIN_TEMPLATE
        source = _read_template(tdir / f"{conv.default_template}.md")
        if source is not None:
            return source
        logger.warning("vault default template %r missing or malformed — using built-in",
                       conv.default_template)
    return BUILTIN_TEMPLATE


def ensure_hub_link(content: str, hub: str | None) -> str:
    """Guarantee the hub wikilink in a note's frontmatter `related:` list.

    No-op when hub is falsy or any casing/alias form of the link is already
    present. Callers: patch_snippet (fresh append) and the duplicate-block
    branch of _execute_patch — the repair must land even when the snippet
    itself is skipped, or lint fails the op forever."""
    from silica.kernel.ofm import has_wikilink
    if not hub or has_wikilink(content, hub):
        return content
    if content.startswith("---\n"):
        end_idx = content.find("\n---\n", 4)
        if end_idx == -1:
            return content
        if "\nrelated:\n" in content[:end_idx]:
            parts = content.split("\nrelated:\n", 1)
            return parts[0] + f'\nrelated:\n  - "[[{hub}]]"\n' + parts[1]
        return content[:end_idx] + f'\nrelated:\n  - "[[{hub}]]"' + content[end_idx:]
    today = datetime.date.today().isoformat()
    frontmatter = f"""---
parent note: "[[{hub}]]"
related:
  - "[[{hub}]]"
last modified: {today}
AI: true
---
"""
    return frontmatter + content


def patch_snippet(heading: str, snippet: str, source_basename: str, hub: str | None = None, existing_content: str | None = None) -> str:
    patch_text = f"""

## Note aggiuntive — {heading} (da {source_basename})

{close_unbalanced_fences(snippet.strip())}
"""
    if existing_content is not None:
        existing_content = ensure_hub_link(existing_content, hub)
        return existing_content.rstrip() + "\n" + patch_text

    return patch_text


_AI_KEY_RE = re.compile(r"^AI:\s", re.MULTILINE)


def ensure_ai_flag(content: str) -> str:
    """Stamp `AI: true` into an existing frontmatter block that lacks the field.

    patch/overwrite touch user-authored notes that predate the `AI` convention;
    the OFM lint (ofm.py) requires a boolean `AI` on the *whole* note, so a patch
    to a legacy note would be reverted. Marking `AI: true` is honest provenance —
    the agent is now contributing content. String-level (no YAML round-trip) so
    the rest of the user's frontmatter is left byte-for-byte intact.

    No-ops when there is no frontmatter (fresh writes carry it via template_spoke)
    or the `AI` key already exists (never overwrites the user's own value).
    """
    if not content.startswith("---\n"):
        return content
    end_idx = content.find("\n---\n", 4)
    if end_idx == -1:
        return content  # unterminated frontmatter — leave for the lint to flag
    if _AI_KEY_RE.search(content[4:end_idx]):
        return content
    return content[:end_idx] + "\nAI: true" + content[end_idx:]


_LAST_MODIFIED_RE = re.compile(r"^last modified:.*$", re.MULTILINE)
_AGENT_KEY_RE = re.compile(r"^agent:.*$", re.MULTILINE)


def _stamp_agent(content: str) -> str:
    """Set/refresh `agent: "<id>"` in the frontmatter head when SILICA_AGENT_ID
    is set — provenance for a vault written by a fleet of agents.

    Last-writer-wins, exactly like `last modified`: the field names who last
    touched the note; git keeps the full authorship history. Unset env → the
    field is never added and any existing one is left intact, so single-user
    writes are byte-for-byte unchanged. The value is quoted and escaped so a
    stray value can never break or inject YAML.
    """
    agent = os.environ.get("SILICA_AGENT_ID", "").strip()
    if not agent or not content.startswith("---\n"):
        return content
    end = content.find("\n---\n", 4)
    if end == -1:
        return content
    val = agent.splitlines()[0].replace("\\", "\\\\").replace('"', '\\"')
    line = f'agent: "{val}"'
    head = content[4:end]
    if _AGENT_KEY_RE.search(head):
        return "---\n" + _AGENT_KEY_RE.sub(line, head, count=1) + content[end:]
    return content[:end] + "\n" + line + content[end:]


def ensure_system_floor(content: str, prior: str | None = None) -> str:
    """String-level floor under every write: `AI: true` + `last modified`
    always land, whatever the model emitted. No YAML round-trip.

    - content has a frontmatter block: ensure_ai_flag, unchanged.
    - content has none but `prior` (the pre-write note) has one: re-inject
      the prior block verbatim on top of the new body — omission means
      "keep the user's metadata", not "delete it" — then ensure AI: true
      and refresh `last modified` to today.
    - no block anywhere: create the minimal one.
    """
    if content.startswith("---\n"):
        return _stamp_agent(ensure_ai_flag(content))
    today = datetime.date.today().isoformat()
    pm = frontmatter.FM_RE.match(prior) if prior else None
    if pm is None:
        return _stamp_agent(f"---\nAI: true\nlast modified: {today}\n---\n\n{content.lstrip(chr(10))}")
    # Rebuild the prior block with canonical bare fences: FM_RE tolerates CRLF
    # and fence-line whitespace, but the splices below assume exactly
    # "---\n...\n---\n".
    block = "---\n" + pm.group(1) + "\n---\n"
    merged = ensure_ai_flag(block + "\n" + content.lstrip("\n"))
    end_idx = merged.find("\n---\n", 4)
    head, tail = merged[:end_idx], merged[end_idx:]
    if _LAST_MODIFIED_RE.search(head):
        head = _LAST_MODIFIED_RE.sub(f"last modified: {today}", head, count=1)
    else:
        head += f"\nlast modified: {today}"
    return _stamp_agent(head + tail)


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
