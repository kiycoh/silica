"""Media handling for note text — image stripping.

Image embeds are removed from text before it reaches the embedding model,
the distiller payload, or any LLM context.  Obsidian-flavored embeds
(![[file.jpg]]) and standard Markdown images (![alt](src)) are both removed.

Call strip_images(text) anywhere you want images silently removed.
"""
from __future__ import annotations

import os
import re

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Obsidian-flavored image embed: ![[path/to/file.ext]] or ![[file.ext|300]]
# Matches any file extension that looks like a raster/vector image or media.
_IMAGE_EXTENSIONS = r"(?:jpe?g|png|gif|webp|svg|bmp|tiff?|avif|mp4|mov|avi|mkv|pdf)"

# Plain-tuple form of the same extensions, for endswith() checks when re-attaching.
_IMG_EXT_TUPLE = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp",
    ".tif", ".tiff", ".avif", ".mp4", ".mov", ".avi", ".mkv", ".pdf",
)

_OFM_IMAGE_RE = re.compile(
    rf"!\[\[([^\]]*\.{_IMAGE_EXTENSIONS})(\|[^\]]*)?\]\]",
    re.IGNORECASE,
)

# Standard Markdown image: ![alt text](src) — local or remote
# Also matches empty alt: ![](...) and wikilink-style src paths
_MD_IMAGE_RE = re.compile(
    r"!\[[^\]]*\]\([^)]*\)",
    re.IGNORECASE,
)

# Same as _MD_IMAGE_RE but capturing the src, for re-attaching (not stripping).
_MD_IMAGE_SRC_RE = re.compile(
    r"!\[[^\]]*\]\(([^)]+)\)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def strip_images(text: str) -> str:
    """Remove all image embeds from *text*, returning clean prose.

    Handles:
      - ``![[images/abc123.jpg]]``           Obsidian embed (no alt)
      - ``![[file.png|300]]``                Obsidian embed with size hint
      - ``![](path/to/image.jpeg)``          Standard Markdown (empty alt)
      - ``![alt text](https://…/img.png)``   Standard Markdown (remote)

    Wikilinks without ``!`` prefix (``[[Note]]``) are left untouched.
    Plain text, headings, and code blocks are left untouched.

    Empty lines left by removed embeds are collapsed to at most one blank line
    so the surrounding text reads cleanly.
    """
    # 1. Remove Obsidian-flavor embeds
    text = _OFM_IMAGE_RE.sub("", text)
    # 2. Remove standard Markdown images
    text = _MD_IMAGE_RE.sub("", text)
    # 3. Collapse runs of blank/whitespace-only lines (≥2) into a single blank line
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)
    return text


def _add_embed(out: list[str], raw: str) -> None:
    """Normalize one image src to an Obsidian basename embed and append (deduped).

    Remote URLs are skipped (they'd resolve to a nonexistent local file).
    Normalizing to ``![[basename.ext]]`` matches convert.py and makes the embed
    resolve from any note location in the vault, not just the inbox folder.
    """
    raw = raw.strip()
    if raw.startswith(("http://", "https://", "mailto:")):
        return
    base = os.path.basename(raw.split("|", 1)[0].strip())  # drop OFM |size hint
    if not base.lower().endswith(_IMG_EXT_TUPLE):
        return
    embed = f"![[{base}]]"
    if embed not in out:
        out.append(embed)


def images_for_section(source: str, concept: str) -> list[str]:
    """Image embeds living in *concept*'s section of *source*, normalized.

    The section is the one whose heading contains *concept* — the SAME match
    payload.py uses to build the concept's excerpt, so a note gets exactly the
    images from the section its content was distilled from. No matching heading
    → no images (concepts pulled from windows have no well-defined section).
    Embeds are returned as deduped, in-order ``![[basename.ext]]``.
    """
    from silica.kernel.payload import find_heading, extract_section  # lazy: avoid cycle

    h = find_heading(source, concept)
    if not h:
        return []
    section = extract_section(source, h)
    out: list[str] = []
    for m in _OFM_IMAGE_RE.finditer(section):
        _add_embed(out, m.group(1))
    for m in _MD_IMAGE_SRC_RE.finditer(section):
        _add_embed(out, m.group(1))
    return out


def append_section_images(snippet: str, source: str, concept: str) -> str:
    """Append *concept*'s section images to *snippet*. No-op when there are none."""
    embeds = images_for_section(source, concept)
    if not embeds:
        return snippet
    block = "\n".join(embeds)
    base = snippet.rstrip()
    return f"{base}\n\n{block}" if base else block


