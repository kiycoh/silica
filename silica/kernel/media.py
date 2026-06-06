"""Media handling for note text — image stripping and (future) VLM distillation.

Two modes, controlled by CONFIG.image_mode (SILICA_IMAGE_MODE env var):

  strip (default)
    Remove all image embeds from text before it reaches the embedding model,
    the distiller payload, or any LLM context.  Obsidian-flavored embeds
    (![[file.jpg]]) and standard Markdown images (![alt](src)) are both removed.

  vlm (future / stub)
    Each image embed is replaced by a textual description produced by a vision
    language model (CONFIG.vlm_model).  Not yet implemented; falls back to strip.

Call strip_images(text) anywhere you want images silently removed.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Obsidian-flavored image embed: ![[path/to/file.ext]] or ![[file.ext|300]]
# Matches any file extension that looks like a raster/vector image or media.
_IMAGE_EXTENSIONS = r"(?:jpe?g|png|gif|webp|svg|bmp|tiff?|avif|mp4|mov|avi|mkv|pdf)"

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


def preprocess_text(text: str) -> str:
    """Apply all configured media preprocessing to *text*.

    Currently respects CONFIG.image_mode:
      - "strip"  → call strip_images()
      - "vlm"    → stub, falls back to strip_images() until implemented

    This is the single entry point callers should use so future modes
    (e.g. LaTeX stripping, audio transcripts) can be added here centrally.
    """
    from silica.config import CONFIG  # late import to avoid circular at module level

    mode = getattr(CONFIG, "image_mode", "strip")

    if mode == "vlm":
        # Future: call VLM to produce alt-text, replace embed with description.
        # For now, fall back to strip so behaviour is always well-defined.
        return strip_images(text)

    # Default: strip
    return strip_images(text)
