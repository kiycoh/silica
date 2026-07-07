"""kernel/text — THE seam for «note text → clean prose → tokens/stems» (C1).

Before this module, three pipelines (cooccurrence, keyphrase/recon, cohesion)
and one rogue regex (the MOC writer's private Italian detector) each owned a
divergent copy of «how a note body becomes tokens»: incompatible stripping
policies put ``frac``/``nabla`` nodes in the co-occurrence graph and image
paths in classify's stems. Stripping, stopwords and stemming live behind two
functions; a stripping bug is fixed here once, for every caller.

Leaf module (like language.py): imports no LLM machinery, deterministic,
never raises on odd input.
"""
from __future__ import annotations

import re
from typing import Any

from silica.kernel import frontmatter, language
from silica.kernel.media import strip_images

MIN_TOKEN_LEN = 3

# Math spans first (so commands *inside* them vanish with their content), then
# any residual \command outside a span. Strips only the transient extraction
# string — the note on disk keeps its LaTeX. (Moved here from recon.py.)
MATH_SPANS = re.compile(
    r"\$\$.*?\$\$|\$[^$\n]*?\$|\\\[.*?\\\]|\\\(.*?\\\)", re.DOTALL
)
LATEX_CMD = re.compile(r"\\[a-zA-Z]+\*?")

_FENCE_RE = re.compile(r"^(```|~~~).*?^\1[^\S\n]*$\n?", re.DOTALL | re.MULTILINE)
# Inline code `x` rides the same `fences` flag: prose vaults treat `test.md`,
# identifiers etc. as noise. URLs never carry co-occurrence signal (utm params,
# `git clone` remotes) — stripped unconditionally, like images. `[^\s)>\]]+`
# stops at the closing paren/bracket of a markdown link so `[text](url)` keeps
# its text.
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s)>\]]+", re.I)
# Obsidian/Excalidraw block-reference ids (^9eUffsl8) — drawing/anchor
# metadata, never prose. Length ≥4 avoids clipping any legit `^` use (math
# superscripts live inside $…$ and are already gone via strip_math).
_BLOCK_ID_RE = re.compile(r"\^[A-Za-z0-9]{4,}")

_SENTENCE_SPLIT = re.compile(r"[.!?;\n]+")
_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ]+")

# Cache stemmers per language (snowballstemmer objects are reusable).
_STEMMERS: dict[str, Any] = {}


def _get_stemmer(lang: str) -> Any:
    # 'auto' is a config sentinel resolved at build time; if it ever reaches
    # here (an unbuilt/empty store) Snowball would KeyError — fall back.
    if lang == "auto":
        lang = "english"
    if lang not in _STEMMERS:
        import snowballstemmer
        _STEMMERS[lang] = snowballstemmer.stemmer(lang)
    return _STEMMERS[lang]


def stem_word(word: str, *, lang: str) -> str:
    """Snowball stem of a single (already lowercased) word."""
    return _get_stemmer(lang).stemWord(word)


def strip_math(text: str) -> str:
    """Blank out LaTeX math spans and residual commands (transient only)."""
    return LATEX_CMD.sub(" ", MATH_SPANS.sub(" ", text))


def is_drawing_note(text: str) -> bool:
    """True for Obsidian Excalidraw notes (front-matter `excalidraw-plugin`).

    These hold compressed drawing data + element ids, never prose — concept
    extraction skips them so their SVG/label soup never becomes graph nodes.
    Keyed on the plugin marker, not the folder (drawings scatter across
    `Excalidraw/` subdirs) nor a `#excalidraw` tag (a prose note may mention
    the tool). YAML-broken front-matter falls back to a raw substring check.
    """
    data, raw_fm, _body = frontmatter.split(text)
    if isinstance(data, dict) and "excalidraw-plugin" in data:
        return True
    return raw_fm is not None and "excalidraw-plugin:" in raw_fm


def clean_body(text: str, *, fences: bool) -> str:
    """Note text → clean prose: frontmatter, math and images always stripped.

    ``fences`` is the caller's explicit choice (no default on purpose):
    keyphrase and prose-vault co-occurrence strip code (fenced + inline) so
    identifiers never become nodes; a code vault keeps them — identifiers ARE
    the graph signal of code notes. URLs (incl. schemeless www.) and
    Obsidian/Excalidraw block-ref ids are always stripped.
    """
    if not text:
        return ""
    _data, _fm, body = frontmatter.split(text)
    body = _BLOCK_ID_RE.sub(" ", _URL_RE.sub(" ", strip_math(strip_images(body))))
    if fences:
        body = _INLINE_CODE_RE.sub(" ", _FENCE_RE.sub(" ", body))
    return body


def tokens(
    text: str,
    *,
    lang: str,
    stem: bool = True,
    stopword_lang: str | None = None,
    min_len: int = MIN_TOKEN_LEN,
) -> list[list[tuple[str, str]]]:
    """Sentences of (token, surface) pairs — the one tokenization pipeline.

    Per sentence: word tokens → lowercase → drop stopwords and tokens shorter
    than ``min_len`` → Snowball stem (``stem=False`` keeps the surface as the
    token, for callers that match verbatim).

    ``lang`` is the primary/stemming language (a store freezes exactly one —
    node keys are stems, and a per-note stemmer would split cross-language
    shared terms). ``stopword_lang`` is per-text: ``None`` detects it from
    ``text`` via language.detect; pass an explicit language to pin it (e.g.
    matching a 2-4 word label, where detection is noise).
    """
    stopwords = language.stopwords_for(stopword_lang or language.detect(text))
    stemmer = _get_stemmer(lang) if stem else None
    out: list[list[tuple[str, str]]] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        toks: list[tuple[str, str]] = []
        for raw in _TOKEN_RE.findall(sentence):
            surface = raw.lower()
            if len(surface) < min_len or surface in stopwords:
                continue
            toks.append((stemmer.stemWord(surface) if stemmer else surface, surface))
        if toks:
            out.append(toks)
    return out
