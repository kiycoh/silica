"""Mechanical concept-recon over note content.

Vocabulary (stopwords + noise patterns) is supplied by the domain overlay seam;
see silica.kernel.overlay for the default English-generic overlay and the
load_overlay / get_active_overlay API.
"""
from __future__ import annotations

import re

from silica.kernel.overlay import DomainOverlay, get_active_overlay

MIN_LEN, MAX_LEN = 3, 50
TITLE_BONUS = 50
TOP_K_HITS = 3

_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n?", re.DOTALL)
LEADING_GARBAGE = re.compile(r'^[\W_]+')


def normalize(s: str) -> str:
    s = LEADING_GARBAGE.sub('', s)
    return re.sub(r'\s+', ' ', s).rstrip()


def is_concept(s: str, overlay: DomainOverlay | None = None) -> bool:
    """Return True if *s* qualifies as a candidate concept under *overlay*.

    If *overlay* is None, the active vault overlay is resolved via
    ``get_active_overlay()`` (CONFIG-dependent, cached at module level).
    Pass an explicit overlay to make the call CONFIG-free.
    """
    if overlay is None:
        overlay = get_active_overlay()
    if s.lower().strip() in overlay.stopwords:
        return False
    if not (MIN_LEN <= len(s) <= MAX_LEN):
        return False
    if not re.search(r'[A-Za-zÀ-ÿ]{3,}', s):
        return False
    return not any(p.search(s) for p in overlay.noise_patterns)


def _strip_frontmatter(content: str) -> str:
    return _FRONTMATTER_RE.sub('', content, count=1)


# Math spans first (so commands *inside* them vanish with the content), then any
# residual \command outside a span. Strips only the transient extraction string —
# the note on disk keeps its LaTeX.
_MATH_SPANS = re.compile(
    r"\$\$.*?\$\$|\$[^$\n]*?\$|\\\[.*?\\\]|\\\(.*?\\\)", re.DOTALL
)
_LATEX_CMD = re.compile(r"\\[a-zA-Z]+\*?")


def _strip_math(text: str) -> str:
    text = _MATH_SPANS.sub(" ", text)
    return _LATEX_CMD.sub(" ", text)


def is_title_match(c: str, stem: str) -> bool:
    c_lower, stem_lower = c.lower(), stem.lower()
    if c_lower == stem_lower: return True
    if c_lower in stem_lower or stem_lower in c_lower: return True
    c_words = set(re.findall(r'\w+', c_lower))
    s_words = set(re.findall(r'\w+', stem_lower))
    if c_words and s_words and (c_words.issubset(s_words) or s_words.issubset(c_words)):
        return True
    return False


def hit_score(body_count: int, in_title: bool) -> int:
    return body_count + (TITLE_BONUS if in_title else 0)


def rank_hits(raw: list, top_k: int = TOP_K_HITS) -> list:
    return sorted(raw, key=lambda h: hit_score(h["count"], h["in_title"]), reverse=True)[:top_k]


def collision_priority(c: dict) -> tuple:
    if c["best_match"] == "title": return (0, -c["total_hits"])
    if c["total_hits"] >= 3: return (1, -c["total_hits"])
    return (2, -c["total_hits"])
