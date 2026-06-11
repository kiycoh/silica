"""DomainOverlay seam — domain-specific vocabulary plugged in without per-domain code paths.

ADR-0005: domain knowledge lives in overlays, not in kernel conditionals.
The default overlay is English-generic. Per-domain overlays are YAML files
that extend or replace it; the active overlay for a vault lives at
  <vault>/_silica/overlay.yaml

See also: examples/overlays/it-academic.yaml for an Italian-academic overlay.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from stop_words import get_stop_words


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DomainOverlay:
    """Immutable vocabulary overlay applied at concept-extraction time."""

    stopwords: frozenset[str]
    noise_patterns: tuple[re.Pattern, ...]


# ---------------------------------------------------------------------------
# Default (English-generic) overlay
# ---------------------------------------------------------------------------

_ENGLISH_FUNCTION_WORDS: frozenset[str] = frozenset(get_stop_words('en'))

_ENGLISH_STRUCTURAL_TERMS: frozenset[str] = frozenset({
    # headings
    "chapter", "lesson", "exercise", "summary", "topics", "references",
    "objectives", "prerequisites", "syllabus", "calendar", "course",
    # materials
    "notes", "slides", "slide", "presentation", "page", "pages",
    # people / institutions
    "professor", "lecturer", "university",
    # academic logistics
    "exam", "contents", "books", "year", "study", "questions", "answers",
    "part", "text", "online", "website", "link",
    # generic concept-name terms with no discriminating power
    # (English equivalents of domain-neutral Italian structural terms)
    "system", "systems", "model", "models",
    "method", "methods", "approach", "approaches",
    "technique", "techniques", "type", "types",
    "structure", "structures", "concept", "concepts",
    "introduction", "definition", "analysis", "description",
    "principle", "principles", "foundation", "foundations",
})

_DEFAULT_STOPWORDS: frozenset[str] = _ENGLISH_FUNCTION_WORDS | _ENGLISH_STRUCTURAL_TERMS

_DEFAULT_NOISE_PATTERN_STRINGS: tuple[str, ...] = (
    # Numeric list markers: "1. ", "2) ", "3- "
    r'^\s*\d+[\.\)\-]\s+',
    # Academic year ranges: "2023-2024", "2023–2024"
    r'^\s*\d{4}[\-–]\d{4}',
    # Trailing colon (section headers with no title)
    r':\s*$',
    # Question suffix (meta-questions, not content)
    r'\?\s*$',
    # Comparison noise: "X vs Y"
    r'\s+vs\.?\s+',
    # Uppercase-letter prefix (NB:, TODO:, WARN:, ...)
    r'^[A-Z]{2,6}:\s',
    # Lone "q " prefix (quiz/question shorthand)
    r'^q\s',
    # Continued-slide markers
    r'\((continued)\)\s*$',
    # English structural headings
    r'^(Chapter|Lesson|Exercise)\b[:\s]',
    r'^(Summary|Topics|References|Objectives|Prerequisites|Calendar)\s*$',
    # "What's / What is"
    r"^What('?s| is)\b",
)

DEFAULT_OVERLAY: DomainOverlay = DomainOverlay(
    stopwords=_DEFAULT_STOPWORDS,
    noise_patterns=tuple(
        re.compile(p, re.IGNORECASE) for p in _DEFAULT_NOISE_PATTERN_STRINGS
    ),
)


# ---------------------------------------------------------------------------
# load_overlay
# ---------------------------------------------------------------------------

def load_overlay(path: Path) -> DomainOverlay:
    """Parse a YAML overlay file and return a DomainOverlay.

    YAML schema::

        extends_default: true   # optional; default true
        stopwords: [ ... ]      # list of lowercase strings
        noise_patterns: [ ... ] # list of regex strings (compiled IGNORECASE)

    When ``extends_default`` is true (or absent) the result is the union of the
    default overlay and the file's entries. When false the file fully replaces
    the default.

    Raises ValueError (naming the file path) on malformed YAML or invalid regex.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"overlay: failed to parse YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"overlay: expected a YAML mapping in {path}, got {type(raw).__name__}"
        )

    for key in ("stopwords", "noise_patterns"):
        value = raw.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            raise ValueError(
                f"overlay: '{key}' in {path} must be a list, got {type(value).__name__}"
            )
        for i, item in enumerate(value):
            if not isinstance(item, str):
                raise ValueError(
                    f"overlay: '{key}[{i}]' in {path} must be a string, got {type(item).__name__}"
                )

    extends = raw.get("extends_default", True)
    file_stopwords: list[str] = [w.lower() for w in (raw.get("stopwords") or [])]
    file_patterns_raw: list[str] = list(raw.get("noise_patterns") or [])

    compiled: list[re.Pattern] = []
    for pat_str in file_patterns_raw:
        try:
            compiled.append(re.compile(pat_str, re.IGNORECASE))
        except re.error as exc:
            raise ValueError(
                f"overlay: invalid regex {pat_str!r} in {path}: {exc}"
            ) from exc

    if extends:
        merged_stopwords = DEFAULT_OVERLAY.stopwords | frozenset(file_stopwords)
        merged_patterns = DEFAULT_OVERLAY.noise_patterns + tuple(compiled)
    else:
        merged_stopwords = frozenset(file_stopwords)
        merged_patterns = tuple(compiled)

    return DomainOverlay(stopwords=merged_stopwords, noise_patterns=merged_patterns)


# ---------------------------------------------------------------------------
# get_active_overlay — vault-scoped, cached
# ---------------------------------------------------------------------------

_OVERLAY_REL = "_silica/overlay.yaml"
_cached_overlay: DomainOverlay | None = None


def reset_overlay_cache() -> None:
    """Invalidate the module-level overlay cache. Use in tests and after vault switch."""
    global _cached_overlay
    _cached_overlay = None


def get_active_overlay() -> DomainOverlay:
    """Return the active overlay for the current vault, loading and caching it once.

    Resolution order:
      1. ``<vault>/_silica/overlay.yaml`` (if vault_path is set and file exists)
      2. DEFAULT_OVERLAY

    Call ``reset_overlay_cache()`` to force a reload (e.g. after config change or in tests).
    """
    global _cached_overlay
    if _cached_overlay is not None:
        return _cached_overlay

    from silica.config import CONFIG
    vault = getattr(CONFIG, "vault_path", "") or ""
    overlay_path = Path(vault) / _OVERLAY_REL if vault else None

    if overlay_path and overlay_path.exists():
        _cached_overlay = load_overlay(overlay_path)
    else:
        _cached_overlay = DEFAULT_OVERLAY

    return _cached_overlay
