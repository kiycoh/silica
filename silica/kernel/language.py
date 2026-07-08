# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Centralized language resolution — single source of truth.

Multiple kernel modules (cooccurrence, overlay, keyphrase, cohesion,
prep_delegation) each grew their own copy of language-name mapping,
stopword loading and function-word detection, reaching into each other's
private module state to do it. This module reifies that logic into one
leaf: it imports NO other silica module, does zero LLM work, is fully
offline and deterministic, and never raises — every function degrades to
a usable value on failure.
"""
from __future__ import annotations

import re

try:
    from stop_words import StopWordError, get_stop_words
except ImportError:  # pragma: no cover - stop_words is a declared dependency
    get_stop_words = None  # type: ignore[assignment]
    StopWordError = Exception  # type: ignore[assignment,misc]


# Snowball-style language names ("italian") -> ISO codes ("it"). This module
# is the SOLE home of this mapping (overlay.py's former private copy was
# deleted; every other module resolves through here). Deliberately NOT a
# verbatim ISO-639-1 table: "norwegian" maps to "nb" (Bokmål), not the
# ambiguous macrolanguage code "no" — a root-fix for the stop_words package
# only shipping a Bokmål stopword list under "nb".
SNOWBALL_TO_ISO: dict[str, str] = {
    "arabic": "ar", "danish": "da", "dutch": "nl", "english": "en",
    "finnish": "fi", "french": "fr", "german": "de", "hungarian": "hu",
    "italian": "it", "norwegian": "nb", "portuguese": "pt", "romanian": "ro",
    "russian": "ru", "spanish": "es", "swedish": "sv",
}

# Bundled hand-rolled fallback stopwords, used only when the stop_words
# package is unavailable or raises StopWordError. Verbatim copy of the data
# at silica/kernel/cooccurrence.py's `_STOPWORDS`.
_FALLBACK_STOPWORDS: dict[str, frozenset[str]] = {
    "english": frozenset({
        "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on",
        "at", "by", "for", "with", "as", "is", "are", "was", "were", "be",
        "been", "being", "it", "its", "this", "that", "these", "those", "he",
        "she", "they", "we", "you", "his", "her", "their", "our", "your",
        "from", "into", "than", "then", "so", "not", "no", "do", "does", "did",
        "has", "have", "had", "can", "could", "would", "should", "will", "shall",
        "may", "might", "must", "about", "which", "who", "whom", "what", "when",
        "where", "how", "why", "all", "any", "some", "such", "more", "most",
    }),
    "italian": frozenset({
        "di", "da", "in", "con", "su", "per", "tra", "fra", "a", "e", "o", "ma",
        "se", "anche", "come", "il", "lo", "la", "i", "gli", "le", "un", "uno",
        "una", "del", "dello", "della", "dei", "degli", "delle", "al", "allo",
        "alla", "ai", "agli", "alle", "dal", "dalla", "nel", "nella", "sul",
        "sulla", "che", "chi", "cui", "non", "ne", "ci", "vi", "si", "ho", "hai",
        "ha", "abbiamo", "hanno", "sono", "sei", "siamo", "siete", "era", "essere",
        "questo", "questa", "questi", "queste", "quello", "quella", "suo", "sua",
        "loro", "nostro", "vostro", "mio", "tuo",
    }),
}

_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ]+")

# Loaded stopword sets, cached per Snowball language name.
_stopwords_cache: dict[str, frozenset[str]] = {}


def stopwords_for(lang: str) -> frozenset[str]:
    """Return the stopword set for `lang`, lazily loaded and cached.

    Loads from the `stop_words` package (get_stop_words(iso)) for any
    Snowball language in SNOWBALL_TO_ISO. Falls back to the bundled
    hand-rolled en/it sets if the package is missing (ImportError at module
    load, surfaced here as `get_stop_words is None`) or raises
    StopWordError. Unknown languages -> empty frozenset (filters nothing).
    Never raises.
    """
    if lang in _stopwords_cache:
        return _stopwords_cache[lang]

    iso = SNOWBALL_TO_ISO.get(lang)
    if iso is None:
        result = frozenset()
    elif get_stop_words is None:
        result = _FALLBACK_STOPWORDS.get(lang, frozenset())
    else:
        try:
            result = frozenset(get_stop_words(iso))
        except StopWordError:
            result = _FALLBACK_STOPWORDS.get(lang, frozenset())

    _stopwords_cache[lang] = result
    return result


def detect(text: str) -> str:
    """Pick the language whose function-word set best matches `text`.

    Function-word-hit argmax over all languages with a non-empty
    stopwords_for() set (every SNOWBALL_TO_ISO key when the stop_words
    package works; degrades to en/it when it is broken — today's
    behavior). Empty text or no hits -> "english", deterministically.
    Candidate order is english-first, then SNOWBALL_TO_ISO insertion order,
    so max() (which keeps the first max on a tie) resolves any tie
    involving english to english, and other ties by that fixed order.

    ponytail: stopword-ratio classifier; swap to langdetect confined here
    if confusables (es/pt/fr/it) misfire on real prose.
    """
    words = [w.lower() for w in _TOKEN_RE.findall(text)]
    if not words:
        return "english"

    ordered = ["english"] + [name for name in SNOWBALL_TO_ISO if name != "english"]
    candidates = [name for name in ordered if stopwords_for(name)]
    if not candidates:
        return "english"

    return max(
        candidates,
        key=lambda name: sum(1 for w in words if w in stopwords_for(name)),
    )


def resolve(lang: str, sample: str) -> str:
    """Resolve the 'auto' sentinel to a concrete language via detect(sample).

    Anything other than 'auto' is returned unchanged (sample ignored).
    """
    return detect(sample) if lang == "auto" else lang


def display_name(lang: str) -> str:
    """Human-readable form for the distiller {LANGUAGE} placeholder.

    e.g. "italian" -> "Italian".
    """
    return lang.capitalize()
