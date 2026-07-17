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

from stop_words import StopWordError, get_stop_words


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

_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ]+")

# Loaded stopword sets, cached per Snowball language name.
_stopwords_cache: dict[str, frozenset[str]] = {}


def stopwords_for(lang: str) -> frozenset[str]:
    """Return the stopword set for `lang`, lazily loaded and cached.

    Loads from the `stop_words` package (get_stop_words(iso)) for any
    Snowball language in SNOWBALL_TO_ISO. Unknown languages, or a
    StopWordError from the package -> empty frozenset (filters nothing).
    Never raises.
    """
    if lang in _stopwords_cache:
        return _stopwords_cache[lang]

    iso = SNOWBALL_TO_ISO.get(lang)
    if iso is None:
        result = frozenset()
    else:
        try:
            result = frozenset(get_stop_words(iso))
        except StopWordError:
            result = frozenset()

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
