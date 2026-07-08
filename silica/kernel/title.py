# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""kernel/title — THE identity for note titles (C3).

Before this module, five call sites held five divergent normalizations
(slugify — not even case-insensitive; recon.normalize; _names_agree's private
lowercase fold; the driver index .lower(); is_title_match) and the write path
never compared a freshly coined title against the vault — «Machine Learning
(9 CFU)» happily created the fourth umbrella note.

Two functions, stdlib only:
  title_key(t)              — the equivalence key (casefold, punctuation fold,
                              parenthetical/dash suffix strip, stopword drop,
                              plural-fold via the kernel/text stemmer).
  near_titles(t, titles)    — fuzzy neighbours below key-equality, via
                              difflib.SequenceMatcher.

`templates.slugify` stays what it is — a filename sanitizer, not an identity.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

# Fuzzy band default: below key-equality, above unrelated titles. Chosen so
# Descriptor/Description land inside and «ML per la statistica» vs «Machine
# Learning» stays out (fork ⚑ — revisit with the labelled borderline set).
NEAR_BAND = 0.80

_PAREN_SUFFIX = re.compile(r"\s*\([^()]*\)\s*$")
_DASH_SUFFIX = re.compile(r"\s+[—–-]\s+.*$")
_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)


def title_key(t: str, *, lang: str | None = None) -> str:
    """Equivalence key: two titles with the same key name the same note.

    ``lang`` pins the stemmer (plural-fold); ``None`` detects it from the
    title itself — pass the vault language when you have it, detection on a
    2-4 word label is weak.
    """
    from silica.kernel import language
    from silica.kernel.text import tokens

    s = (t or "").strip()
    for rx in (_PAREN_SUFFIX, _DASH_SUFFIX):
        stripped = rx.sub("", s).strip()
        if stripped:  # never strip a title down to nothing
            s = stripped
    s = _PUNCT.sub(" ", s.casefold())
    if not s.strip():
        return ""
    lang = lang or language.detect(s)
    stems = [
        stem
        for sent in tokens(s, lang=lang, stem=True, stopword_lang=lang, min_len=2)
        for (stem, _surface) in sent
    ]
    # Titles made purely of stopwords/short tokens ("Le basi") must not all
    # collapse onto the empty key: fall back to the folded surface.
    return " ".join(stems) if stems else s.strip()


def near_titles(
    t: str,
    titles: dict[str, str] | list[str],
    band: float = NEAR_BAND,
    *,
    lang: str | None = None,
) -> list[tuple[str, float]]:
    """Titles fuzzy-close to `t` — similar under the band but NOT key-equal.

    `titles` maps title -> anything (only keys are compared) or is a plain
    list. Returns [(title, ratio)] sorted by ratio desc. Key-equal entries are
    excluded: that is a different verdict (coercion, not review).
    """
    key = title_key(t, lang=lang)
    out: list[tuple[str, float]] = []
    for other in titles:
        other_key = title_key(other, lang=lang)
        if not other_key or other_key == key:
            continue
        ratio = SequenceMatcher(None, key, other_key).ratio()
        if ratio >= band:
            out.append((other, ratio))
    return sorted(out, key=lambda kv: -kv[1])
