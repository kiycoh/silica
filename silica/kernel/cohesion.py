# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Post-distillation cohesion pass: detect sibling write ops and inject cross-references.

Siblings are write ops whose display names share ≥ 1 discriminating content token
after filtering vocabulary from the active domain overlay (see silica.kernel.overlay)
and a small language-neutral structural set.

The injected names land in each op's `related` list as bare titles — template_spoke
wraps them in [[...]] when the note is written.

Scope: operates on a single chunk's ops list (list of raw dicts). Cross-chunk
sibling linking is out of scope here; the AUTOLINK+BACKLINK phases handle
that via the vault title index after all chunks are written.

No-free-lunch trade-off: token matching may produce false positives when two
unrelated concepts share a generic discriminating word (e.g. "Neural Networks"
and "Bayesian Networks" both contain "networks" — but in context they ARE
siblings). The `related` field is advisory in Obsidian, so over-linking is
preferable to under-linking for note discovery.

CONFIG: language-specific stopwords (function words, structural-academic terms) come
from the DomainOverlay passed explicitly, or — when ``overlay`` is ``None`` — from
per-note language detection (``silica.kernel.language.detect`` +
``silica.kernel.overlay.overlay_for_lang``). English-generic stopwords live in
DEFAULT_OVERLAY; Italian academic stopwords live in silica/overlays/italian.yaml.
"""
from __future__ import annotations

from typing import Any

from silica.kernel import language
from silica.kernel.overlay import DomainOverlay, overlay_for_lang

# Language-neutral tokens that have no discriminating power in any domain.
# Deliberately narrow: only roman numerals used as chapter/section prefixes.
# Language-specific stopwords (function words, structural academic terms) are
# supplied by the active DomainOverlay — see silica.kernel.overlay.
_STRUCTURAL_TOKENS: frozenset[str] = frozenset({
    # Roman numerals used as chapter/section prefixes
    "ii", "iii", "iv", "vi", "vii", "viii", "ix",
})


def _display_name(op: dict[str, Any]) -> str:
    """The note's user-visible name: `title` (if set by distiller) else `heading`."""
    return (op.get("title") or op.get("heading") or "").strip()


def _content_tokens(name: str, overlay: DomainOverlay | None = None) -> frozenset[str]:
    """Lowercase alphabetic tokens ≥ 2 chars, filtered by overlay stopwords and structural tokens.

    CONFIG: language-specific stopwords come from ``overlay`` (or, when ``None``, from
    ``overlay_for_lang(language.detect(name))`` — the name is the only text this function
    has to detect a language from).  Language-neutral roman-numeral structural tokens are
    always filtered via ``_STRUCTURAL_TOKENS`` regardless of overlay.

    Args:
        name:    The display name to tokenise.
        overlay: DomainOverlay to use for stopword filtering.  ``None`` resolves to
                 ``overlay_for_lang(language.detect(name))``.
    """
    from silica.kernel.text import tokens

    lang = language.detect(name)
    if overlay is None:
        overlay = overlay_for_lang(lang)
    stopwords = overlay.stopwords | _STRUCTURAL_TOKENS
    # Word extraction through the kernel/text seam (C1); the overlay/structural
    # filter is cohesion policy and stays here. min_len=2: display names carry
    # meaningful 2-char tokens (acronyms) that body tokenization drops.
    words = (
        s
        for sent in tokens(name, lang=lang, stem=False, min_len=2, stopword_lang=lang)
        for (_t, s) in sent
    )
    return frozenset(w for w in words if w not in stopwords)


def cohesion_pass(ops_raw: list[dict[str, Any]], overlay: DomainOverlay | None = None) -> list[dict[str, Any]]:
    """Enrich sibling write ops' `related` lists with cross-references.

    Two write ops are siblings when their display names share at least one
    discriminating content token. Skips, patches, and overwrites are passed
    through untouched.

    CONFIG: stopword filtering uses ``overlay`` when given explicitly — resolved ONCE
    on entry and threaded into all ``_content_tokens`` calls, exactly as before. When
    ``overlay`` is ``None`` (the distill.py call site's default), resolution is
    PER-OP instead: each write op's body text (``op["snippet"]``, falling back to its
    display name when the snippet is empty) is language-detected and mapped to
    ``overlay_for_lang()``, so a vault with no ``overlay.yaml`` still gets
    language-appropriate stopword filtering per note rather than one vault-global
    (English) overlay for every op.

    Returns a new list (shallow-copied dicts for write ops that gain siblings;
    originals reused for all others).

    Args:
        ops_raw: List of raw operation dicts from the distiller.
        overlay: DomainOverlay to use for stopword filtering. ``None`` resolves
                 per-op via ``overlay_for_lang(language.detect(body_text))``.
    """
    write_indices = [
        i for i, op in enumerate(ops_raw)
        if op.get("op") == "write"
    ]

    if len(write_indices) < 2:
        return ops_raw

    tokens: dict[int, frozenset[str]]
    if overlay is not None:
        # Explicit overlay: resolve once, thread it down — unchanged behavior.
        tokens = {
            i: _content_tokens(_display_name(ops_raw[i]), overlay=overlay)
            for i in write_indices
        }
    else:
        # overlay=None: resolve per-op from each write op's body language.
        tokens = {}
        for i in write_indices:
            op = ops_raw[i]
            name = _display_name(op)
            body_text = op.get("snippet") or name
            per_op_overlay = overlay_for_lang(language.detect(body_text))
            tokens[i] = _content_tokens(name, overlay=per_op_overlay)

    # Pairwise: find which indices have a sibling
    siblings: dict[int, set[int]] = {i: set() for i in write_indices}
    wi = write_indices
    for pos_a, a in enumerate(wi):
        for b in wi[pos_a + 1:]:
            if tokens[a] and tokens[b] and tokens[a] & tokens[b]:
                siblings[a].add(b)
                siblings[b].add(a)

    # Nothing to do if no siblings found
    if not any(siblings.values()):
        return ops_raw

    result = list(ops_raw)  # top-level list is new; non-sibling ops reuse their dict
    for idx, sib_set in siblings.items():
        if not sib_set:
            continue
        op_copy = dict(ops_raw[idx])
        existing: list[str] = list(op_copy.get("related") or [])
        for sib_idx in sorted(sib_set):
            name = _display_name(ops_raw[sib_idx])
            if name and name not in existing:
                existing.append(name)
        op_copy["related"] = existing
        result[idx] = op_copy

    return result
