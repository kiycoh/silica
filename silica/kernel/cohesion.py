"""Post-distillation cohesion pass: detect sibling write ops and inject cross-references.

Siblings are write ops whose display names share ≥ 1 discriminating content token
after filtering Italian/English stopwords and overly generic academic terms.

The injected names land in each op's `related` list as bare titles — template_spoke
wraps them in [[...]] when the note is written.

Scope: operates on a single chunk's ops list (list of raw dicts). Cross-chunk
sibling linking is out of scope here; the AUTOLINK+BACKLINK phases handle
that via the vault title index after all chunks are written.

No-free-lunch trade-off: token matching may produce false positives when two
unrelated concepts share a generic discriminating word (e.g. "Reti Neurali" and
"Reti Bayesiane" both contain "reti" — but in context they ARE siblings). The
`related` field is advisory in Obsidian, so over-linking is preferable to
under-linking for note discovery.
"""
from __future__ import annotations

import re
from typing import Any

# Words that appear in many concept names but carry no discriminating power.
# Deliberately narrow: domain-specific terms (peas, reti, backpropagation) are
# NOT stopwords — they ARE the signal we want to match on.
_STOPWORDS: frozenset[str] = frozenset({
    # Italian articles / prepositions / conjunctions
    "di", "da", "in", "con", "su", "per", "tra", "fra", "a", "e", "o", "ma",
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "del", "dello", "della", "dei", "degli", "delle",
    "al", "allo", "alla", "agli", "alle",
    "dal", "dallo", "dalla", "dagli", "dalle",
    "nel", "nello", "nella", "nei", "negli", "nelle",
    "che", "cui", "chi", "come", "quando", "dove",
    # Roman numerals used as chapter/section prefixes
    "ii", "iii", "iv", "vi", "vii", "viii", "ix",
    # Structural/generic academic terms that span every topic
    "sistema", "sistemi", "modello", "modelli",
    "metodo", "metodi", "approccio", "approcci",
    "tecnica", "tecniche", "tipo", "tipi",
    "struttura", "strutture", "concetto", "concetti",
    "introduzione", "definizione", "analisi", "descrizione",
    "base", "fondamento", "fondamenti", "principio", "principi",
    # English stopwords (concept names often mix languages in Italian courses)
    "the", "of", "and", "to", "for", "an", "with", "by", "from",
})


def _display_name(op: dict[str, Any]) -> str:
    """The note's user-visible name: `title` (if set by distiller) else `heading`."""
    return (op.get("title") or op.get("heading") or "").strip()


def _content_tokens(name: str) -> frozenset[str]:
    """Lowercase alphabetic tokens ≥ 2 chars, stopwords removed."""
    words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", name.lower())
    return frozenset(w for w in words if w not in _STOPWORDS)


def cohesion_pass(ops_raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enrich sibling write ops' `related` lists with cross-references.

    Two write ops are siblings when their display names share at least one
    discriminating content token. Skips, patches, and overwrites are passed
    through untouched.

    Returns a new list (shallow-copied dicts for write ops that gain siblings;
    originals reused for all others).
    """
    write_indices = [
        i for i, op in enumerate(ops_raw)
        if op.get("op") == "write"
    ]

    if len(write_indices) < 2:
        return ops_raw

    # Token sets keyed by list index
    tokens: dict[int, frozenset[str]] = {
        i: _content_tokens(_display_name(ops_raw[i]))
        for i in write_indices
    }

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
