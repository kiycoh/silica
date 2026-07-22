# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""probe_classify — placement agreement.

Masked signal: the note's folder. The taxonomy is DERIVED from the vault's own
top-level domains via deterministic c-TF-IDF (top-N stems per domain from the
co-occurrence store), then the real ``classify_notes`` (L1 only, LLM arbiter
off) is asked to reproduce each note's parent domain.

Ceiling (accepted): the taxonomy is derived from the same vault it scores —
leakage is negligible at placement grain. Name-only folder themes score far
worse (17.6% vs 33.4%), so the derivation method + top-N go in the config
snapshot: a change to either invalidates baseline comparison.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from silica.kernel.classify import classify_notes
from silica.kernel.taxonomy import FolderRule, Taxonomy

TOP_N = 15
DERIVATION = "ctfidf-top15"
_UNCATEGORIZED = "Uncategorized"


def vault_domains(vault) -> list[str]:
    """Sorted top-level directory names (dot-prefixed skipped)."""
    return sorted(
        d.name for d in vault.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def derive_taxonomy(domains: list[str], store, top_n: int = TOP_N) -> Taxonomy:
    """c-TF-IDF top-N cooccur stems per domain → a FolderRule per domain.

    Stems present in *every* domain carry no discriminative signal and are
    dropped (the ``df < n_domains`` filter, matching the scratchpad experiment).
    """
    dom_stems: dict[str, Counter] = {d: Counter() for d in domains}
    for path in store.paths():
        d = path.split("/")[0]
        if d in dom_stems:
            dom_stems[d].update(store.note_nodes(path))

    df: Counter = Counter()  # in how many domains each stem appears
    for d in domains:
        for stem in dom_stems[d]:
            df[stem] += 1

    n = len(domains)
    rules = []
    for d in domains:
        scored = {
            stem: count * math.log(n / df[stem])
            for stem, count in dom_stems[d].items()
            if df[stem] < n
        }
        top = [stem for stem, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:top_n]]
        rules.append(FolderRule(folder=d, themes=top))
    return Taxonomy(rules=rules, uncategorized=_UNCATEGORIZED)


def domain_paths(vault, domains: list[str]) -> list[str]:
    """Vault-relative ``*.md`` paths under one of the domain folders (sorted)."""
    from evals.golden.runner import iter_notes

    dom = set(domains)
    out = []
    for p in iter_notes(vault):
        rel = p.relative_to(vault).as_posix()
        if rel.split("/")[0] in dom:
            out.append(rel)
    return out


def run(vault, store, *, verbose: bool = False) -> dict:
    domains = vault_domains(vault)
    tax = derive_taxonomy(domains, store, top_n=TOP_N)
    paths = domain_paths(vault, domains)
    if not paths:
        return {"agreement": 0.0, "uncategorized_rate": 0.0, "notes": 0}

    res = classify_notes(
        paths, tax,
        cooccur_store=store,
        llm_arbiter=False,
        props_map={p: {} for p in paths},  # skip DRIVER props lookups
    )
    n = len(res)
    hits = sum(1 for c in res if c.target_folder == c.note_path.split("/")[0])
    uncat = sum(1 for c in res if c.target_folder == _UNCATEGORIZED)

    if verbose:
        per_dom: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        confusion: Counter = Counter()
        for c in res:
            d = c.note_path.split("/")[0]
            per_dom[d][1] += 1
            if c.target_folder == d:
                per_dom[d][0] += 1
            elif c.target_folder != _UNCATEGORIZED:
                confusion[(d, c.target_folder)] += 1
        print(f"\nclassify: {hits}/{n} = {hits / n:.1%} agreement, "
              f"{uncat} ({uncat / n:.1%}) uncategorized")
        for d in domains:
            h, t = per_dom[d]
            if t:
                print(f"  {d:24s} {h:4d}/{t:<4d} = {h / t:.0%}")
        for (src, dst), cnt in confusion.most_common(15):
            print(f"  confuse {src} -> {dst}: {cnt}")

    return {
        "agreement": round(hits / n, 4),
        "uncategorized_rate": round(uncat / n, 4),
        "notes": n,
    }
