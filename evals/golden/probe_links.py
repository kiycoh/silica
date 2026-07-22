# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""probe_links — masked-link recall.

Masked signal: wikilinks. For each note with >=1 body ``[[link]]`` the links
are stripped to their surface text, the whole-vault title index is rebuilt, and
the real ``autolink`` is asked to recover them.

Only recall gates. Precision is unknowable without negative labels — autolink
finding a link the human didn't make isn't necessarily wrong (humans
under-link), so ``extra_per_note`` is reported but informational.

Ceiling (accepted): a link whose alias differs from its target
(``[[Target|other words]]``) is unrecoverable by design — a constant offset,
irrelevant to regression deltas.
"""
from __future__ import annotations

import re

from silica.kernel import frontmatter
from silica.kernel.autolink import autolink, build_title_index

# (!?) embed marker, target, optional #anchor, optional |alias
_WIKILINK = re.compile(r"(!?)\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]*))?\]\]")


def mask_links(body: str) -> tuple[str, list[str]]:
    """Replace non-embed ``[[...]]`` with their surface text; return (masked, targets).

    Surface text = alias when present, else the raw target. Embeds (``![[...]]``)
    are left untouched (they carry no prose link to recover).
    """
    targets: list[str] = []

    def _sub(m: re.Match) -> str:
        embed, target, alias = m.group(1), m.group(2), m.group(3)
        if embed:
            return m.group(0)  # leave embeds verbatim
        targets.append(target.strip())
        return (alias if alias is not None else target).strip()

    return _WIKILINK.sub(_sub, body), targets


def run(vault, *, verbose: bool = False) -> dict:
    from evals.golden.runner import iter_notes

    all_md = iter_notes(vault)
    title_index = build_title_index([p.stem for p in all_md])  # drops ambiguous basenames
    index_cf = {t.casefold(): t for t in title_index}

    total_wanted = 0
    total_recovered = 0
    total_extra = 0
    notes_evaluated = 0
    misses: list[tuple[str, list[str]]] = []

    for p in all_md:
        _data, _raw, body = frontmatter.split(p.read_text(encoding="utf-8"))
        masked, targets = mask_links(body)
        if not targets:
            continue

        # denominator = links whose basename is a resolvable, unambiguous title
        wanted: dict[str, str] = {}  # casefold -> index spelling
        for t in targets:
            base = t.replace("\\", "/").split("/")[-1]
            hit = index_cf.get(base.casefold())
            if hit is not None:
                wanted[hit.casefold()] = hit
        if not wanted:
            continue

        stem = p.stem
        # substring prefilter — semantics-preserving (a title absent as a
        # substring can never whole-word-match), cuts runtime to seconds.
        low = masked.casefold()
        cands = [t for t in title_index if t.casefold() in low]
        _new_body, added = autolink(masked, title_index, candidates=cands, self_title=stem)
        added_cf = {a.casefold() for a in added}

        recovered = wanted.keys() & added_cf
        notes_evaluated += 1
        total_wanted += len(wanted)
        total_recovered += len(recovered)
        total_extra += len(added_cf - wanted.keys())
        if verbose and len(recovered) < len(wanted):
            misses.append((p.relative_to(vault).as_posix(),
                           [wanted[k] for k in wanted.keys() - added_cf]))

    recall = round(total_recovered / total_wanted, 4) if total_wanted else 0.0
    extra_per_note = round(total_extra / notes_evaluated, 4) if notes_evaluated else 0.0

    if verbose:
        print(f"\nlinks: recall {total_recovered}/{total_wanted} = {recall:.1%}, "
              f"{extra_per_note:.2f} extra/note over {notes_evaluated} notes")
        for path, missed in misses[:25]:
            print(f"  miss {path}: {missed}")

    return {
        "recall": recall,
        "extra_per_note": extra_per_note,
        "links_evaluated": total_wanted,
        "notes_evaluated": notes_evaluated,
    }
