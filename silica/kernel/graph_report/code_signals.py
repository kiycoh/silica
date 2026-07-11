# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Code-lane report signals: documentation coverage + import-edge autolink
candidates (spec-code-lane §4b, §5). Deterministic, LLM-free — listed in the
'graph structure is deterministic' import-linter contract."""
from __future__ import annotations

import logging

from silica.kernel.graph_report.models import AutolinkCandidate, CodeCoverage

logger = logging.getLogger(__name__)


def _coverage_from(graph, docmap: dict[str, list[str]]) -> CodeCoverage:
    documented = {p for paths in docmap.values() for p in paths if p in graph.files}
    undocumented = sorted(
        (p for p in graph.files if p not in documented),
        key=lambda p: (-graph.fan_in(p), p),
    )
    return CodeCoverage(
        documented=len(documented),
        total=len(graph.files),
        undocumented=[[p, graph.fan_in(p)] for p in undocumented],
    )


def _import_autolinks_from(
    graph, docmap: dict[str, list[str]], wikilinks: set[tuple[str, str]]
) -> list[AutolinkCandidate]:
    """Note A documents f1, note B documents f2, f1→f2 import edge, no
    wikilink A↔B → PROPOSED candidate. Never written automatically."""
    notes_of: dict[str, list[str]] = {}
    for note, paths in docmap.items():
        for p in paths:
            notes_of.setdefault(p, []).append(note)
    out: list[AutolinkCandidate] = []
    seen: set[tuple[str, str]] = set()
    for f1, entry in graph.files.items():
        for f2 in entry.get("imports", []):
            for a in notes_of.get(f1, []):
                for b in notes_of.get(f2, []):
                    if a == b:
                        continue
                    pair = (min(a, b), max(a, b))
                    if pair in seen or pair in wikilinks:
                        continue
                    seen.add(pair)
                    out.append(AutolinkCandidate(
                        source=a, target=b, weight=1.0,
                        shared=[f"{f1} imports {f2}"], provenance="import",
                    ))
    return sorted(out, key=lambda c: (c.source, c.target))


def _compute_code_signals(
    vault: str, wikilinks: set[tuple[str, str]], graph=None
) -> tuple[CodeCoverage | None, list[AutolinkCandidate]]:
    """Entry point for compute.py. Soft-None when the codegraph is disabled
    (vault outside git) — the report simply has no code section."""
    if graph is None:
        from silica.kernel.codegraph import load_codegraph
        graph = load_codegraph(vault)
    if graph is None or not graph.files:
        return None, []
    from silica.kernel.codedocs import documents_of, iter_documenting_notes
    docmap = {note: documents_of(data) for note, data, _ in iter_documenting_notes(vault)}
    return _coverage_from(graph, docmap), _import_autolinks_from(graph, docmap, wikilinks)
