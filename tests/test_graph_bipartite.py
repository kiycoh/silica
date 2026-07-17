# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""build_bipartite_data (F4 concepts view): the note->Concept-set incidence
already on disk, rendered as a bipartite expansion. The unit is the CORRELATE
Concept set (correlate.topk_set, the glossary term); a df ceiling suppresses
hub concepts, mirroring the facade's IDF hub-suppression."""
from __future__ import annotations

from silica.kernel.cooccurrence import CooccurStore


def _store(tmp_path):
    s = CooccurStore(path=tmp_path / "cooccur.json", lang="english")

    def contrib(**stems):
        return {"nodes": {k: {"label": k, "count": v} for k, v in stems.items()},
                "edges": []}

    s.upsert_note("n1.md", contrib(ml=5, ai=3, solo=1))
    s.upsert_note("n2.md", contrib(ml=4, py=2))
    s.upsert_note("n3.md", contrib(ml=1, py=9, ai=2))
    return s


def _notes():
    return [{"id": "n1.md"}, {"id": "n2.md"}, {"id": "n3.md"}]


def test_bipartite_builds_note_concept_incidence(tmp_path):
    from silica.kernel.graph_export import build_bipartite_data

    cnodes, cedges = build_bipartite_data(_notes(), _store(tmp_path))
    # solo has df=1 < min_df; with 3 notes the default cap is 3, so ml (df=3) stays.
    assert {n["id"] for n in cnodes} == {"concept:ml", "concept:py", "concept:ai"}
    assert all(n["type"] == "concept" for n in cnodes)
    by_id = {n["id"]: n for n in cnodes}
    assert by_id["concept:ml"]["df"] == 3
    pairs = {(e["from"], e["to"]) for e in cedges}
    assert ("n1.md", "concept:ml") in pairs
    assert ("n3.md", "concept:py") in pairs
    assert all(e["type"] == "CONCEPT" for e in cedges)
    assert len(pairs) == 3 + 2 + 2  # ml x3, py x2, ai x2


def test_bipartite_df_cap_suppresses_hub_concepts(tmp_path):
    from silica.kernel.graph_export import build_bipartite_data

    cnodes, _ = build_bipartite_data(_notes(), _store(tmp_path), df_cap=2)
    assert {n["id"] for n in cnodes} == {"concept:py", "concept:ai"}


def test_bipartite_skips_ghosts_and_unindexed_notes(tmp_path):
    from silica.kernel.graph_export import build_bipartite_data

    notes = _notes() + [{"id": "ghost.md", "type": "ghost"},
                        {"id": "unindexed.md"}]
    cnodes, cedges = build_bipartite_data(notes, _store(tmp_path), min_df=1)
    assert all(e["from"] not in ("ghost.md", "unindexed.md") for e in cedges)
    # min_df=1 admits the single-note concept.
    assert "concept:solo" in {n["id"] for n in cnodes}
