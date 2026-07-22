# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""fsm-hybrid overlay: verbatim session leaves attached to a distilled vault,
each distilled note linked to its source session(s) via provenance. The graph
stays navigable; the agent follows a note's `## Sources` link to the exact
transcript. Idempotent on vault reuse."""

from silica.kernel.provenance import append_record
from evals.locomo.runner import build_hybrid_overlay


def _inst():
    return {"conversation": {
        "session_1": [
            {"speaker": "Elena", "text": "I signed up for the pottery class."},
            {"speaker": "Sam", "text": "Nice, when does it start?"},
        ],
        "session_1_date_time": "1:56 pm on 8 May, 2023",
    }}


def _seed_distilled_note(vault):
    (vault / "memory").mkdir(parents=True, exist_ok=True)
    (vault / "memory" / "Pottery.md").write_text(
        "---\ntitle: Pottery\n---\n\nElena's beginners pottery class.\n",
        encoding="utf-8")
    append_record("session_1.md", "sha", "run1", ["memory/Pottery"],
                  vault_path=str(vault))


def test_leaves_written_and_notes_linked(tmp_path):
    _seed_distilled_note(tmp_path)
    out = build_hybrid_overlay(tmp_path, _inst())

    leaf = tmp_path / "sources" / "session_1.md"
    assert leaf.is_file()
    assert "Elena: I signed up for the pottery class." in leaf.read_text()  # verbatim

    note = (tmp_path / "memory" / "Pottery.md").read_text()
    assert "## Sources" in note and "[[session_1]]" in note  # linked to its source
    assert out == {"leaves": 1, "linked": 1}


def test_idempotent_on_rerun(tmp_path):
    _seed_distilled_note(tmp_path)
    build_hybrid_overlay(tmp_path, _inst())
    out2 = build_hybrid_overlay(tmp_path, _inst())

    note = (tmp_path / "memory" / "Pottery.md").read_text()
    assert note.count("## Sources") == 1  # no double-linking
    assert out2["linked"] == 0
