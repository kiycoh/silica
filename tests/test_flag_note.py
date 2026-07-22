# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""silica_flag_note — the use-time correction entry point.

An agent or user flags a note found wrong/stale in use; the note gains the
contested frontmatter flag (reused from the capture-time contested layer) via
the light single-note write path, and the flag can be cleared the same way.
"""
from __future__ import annotations

from silica.kernel.contested import contested_refs

NOTE = (
    "---\n"
    "tags:\n  - x\n"
    "AI: true\n"
    "last modified: 2026, 07, 02\n"
    'related:\n  - "[[H]]"\n'
    "---\n\n"
    "# T\n\nThe dosage is 5mg.\n"
)


def test_flag_note_marks_contested(tmp_vault):
    from silica.tools.notes import silica_flag_note

    path = tmp_vault.note("area/T.md", NOTE)
    res = silica_flag_note("area/T.md", reason="dosage is stale")

    assert "error" not in res, res
    refs = contested_refs(tmp_vault.read(path))
    assert len(refs) == 1
    assert "dosage is stale" in refs[0]
    assert refs[0].startswith("flagged:") and "user" in refs[0]
    assert "The dosage is 5mg." in tmp_vault.read(path)  # body untouched


def test_flag_note_clear_removes_contested(tmp_vault):
    from silica.tools.notes import silica_flag_note

    path = tmp_vault.note("area/T.md", NOTE)
    silica_flag_note("area/T.md", reason="stale")
    res = silica_flag_note("area/T.md", clear=True)

    assert "error" not in res, res
    assert contested_refs(tmp_vault.read(path)) == []


def test_flag_note_updates_register(tmp_vault):
    from silica.kernel import contested_register
    from silica.tools.notes import silica_flag_note

    tmp_vault.note("area/T.md", NOTE)
    silica_flag_note("area/T.md", reason="stale")
    assert "area/T.md" in contested_register.entries()

    silica_flag_note("area/T.md", clear=True)
    assert "area/T.md" not in contested_register.entries()


def test_digest_surfaces_and_clears_contested(tmp_path, tmp_vault, monkeypatch):
    import silica.kernel.progress as _mod
    monkeypatch.setattr(_mod, "_RUNS_DIR", tmp_path)
    from silica.kernel.progress import ProgressLedger
    from silica.tools.notes import silica_flag_note

    tmp_vault.note("area/T.md", NOTE)
    silica_flag_note("area/T.md", reason="stale dosage")

    d = ProgressLedger.new(mode="inject", inputs={}).digest()
    assert "contested note: area/T.md" in d
    assert "stale dosage" in d

    silica_flag_note("area/T.md", clear=True)
    assert "contested note:" not in ProgressLedger.new(mode="inject", inputs={}).digest()


def test_digest_self_heals_resolved_contested(tmp_path, tmp_vault, monkeypatch):
    import silica.kernel.progress as _mod
    monkeypatch.setattr(_mod, "_RUNS_DIR", tmp_path)
    from silica.kernel import contested_register
    from silica.kernel.progress import ProgressLedger

    tmp_vault.note("area/T.md", NOTE)          # clean note, never contested
    contested_register.add("area/T.md")         # a stale register entry

    d = ProgressLedger.new(mode="inject", inputs={}).digest()
    assert "contested note:" not in d                        # not surfaced
    assert "area/T.md" not in contested_register.entries()   # self-healed away
