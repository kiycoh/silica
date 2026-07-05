"""Precision gate at write — a note body under MIN_WRITE_SNIPPET_CHARS is
deferred, never written as a «(da espandere)» placeholder.

Real fixture: run 5d0a3350 (2026-07-04) — the distiller returned whole chunks
of write ops with snippet="" despite full inbox excerpts in the payload;
validate passed them and execute_write filled the vault with placeholders
(«Matrici diagonali», «Matrice dei Vettori», «Variabile casuale continua»).
"""
from __future__ import annotations

from silica.kernel.validate import MIN_WRITE_SNIPPET_CHARS, validate_operations


def _write_op(heading: str, snippet: str) -> dict:
    return {
        "op": "write",
        "path": f"Corso/{heading}.md",
        "heading": heading,
        "source_basename": "lez.md",
        "snippet": snippet,
    }


def test_empty_snippet_write_is_rejected(tmp_vault):
    validated, rejected = validate_operations(
        [_write_op("Matrici diagonali", "")], [], "Corso",
    )
    assert validated == []
    assert len(rejected) == 1
    assert "too short" in rejected[0].reason


def test_short_snippet_write_is_rejected(tmp_vault):
    validated, rejected = validate_operations(
        [_write_op("Matrici diagonali", "x" * (MIN_WRITE_SNIPPET_CHARS - 1))],
        [], "Corso",
    )
    assert validated == []
    assert len(rejected) == 1
    assert "too short" in rejected[0].reason


def test_sufficient_snippet_write_passes(tmp_vault):
    validated, rejected = validate_operations(
        [_write_op("Matrici diagonali", "x" * MIN_WRITE_SNIPPET_CHARS)],
        [], "Corso",
    )
    assert rejected == []
    assert any(o.heading == "Matrici diagonali" for o in validated)


def test_whitespace_padding_does_not_beat_the_gate(tmp_vault):
    padded = ("x" * 40) + " " * MIN_WRITE_SNIPPET_CHARS
    validated, rejected = validate_operations(
        [_write_op("Matrici diagonali", padded)], [], "Corso",
    )
    assert validated == []
    assert "too short" in rejected[0].reason


def test_near_title_still_wins_over_short_snippet(tmp_vault):
    """The fuzzy-title band routes to dedup review; the length gate must not
    shadow it (a near-duplicate needs the judge, not a blind retry)."""
    tmp_vault.note("Corso/Descriptor.md", "# Descriptor\n\ncorpo")
    validated, rejected = validate_operations(
        [_write_op("Description", "")], [], "Corso",
    )
    assert validated == []
    assert "near_title" in rejected[0].reason


def test_patch_ops_are_not_gated(tmp_vault):
    """Patch appends to an existing note — a short addendum is legitimate."""
    tmp_vault.note("Corso/Norma.md", "# Norma\n\ncorpo")
    op = {
        "op": "patch",
        "path": "Corso/Norma.md",
        "heading": "Norma",
        "source_basename": "lez.md",
        "snippet": "breve aggiunta",
    }
    validated, rejected = validate_operations([op], [], "Corso")
    assert rejected == []
    assert any(o.heading == "Norma" for o in validated)
