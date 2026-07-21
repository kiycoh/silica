# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Tests for the silica_lexical_refresh bulk-seed tool.

Closes the bootstrap gap: the write-choke-point hook (write.py) only
maintains an existing lexical.json — nothing previously created one. This
tool is the lexical twin of silica_cooccurrence_refresh: seeds the whole-vault
BM25/fuzzy index from note text, with no LM Studio / no network.
"""
from __future__ import annotations

import pytest

from silica.kernel.lexical import LexicalStore, get_lexical_store


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """Isolated fs vault with two seed notes; points the global DRIVER at it
    and redirects the default lexical index to a per-test tmp path (there is
    no autouse conftest fixture for lexical, unlike embed/cooccurrence)."""
    vault_dir = tmp_path / "vault"
    (vault_dir / "Concepts").mkdir(parents=True)
    (vault_dir / "Concepts" / "Neural.md").write_text(
        "---\ntags:\n  - ai\n---\n\n# Neural\n\nneural network architecture\n",
        encoding="utf-8",
    )
    (vault_dir / "Concepts" / "Boats.md").write_text(
        "# Boats\n\nsailing boat harbour\n", encoding="utf-8"
    )
    monkeypatch.setattr("silica.config.CONFIG.backend", "fs")
    monkeypatch.setattr("silica.config.CONFIG.vault_path", str(vault_dir))
    monkeypatch.setattr("silica.driver._driver", None)
    import silica.kernel.lexical as lex_mod
    monkeypatch.setattr(lex_mod, "_index_path", lambda: tmp_path / "lexical_index.json")
    yield vault_dir
    monkeypatch.setattr("silica.driver._driver", None)


def test_tool_is_registered():
    import silica.tools.composed  # noqa: F401 — importing registers @tool fns
    from silica.tools import TOOLS
    assert "silica_lexical_refresh" in TOOLS
    assert TOOLS["silica_lexical_refresh"].cls == "composed"


def test_refresh_indexes_all_vault_notes(vault):
    from silica.tools.composed import silica_lexical_refresh
    res = silica_lexical_refresh(folder="", force=True)
    assert res["indexed"] == 2
    assert res["total_notes"] == 2
    store = get_lexical_store()
    paths = store.paths()
    assert any(p.endswith("Neural") for p in paths)
    assert any(p.endswith("Boats") for p in paths)


def test_refresh_builds_real_queryable_contributions(vault):
    from silica.tools.composed import silica_lexical_refresh
    silica_lexical_refresh(force=True)
    store = get_lexical_store()
    # "sailing boat harbour" -> a rare token that should surface the Boats note.
    ranked = store.rank("harbour", k=5)
    assert ranked
    assert ranked[0][0].endswith("Boats")


def test_refresh_persists_to_disk(vault):
    """The bulk-seed run must actually create lexical.json (the bootstrap gap
    this tool closes): a fresh LexicalStore.load() from disk finds it."""
    from silica.tools.composed import silica_lexical_refresh
    res = silica_lexical_refresh(force=True)
    from pathlib import Path
    idx_path = Path(res["index_path"])
    assert idx_path.is_file()
    reloaded = LexicalStore.load(idx_path)
    assert len(reloaded) == 2
    assert reloaded.rank("harbour", k=5)[0][0].endswith("Boats")


def test_gc_drops_notes_removed_from_vault(vault):
    """A note deleted from the vault must be dropped from the index on the
    next refresh (mirrors the cooccurrence-refresh GC behaviour)."""
    from silica.driver import DRIVER
    from silica.tools.composed import silica_lexical_refresh
    silica_lexical_refresh(folder="", force=True)
    store = get_lexical_store()
    assert any(p.endswith("Boats") for p in store.paths())

    # DRIVER.delete keeps the driver's in-memory index consistent (a raw
    # filesystem unlink would leave the stale in-memory NoteRef behind).
    DRIVER.delete("Concepts/Boats.md")
    res = silica_lexical_refresh(folder="", force=False)
    assert res["total_notes"] == 1
    store = get_lexical_store()
    assert not any(p.endswith("Boats") for p in store.paths())
    assert any(p.endswith("Neural") for p in store.paths())


def test_force_clears_entries_outside_scanned_folder(vault):
    """force=True rebuilds the whole index from empty; a plain incremental
    refresh only GCs stale entries WITHIN the scanned folder, leaving entries
    outside that scope untouched."""
    from silica.tools.composed import silica_lexical_refresh

    # Seed a bogus entry outside the "Concepts" folder this run will scan.
    store = get_lexical_store()
    store.upsert("Other/Ghost", "Ghost", "unrelated content")
    store.save()

    res = silica_lexical_refresh(folder="Concepts", force=False)
    assert res["total_notes"] == 2
    store = get_lexical_store()
    # Out-of-scope GC never touches it.
    assert "Other/Ghost" in store.paths()

    res = silica_lexical_refresh(folder="Concepts", force=True)
    assert res["total_notes"] == 2
    store = get_lexical_store()
    # force wipes the whole store before reseeding.
    assert "Other/Ghost" not in store.paths()
    assert len(store) == 2


def test_cli_lexical_command_routes_to_tool():
    """'/lexical --force' reaches silica_lexical_refresh with force=True."""
    from unittest.mock import patch
    import silica.tools.composed  # noqa: F401 — ensure tool is registered
    from silica.cli import _handle_direct_shortcut
    from silica.tools import TOOLS

    received: dict = {}

    def capture(folder: str = "", force: bool = False):
        received["folder"] = folder
        received["force"] = force
        return {"indexed": 3, "total_notes": 3, "read_errors": 0, "index_path": "/tmp/x"}

    tool = TOOLS["silica_lexical_refresh"]
    orig = tool.fn
    tool.fn = capture
    try:
        with patch("silica.cli.CONSOLE"):
            result = _handle_direct_shortcut("/lexical --force", [])
    finally:
        tool.fn = orig

    assert result is True
    assert received.get("force") is True


def test_refresh_empty_vault_returns_error(tmp_path, monkeypatch):
    vault_dir = tmp_path / "empty_vault"
    vault_dir.mkdir()
    monkeypatch.setattr("silica.config.CONFIG.backend", "fs")
    monkeypatch.setattr("silica.config.CONFIG.vault_path", str(vault_dir))
    monkeypatch.setattr("silica.driver._driver", None)
    import silica.kernel.lexical as lex_mod
    monkeypatch.setattr(lex_mod, "_index_path", lambda: tmp_path / "lexical_index.json")
    from silica.tools.composed import silica_lexical_refresh
    res = silica_lexical_refresh()
    assert "error" in res
