"""cli_backend.read_note must return the resolved vault-relative ref.path.

Driver contract: fs and ws backends both populate NoteContent.ref.path;
the CLI backend left it '', which made silica_related query the relatedness
indexes with an empty key and return an empty list.
"""
from __future__ import annotations

from unittest.mock import patch

from silica.driver.cli_backend import ObsidianCLIBackend
from silica.driver.base import NoteRef


def _backend() -> ObsidianCLIBackend:
    backend = ObsidianCLIBackend.__new__(ObsidianCLIBackend)
    backend._vault_name = ""
    backend._is_graph_built = True
    backend._notes = {}
    backend._notes_by_name = {}
    return backend


def test_read_note_by_name_populates_path():
    backend = _backend()

    def fake_cli(*args, **kwargs):
        if args[0] == "read":
            return "# Finanza personale\ncontenuto"
        if args[0] == "files":
            return "Economia e Finanza/Gestione denaro/Finanza personale.md\nAltra/Nota.md"
        raise AssertionError(f"unexpected CLI call: {args}")

    with patch.object(ObsidianCLIBackend, "_run_cli", side_effect=fake_cli):
        nc = backend.read_note("Finanza personale")

    assert nc.ref.path == "Economia e Finanza/Gestione denaro/Finanza personale.md"
    assert nc.content.startswith("# Finanza personale")


def test_read_note_by_ref_keeps_existing_path():
    backend = _backend()
    ref = NoteRef(name="Nota", path="Altra/Nota.md")

    def fake_cli(*args, **kwargs):
        assert args[0] == "read"  # no list_files roundtrip when path is known
        return "body"

    with patch.object(ObsidianCLIBackend, "_run_cli", side_effect=fake_cli):
        nc = backend.read_note(ref)

    assert nc.ref.path == "Altra/Nota.md"
