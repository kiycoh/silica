import logging
import pytest
from unittest.mock import MagicMock, patch
from silica.driver.cli_backend import ObsidianCLIBackend
from silica.driver.fs_backend import ObsidianFSBackend
from silica.driver.base import SettleTimeout, NoteRef

def test_cli_create_settle_success():
    backend = ObsidianCLIBackend(vault_name="test_vault")

    call_counts = {"read": 0, "links": 0}

    def mock_read_note(ref):
        call_counts["read"] += 1
        if call_counts["read"] < 3:
            return MagicMock(content="stale")
        return MagicMock(content="new content with [[Target]] link")

    def mock_links(ref):
        call_counts["links"] += 1
        if call_counts["links"] < 3:
            return []
        return [NoteRef(name="Target", path="Target.md")]

    with patch.object(backend, "_run_cli") as mock_run_cli, \
         patch.object(backend, "read_note", side_effect=mock_read_note) as mock_read, \
         patch.object(backend, "links", side_effect=mock_links) as mock_links_method:

        with patch("silica.driver.cli_backend._SETTLE_POLL_INTERVAL", 0.001):
            ref = backend.create("notes/test.md", "new content with [[Target]] link")
            assert ref.name == "test"
            assert ref.path == "notes/test.md"
            assert call_counts["read"] >= 3
            assert call_counts["links"] >= 3

def test_cli_create_settle_timeout_content():
    backend = ObsidianCLIBackend(vault_name="test_vault")

    with patch.object(backend, "_run_cli"), \
         patch.object(backend, "read_note", return_value=MagicMock(content="stale")):

        with patch("silica.driver.cli_backend._SETTLE_POLL_INTERVAL", 0.001), \
             patch("silica.driver.cli_backend._SETTLE_TIMEOUT", 0.01):
            with pytest.raises(SettleTimeout) as exc_info:
                backend.create("notes/test.md", "new content")
            assert "overwrite content" in str(exc_info.value)

def test_cli_create_settle_timeout_links_is_nonfatal(caplog):
    """Link-indexing settle timeout must NOT raise — it logs a warning and returns.

    The note is already on disk (_wait_for_content_reflects passed).
    Upstream callers should not treat this as a write failure.
    """
    backend = ObsidianCLIBackend(vault_name="test_vault")

    with patch.object(backend, "_run_cli"), \
         patch.object(backend, "read_note",
                      return_value=MagicMock(content="new content with [[Target]]")), \
         patch.object(backend, "links", return_value=[]):

        with patch("silica.driver.cli_backend._SETTLE_POLL_INTERVAL", 0.001), \
             patch("silica.driver.cli_backend._SETTLE_TIMEOUT", 0.01), \
             caplog.at_level(logging.WARNING, logger="silica.driver.cli_backend"):
            # Must NOT raise SettleTimeout
            ref = backend.create("notes/test.md", "new content with [[Target]]")
            assert ref.name == "test"
            assert ref.path == "notes/test.md"

    assert any("links indexing" in r.message for r in caplog.records), \
        "Expected a warning about links indexing timeout"


# ---------------------------------------------------------------------------
# links() — JSON array output parsing
# ---------------------------------------------------------------------------

def test_links_parses_json_array():
    """CLI returns '["Folder/A.md","B.md"]' — both notes extracted correctly."""
    backend = ObsidianCLIBackend(vault_name="test_vault")
    with patch.object(backend, "_run_cli", return_value='["Folder/A.md", "B.md"]'):
        result = backend.links(NoteRef(name="src", path="src.md"))
    assert len(result) == 2
    assert result[0].name == "A"
    assert result[0].path == "Folder/A.md"
    assert result[1].name == "B"
    assert result[1].path == "B.md"

def test_links_parses_empty_json_array():
    """CLI returns '[]' (no links yet) — must produce empty list, NOT NoteRef(name='[]')."""
    backend = ObsidianCLIBackend(vault_name="test_vault")
    with patch.object(backend, "_run_cli", return_value="[]"):
        result = backend.links(NoteRef(name="src", path="src.md"))
    assert result == [], f"Expected [], got {result}"

def test_links_parses_plain_text():
    """CLI returns plain-text paths — existing behaviour preserved."""
    backend = ObsidianCLIBackend(vault_name="test_vault")
    with patch.object(backend, "_run_cli", return_value="Folder/A.md\nB.md"):
        result = backend.links(NoteRef(name="src", path="src.md"))
    assert len(result) == 2
    assert result[0].name == "A"
    assert result[1].name == "B"

def test_links_filters_no_links_found():
    """'No links found' response produces empty list regardless of format."""
    backend = ObsidianCLIBackend(vault_name="test_vault")
    with patch.object(backend, "_run_cli", return_value="No links found"):
        result = backend.links(NoteRef(name="src", path="src.md"))
    assert result == []


def test_fs_create_patches_index(tmp_path):
    """FS create() atomically patches the index — no SettleTimeout possible."""
    backend = ObsidianFSBackend(vault_path=str(tmp_path))
    (tmp_path / "test.md").write_text("", encoding="utf-8")
    backend._rebuild_index()

    ref = backend.create("test.md", "some content with [[Missing]]")
    assert ref.path == "test.md"
    # _patch_index sets _links atomically
    assert "Missing" in backend._links.get("test.md", set())
