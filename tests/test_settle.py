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

def test_cli_create_settle_timeout_links():
    backend = ObsidianCLIBackend(vault_name="test_vault")
    
    with patch.object(backend, "_run_cli"), \
         patch.object(backend, "read_note", return_value=MagicMock(content="new content with [[Target]]")), \
         patch.object(backend, "links", return_value=[]):
        
        with patch("silica.driver.cli_backend._SETTLE_POLL_INTERVAL", 0.001), \
             patch("silica.driver.cli_backend._SETTLE_TIMEOUT", 0.01):
            with pytest.raises(SettleTimeout) as exc_info:
                backend.create("notes/test.md", "new content with [[Target]]")
            assert "links indexing" in str(exc_info.value)

def test_fs_create_patches_index(tmp_path):
    """FS create() atomically patches the index — no SettleTimeout possible."""
    backend = ObsidianFSBackend(vault_path=str(tmp_path))
    (tmp_path / "test.md").write_text("", encoding="utf-8")
    backend._rebuild_index()

    ref = backend.create("test.md", "some content with [[Missing]]")
    assert ref.path == "test.md"
    # _patch_index sets _links atomically
    assert "Missing" in backend._links.get("test.md", set())
