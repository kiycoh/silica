import logging
import os
import pytest
from unittest.mock import MagicMock, patch
from silica.driver.cli_backend import ObsidianCLIBackend
from silica.driver.fs_backend import ObsidianFSBackend
from silica.driver.base import SettleTimeout, NoteRef

def test_cli_create_settle_success():
    backend = ObsidianCLIBackend(vault_name="test_vault")
    call_counts = {"read": 0}

    def mock_read_note(ref):
        call_counts["read"] += 1
        if call_counts["read"] < 3:
            return MagicMock(content="stale")
        return MagicMock(content="new content with [[Target]] link")

    with patch.object(backend, "_run_cli"), \
         patch.object(backend, "read_note", side_effect=mock_read_note), \
         patch.object(backend, "_wait_for_resolved_event"):
        with patch("silica.driver.cli_backend._SETTLE_POLL_INITIAL", 0.001), \
             patch("silica.driver.cli_backend._SETTLE_POLL_CAP", 0.001):
            ref = backend.create("notes/test.md", "new content with [[Target]] link")
            assert ref.name == "test"
            assert ref.path == "notes/test.md"
            assert call_counts["read"] >= 3

def test_cli_create_settle_timeout_content():
    backend = ObsidianCLIBackend(vault_name="test_vault")

    with patch.object(backend, "_run_cli"), \
         patch.object(backend, "read_note", return_value=MagicMock(content="stale")):

        with patch("silica.driver.cli_backend._SETTLE_POLL_INITIAL", 0.001), \
             patch("silica.driver.cli_backend._SETTLE_POLL_CAP", 0.001), \
             patch("silica.driver.cli_backend._SETTLE_TIMEOUT", 0.01):
            with pytest.raises(SettleTimeout) as exc_info:
                backend.create("notes/test.md", "new content")
            assert "overwrite content" in str(exc_info.value)

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


# ---------------------------------------------------------------------------
# _settle — backoff primitive
# ---------------------------------------------------------------------------
from silica.driver.cli_backend import ObsidianCLIBackend as _CLI


def test_settle_backoff_sequence_is_capped_and_exponential():
    delays = []
    backend = _CLI(vault_name="t")
    with patch("silica.driver.cli_backend.time.sleep", side_effect=lambda d: delays.append(d)):
        calls = {"n": 0}
        def pred():
            calls["n"] += 1
            return calls["n"] >= 6
        backend._settle(pred, "unit-test", timeout=10.0)
    # initial 0.05, doubling, capped at 0.8: 5 sleeps before the 6th check succeeds
    assert delays == [0.05, 0.1, 0.2, 0.4, 0.8]


def test_settle_raises_settle_timeout_when_predicate_never_true():
    from silica.driver.base import SettleTimeout
    backend = _CLI(vault_name="t")
    with patch("silica.driver.cli_backend.time.sleep"), \
         patch("silica.driver.cli_backend.time.monotonic", side_effect=[0.0, 0.0, 5.0, 10.0, 999.0]):
        with pytest.raises(SettleTimeout) as exc:
            backend._settle(lambda: False, "widget", timeout=1.0)
    assert "widget" in str(exc.value)


# ---------------------------------------------------------------------------
# _wait_for_resolved_event — event-driven link settle
# ---------------------------------------------------------------------------

def test_wait_for_resolved_event_returns_when_sentinel_appears(tmp_path):
    backend = _CLI(vault_name="t")
    sentinel = tmp_path / "resolved.sentinel"

    # _run_cli is the listener registration (fire-and-forget) — simulate it by
    # creating the sentinel as a side effect, as Obsidian's JS would.
    def fake_eval_register(*args, **kwargs):
        sentinel.write_text("1")
        return ""

    with patch.object(backend, "_run_cli", side_effect=fake_eval_register), \
         patch("silica.driver.cli_backend.tempfile.mktemp", return_value=str(sentinel)), \
         patch("silica.driver.cli_backend.time.sleep"):
        backend._wait_for_resolved_event(NoteRef(name="t", path="t.md"), timeout=2.0)
    assert not sentinel.exists(), "sentinel must be cleaned up after success"


def test_wait_for_resolved_event_nonfatal_on_timeout(tmp_path, caplog):
    import logging
    backend = _CLI(vault_name="t")
    sentinel = tmp_path / "never.sentinel"
    with patch.object(backend, "_run_cli", return_value=""), \
         patch("silica.driver.cli_backend.tempfile.mktemp", return_value=str(sentinel)), \
         patch("silica.driver.cli_backend.time.sleep"), \
         patch("silica.driver.cli_backend.time.monotonic", side_effect=[0.0, 0.0, 5.0, 999.0]), \
         caplog.at_level(logging.WARNING, logger="silica.driver.cli_backend"):
        backend._wait_for_resolved_event(NoteRef(name="t", path="t.md"), timeout=1.0)
    assert any("resolved event" in r.message for r in caplog.records)
