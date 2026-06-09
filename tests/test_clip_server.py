"""Tests for the Web Clipper daemon (Tier 2, Item 8, ADR-0009).

Design contract:
  - Localhost-only HTTP server with bearer-token auth.
  - RBAC: writes ONLY to Inbox/ directory, never elsewhere.
  - All incoming content is sanitized before writing.
  - Unauthenticated / wrong-token requests → 401.
  - Path traversal or out-of-inbox destination → 400/403.
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from silica.driver.clip_server import ClipHandler, ClipConfig, clip_request


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return ClipConfig(token="test-secret-token", inbox_dir="Inbox")


@pytest.fixture
def mock_driver():
    with patch("silica.driver.clip_server.DRIVER") as m:
        m.create = MagicMock(return_value=MagicMock(name="Inbox/clip.md"))
        yield m


# ---------------------------------------------------------------------------
# clip_request unit tests (pure function, no HTTP)
# ---------------------------------------------------------------------------

class TestClipRequest:
    def test_valid_clip_writes_to_inbox(self, config, mock_driver):
        result = clip_request(
            token="test-secret-token",
            title="My Article",
            content="Some article content.",
            config=config,
        )
        assert result["status"] == "ok"
        mock_driver.create.assert_called_once()
        call_path = mock_driver.create.call_args[0][0]
        assert call_path.startswith("Inbox/")

    def test_wrong_token_rejected(self, config, mock_driver):
        result = clip_request(
            token="wrong-token",
            title="My Article",
            content="Content.",
            config=config,
        )
        assert result["status"] == "unauthorized"
        mock_driver.create.assert_not_called()

    def test_content_is_sanitized(self, config, mock_driver):
        """Degenerate runs must be collapsed before writing."""
        dirty_content = "aaaaa some text bbbbbbbbb"
        result = clip_request(
            token="test-secret-token",
            title="Dirty",
            content=dirty_content,
            config=config,
        )
        assert result["status"] == "ok"
        written_content = mock_driver.create.call_args[0][1]
        assert "aaaaa" not in written_content
        assert "bbbbbbbbb" not in written_content

    def test_path_traversal_in_title_rejected(self, config, mock_driver):
        """A title containing ../ must not escape the inbox."""
        result = clip_request(
            token="test-secret-token",
            title="../../../etc/passwd",
            content="Content.",
            config=config,
        )
        assert result["status"] in ("ok", "error")
        if mock_driver.create.called:
            call_path = mock_driver.create.call_args[0][0]
            assert ".." not in call_path
            assert call_path.startswith("Inbox/")

    def test_empty_title_gets_fallback(self, config, mock_driver):
        result = clip_request(
            token="test-secret-token",
            title="",
            content="Content.",
            config=config,
        )
        assert result["status"] == "ok"
        call_path = mock_driver.create.call_args[0][0]
        assert call_path.startswith("Inbox/")
        # path must have something after Inbox/
        assert len(call_path) > len("Inbox/")

    def test_empty_content_rejected(self, config, mock_driver):
        result = clip_request(
            token="test-secret-token",
            title="Valid Title",
            content="",
            config=config,
        )
        assert result["status"] == "error"
        mock_driver.create.assert_not_called()

    def test_custom_inbox_dir_respected(self, mock_driver):
        custom_config = ClipConfig(token="tok", inbox_dir="Clips/Web")
        result = clip_request(
            token="tok",
            title="Article",
            content="Body.",
            config=custom_config,
        )
        assert result["status"] == "ok"
        call_path = mock_driver.create.call_args[0][0]
        assert call_path.startswith("Clips/Web/")


class TestClipConfig:
    def test_defaults(self):
        cfg = ClipConfig(token="abc")
        assert cfg.inbox_dir == "Inbox"

    def test_custom_inbox(self):
        cfg = ClipConfig(token="abc", inbox_dir="My Clips")
        assert cfg.inbox_dir == "My Clips"
