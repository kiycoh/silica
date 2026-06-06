"""Tests for CLI-backend eval/escaping foundation helpers."""
from silica.driver.cli_backend import _js_str


def test_js_str_escapes_backslash_and_quote():
    # Explicit (non-raw) strings so the escaping is unambiguous to readers.
    assert _js_str("C:\\notes\\a'b.md") == "C:\\\\notes\\\\a\\'b.md"


def test_js_str_escapes_newline():
    assert _js_str("line1\nline2") == r"line1\nline2"


def test_js_str_plain_passthrough():
    assert _js_str("Folder/Note.md") == "Folder/Note.md"


def test_js_str_escapes_carriage_return():
    # Note bodies with CRLF line endings must not produce a bare CR (a JS
    # SyntaxError) when embedded in a single-quoted literal.
    assert _js_str("a\r\nb") == "a\\r\\nb"


from unittest.mock import patch
from silica.driver.cli_backend import ObsidianCLIBackend


def test_eval_strips_arrow_prefix_and_parses_json():
    backend = ObsidianCLIBackend(vault_name="t")
    with patch.object(backend, "_run_cli", return_value='=> {"a": 1}'):
        assert backend._eval("whatever") == {"a": 1}


def test_eval_handles_no_prefix():
    backend = ObsidianCLIBackend(vault_name="t")
    with patch.object(backend, "_run_cli", return_value='[1, 2, 3]'):
        assert backend._eval("whatever") == [1, 2, 3]


def test_eval_returns_default_on_failure():
    backend = ObsidianCLIBackend(vault_name="t")
    with patch.object(backend, "_run_cli", side_effect=RuntimeError("Obsidian not running")):
        assert backend._eval("whatever", default=[]) == []


def test_eval_raises_when_no_default_given():
    import pytest
    backend = ObsidianCLIBackend(vault_name="t")
    with patch.object(backend, "_run_cli", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            backend._eval("whatever")
