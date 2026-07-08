import logging
import pytest
from silica.ui.logging import (
    AnsiHumanFriendlyFormatter,
    HumanFriendlyFormatter,
    FRIENDLY_TEMPLATES,
)

def test_human_friendly_formatter_mapped_debug():
    formatter = HumanFriendlyFormatter()
    record = logging.LogRecord(
        name="silica.tools",
        level=logging.DEBUG,
        pathname="some_file.py",
        lineno=10,
        msg="Registered tool: %s (class=%s)",
        args=("test_tool", "TestClass"),
        exc_info=None
    )
    formatted = formatter.format(record)
    assert "⚙" in formatted
    assert "Tool registered in system: test_tool (implemented by class TestClass)" in formatted

def test_human_friendly_formatter_mapped_info():
    formatter = HumanFriendlyFormatter()
    record = logging.LogRecord(
        name="silica.router.orchestrator",
        level=logging.INFO,
        pathname="orchestrator.py",
        lineno=50,
        msg="Restored %s to version %d",
        args=("note.md", 3),
        exc_info=None
    )
    formatted = formatter.format(record)
    assert "ℹ" in formatted
    assert "Restored file note.md to version 3" in formatted

def test_human_friendly_formatter_mapped_warning():
    formatter = HumanFriendlyFormatter()
    record = logging.LogRecord(
        name="silica.driver.fs_backend",
        level=logging.WARNING,
        pathname="fs.py",
        lineno=100,
        msg="Failed to index %s: %s",
        args=("my_note.md", "Permission Denied"),
        exc_info=None
    )
    formatted = formatter.format(record)
    assert "⚠" in formatted
    assert "⚠️" not in formatted  # single-width glyph, no emoji variation selector
    assert "Failed to index note my_note.md: Permission Denied" in formatted

def test_human_friendly_formatter_mapped_error():
    formatter = HumanFriendlyFormatter()
    record = logging.LogRecord(
        name="silica.router.orchestrator",
        level=logging.ERROR,
        pathname="fsm.py",
        lineno=200,
        msg="Rollback failed: %s",
        args=("Connection timed out",),
        exc_info=None
    )
    formatted = formatter.format(record)
    assert "✗" in formatted
    assert "Annulling changes (rollback) failed: Connection timed out" in formatted

def test_human_friendly_formatter_unmapped_fallback():
    formatter = HumanFriendlyFormatter()
    record = logging.LogRecord(
        name="silica.some_new_module",
        level=logging.DEBUG,
        pathname="new.py",
        lineno=5,
        msg="Some unmapped technical detail %s",
        args=("value123",),
        exc_info=None
    )
    formatted = formatter.format(record)
    assert "⚙" in formatted
    assert "Some unmapped technical detail value123" in formatted

def test_human_friendly_formatter_non_silica_fallback():
    formatter = HumanFriendlyFormatter()
    record = logging.LogRecord(
        name="urllib3.connectionpool",
        level=logging.DEBUG,
        pathname="pool.py",
        lineno=400,
        msg="Starting new HTTPS connection (1): api.openai.com",
        args=(),
        exc_info=None
    )
    formatted = formatter.format(record)
    assert "⚙" in formatted
    assert "Starting new HTTPS connection (1): api.openai.com" in formatted

def test_human_friendly_formatter_no_history_matches_real_call_site():
    # The only call site (cli_backend.restore_version) logs with TWO
    # placeholders: "No history available for %s: %s" — the template key
    # must match it or the mapping is dead code.
    formatter = HumanFriendlyFormatter()
    record = logging.LogRecord(
        name="silica.driver.cli_backend",
        level=logging.WARNING,
        pathname="cli_backend.py",
        lineno=1076,
        msg="No history available for %s: %s",
        args=("note.md", "boom"),
        exc_info=None,
    )
    formatted = formatter.format(record)
    assert "No version history available for note.md (reason: boom)" in formatted


def test_human_friendly_formatter_bad_args_graceful_fallback():
    formatter = HumanFriendlyFormatter()
    # Template expects 2 args but we only pass 1 (which causes format to fail)
    record = logging.LogRecord(
        name="silica.tools",
        level=logging.DEBUG,
        pathname="some_file.py",
        lineno=10,
        msg="Registered tool: %s (class=%s)",
        args=("test_tool",),
        exc_info=None
    )
    formatted = formatter.format(record)
    # It should fallback gracefully to the standard %-formatted message or original message
    assert "⚙" in formatted
    assert "test_tool" in formatted


def _dedup_record():
    # A real worker-thread call site: it must read human-friendly, not raw.
    return logging.LogRecord(
        name="silica.capabilities.dedup",
        level=logging.INFO,
        pathname="dedup.py",
        lineno=1,
        msg="Restored %s to version %d",  # mapped template, to prove reword survives
        args=("note.md", 2),
        exc_info=None,
    )


def test_ansi_formatter_consumes_markup_and_rewords():
    """The worker seam must interpret markup, never emit literal [muted]/[/] tags.

    (The bug being guarded: writing HumanFriendlyFormatter's markup string raw
    to stderr, which prints the literal rich tags.)
    """
    out = AnsiHumanFriendlyFormatter().format(_dedup_record())
    assert "Restored file note.md to version 2" in out
    assert "[muted]" not in out and "[/muted]" not in out and "[/]" not in out


def test_ansi_formatter_emits_ansi_on_a_terminal(monkeypatch):
    """When CONSOLE is a colour terminal, the worker output carries ANSI codes
    (main-thread parity); when piped it stays plain."""
    class _FakeConsole:
        is_terminal = True
        color_system = "truecolor"
        width = 120

    monkeypatch.setattr("silica.ui.logging.CONSOLE", _FakeConsole())
    out = AnsiHumanFriendlyFormatter().format(_dedup_record())
    assert "\x1b[" in out  # ANSI escape present

    class _PipedConsole:
        is_terminal = False
        color_system = None
        width = 80

    monkeypatch.setattr("silica.ui.logging.CONSOLE", _PipedConsole())
    plain = AnsiHumanFriendlyFormatter().format(_dedup_record())
    assert "\x1b[" not in plain
    assert "Restored file note.md to version 2" in plain


def test_no_root_handler_caches_real_stderr(monkeypatch):
    """Every stderr handler on root must resolve sys.stderr at emit time.

    A handler caching the real stream (plain StreamHandler(sys.stderr)) writes raw
    under an active rich.Live and tears the render — stale frames pile up in
    scrollback as duplicated text.
    """
    import io
    import sys
    from silica.cli import _setup_logging
    from silica.config import CONFIG
    from silica.ui.logging import LiveAwareStreamHandler

    orig_debug = CONFIG.debug_logging
    try:
        for debug in (True, False):
            _setup_logging(debug=debug)
            root = logging.getLogger()
            # No plain StreamHandler holding a cached stream on root
            assert not [h for h in root.handlers if type(h) is logging.StreamHandler]
            # Live-aware handlers follow a sys.stderr swap (rich.Live's redirect)
            proxy = io.StringIO()
            monkeypatch.setattr(sys, "stderr", proxy)
            live_aware = [h for h in root.handlers if isinstance(h, LiveAwareStreamHandler)]
            assert live_aware
            assert all(h.stream is proxy for h in live_aware)
            monkeypatch.undo()
    finally:
        _setup_logging(debug=orig_debug)
