import logging
import pytest
from silica.ui.logging import HumanFriendlyFormatter, FRIENDLY_TEMPLATES

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
    assert "Tool registrato nel sistema: test_tool (implementato dalla classe TestClass)" in formatted

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
    assert "Ripristinato file note.md alla versione 3" in formatted

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
    assert "⚠️" in formatted
    assert "Impossibile indicizzare la nota my_note.md: Permission Denied" in formatted

def test_human_friendly_formatter_mapped_error():
    formatter = HumanFriendlyFormatter()
    record = logging.LogRecord(
        name="silica.router.refiner_fsm",
        level=logging.ERROR,
        pathname="fsm.py",
        lineno=200,
        msg="Rollback failed: %s",
        args=("Connection timed out",),
        exc_info=None
    )
    formatted = formatter.format(record)
    assert "❌" in formatted
    assert "Annullamento modifiche (rollback) fallito: Connection timed out" in formatted

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
