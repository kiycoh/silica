"""Smoke test — verify the tool registry and package imports work."""
from silica.tools import TOOLS


def test_tool_registry_loads():
    """Importing atomic tools should register them in the TOOLS dict."""
    import silica.tools.atomic  # noqa: F401
    assert len(TOOLS) > 0, "No tools registered after importing atomic module"


def test_read_note_registered():
    """silica_read_note should be in the registry."""
    import silica.tools.atomic  # noqa: F401
    assert "silica_read_note" in TOOLS


def test_tool_json_schema():
    """Each tool should produce a valid JSON schema."""
    import silica.tools.atomic  # noqa: F401
    for name, t in TOOLS.items():
        schema = t.json_schema()
        assert "function" in schema, f"{name} missing 'function' key"
        assert "name" in schema["function"], f"{name} missing function name"
        assert "parameters" in schema["function"], f"{name} missing parameters"


def test_config_loads():
    """Config singleton should load without errors."""
    from silica.config import CONFIG
    assert CONFIG.model  # should have a default model
    assert CONFIG.backend in ("cli", "fs")


def test_driver_base_types():
    """Domain types should be importable."""
    from silica.driver.base import (
        NoteRef, NoteContent, Hit, Heading, Link, GraphSnapshot, Txn
    )
    ref = NoteRef(name="Test", path="test.md")
    assert ref.name == "Test"
    content = NoteContent(ref=ref, content="hello", size=5)
    assert content.size == 5
