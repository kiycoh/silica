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


def test_verbose_config_and_logging():
    """Setting CONFIG.verbose to True enables debug logging levels and updates setup."""
    import logging
    from silica.config import CONFIG
    from silica.cli import _setup_logging
    
    # Save original state
    orig_verbose = CONFIG.verbose
    
    try:
        # Enable verbose
        _setup_logging(verbose=True)
        assert CONFIG.verbose is True
        
        # Verify logger level gets set appropriately
        assert logging.getLogger("httpx").level == logging.DEBUG
        assert logging.getLogger("litellm").level == logging.DEBUG
        assert logging.getLogger("openai").level == logging.DEBUG
        
        # Reset logging
        _setup_logging(verbose=False)
        assert CONFIG.verbose is False
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("litellm").level == logging.WARNING
        assert logging.getLogger("openai").level == logging.WARNING
        
    finally:
        # Restore original state
        CONFIG.verbose = orig_verbose
        _setup_logging(verbose=orig_verbose)


def test_verbose_fsm_logging(caplog):
    """FSM transitions are logged in debug/verbose mode."""
    import logging
    from silica.config import CONFIG
    from silica.router.orchestrator import InjectorFSM
    
    orig_verbose = CONFIG.verbose
    CONFIG.verbose = True
    
    # Set logger to DEBUG so caplog captures debug logs
    logger = logging.getLogger("silica.router.orchestrator")
    orig_level = logger.level
    logger.setLevel(logging.DEBUG)
    
    try:
        # Create FSM
        fsm = InjectorFSM(inbox_file="nonexistent.md", target_dir="tmp")
        
        # Testing _make_tmp with verbose logging
        import shutil
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        fsm.target_dir = tmp_dir
        
        with caplog.at_level(logging.DEBUG):
            fsm._make_tmp({"test": "data"})
            assert any("Creato file temporaneo di stage" in rec.message for rec in caplog.records)
            
        shutil.rmtree(tmp_dir)
    finally:
        CONFIG.verbose = orig_verbose
        logger.setLevel(orig_level)

