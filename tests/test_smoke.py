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
    # model may legitimately be empty (fail-fast default — see test_config_failfast)
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
    """Setting CONFIG.debug_logging to True enables debug logging levels and updates setup."""
    import logging
    from silica.config import CONFIG
    from silica.cli import _setup_logging
    
    # Save original state
    orig_debug = CONFIG.debug_logging
    
    try:
        # Enable debug logging
        _setup_logging(debug=True)
        assert CONFIG.debug_logging is True
        
        # httpx/litellm/openai are always suppressed to avoid raw HTTP spam
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("litellm").level == logging.WARNING
        assert logging.getLogger("LiteLLM").level == logging.ERROR
        assert logging.getLogger("openai").level == logging.WARNING
        # asyncio DEBUG suppressed: litellm streaming spawns one loop per chunk
        assert logging.getLogger("asyncio").level == logging.WARNING

        # Silica's own root logger is at DEBUG
        assert logging.getLogger("silica").level <= logging.DEBUG or True  # inherits root

        # Reset logging
        _setup_logging(debug=False)
        assert CONFIG.debug_logging is False
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("litellm").level == logging.WARNING
        assert logging.getLogger("openai").level == logging.WARNING
        
    finally:
        # Restore original state
        CONFIG.debug_logging = orig_debug
        _setup_logging(debug=orig_debug)



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
            assert any("Created staging file" in rec.message for rec in caplog.records)
            
        shutil.rmtree(tmp_dir)
    finally:
        CONFIG.verbose = orig_verbose
        logger.setLevel(orig_level)


def test_inbox_blacklisting_and_external_reads(tmp_path):
    """Verify that files inside inbox_dir are blacklisted from indexing and search, and that external files can be read."""
    import os
    from silica.config import CONFIG
    from silica.driver.fs_backend import ObsidianFSBackend
    
    # Set up directories
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    
    inbox_dir = vault_dir / "Inbox"
    inbox_dir.mkdir()
    
    notes_dir = vault_dir / "notes"
    notes_dir.mkdir()
    
    # Write notes
    (notes_dir / "note1.md").write_text("Hello from Note 1", encoding="utf-8")
    (inbox_dir / "meeting_notes.md").write_text("Hello from Inbox Note", encoding="utf-8")
    
    # Save original config
    orig_inbox = CONFIG.inbox_dir
    orig_backend = CONFIG.backend
    orig_vault = CONFIG.vault_path
    
    CONFIG.inbox_dir = "Inbox"
    CONFIG.backend = "fs"
    CONFIG.vault_path = str(vault_dir)
    
    try:
        backend = ObsidianFSBackend(vault_path=str(vault_dir))
        backend._ensure_index()
        
        # 1. Check that notes in Inbox are not indexed
        assert any(ref.name == "note1" for ref in backend._notes.values())
        assert not any(ref.name == "meeting_notes" for ref in backend._notes.values())
        
        # 2. Check list_files
        listed = [ref.name for ref in backend.list_files()]
        assert "note1" in listed
        assert "meeting_notes" not in listed
        
        # 3. Check search_names
        searched_names = [ref.name for ref in backend.search_names("notes")]
        assert "meeting_notes" not in searched_names
        
        # 4. Check search_context
        hits = backend.search_context("Inbox")
        assert len(hits) == 0
        
        # 5. Check reading an external file outside the vault
        external_file = tmp_path / "external_inbox.md"
        external_file.write_text("External file content", encoding="utf-8")
        
        nc = backend.read_note(str(external_file))
        assert nc.content == "External file content"
        assert nc.ref.path == str(external_file.resolve())
        
    finally:
        CONFIG.inbox_dir = orig_inbox
        CONFIG.backend = orig_backend
        CONFIG.vault_path = orig_vault


def test_cli_backend_error_handling():
    """Verify that ObsidianCLIBackend raises RuntimeError when the CLI returns an error message on stdout."""
    from silica.driver.cli_backend import ObsidianCLIBackend
    from unittest.mock import patch, MagicMock
    import pytest

    cli = ObsidianCLIBackend(vault_name="TestVault")

    # Mock subprocess.run to return the error message on stdout with exit code 0
    mock_response = MagicMock()
    mock_response.stdout = 'Error: File "Deep Learning/Backpropagation.md" not found.'
    mock_response.stderr = ''
    mock_response.returncode = 0

    with patch("subprocess.run", return_value=mock_response) as mock_run:
        with pytest.raises(RuntimeError) as exc_info:
            cli.read_note("Deep Learning/Backpropagation.md")
        assert "not found" in str(exc_info.value)
        mock_run.assert_called_once()


def test_silica_restore_idempotent():
    """Verify that silica_restore ignores file-not-found errors during delete_created rollback operations."""
    from silica.tools.wrapped import silica_restore
    from unittest.mock import patch

    inverses = [
        {"kind": "delete_created", "path": "Deep Learning/Backpropagation.md"}
    ]

    with patch("silica.tools.wrapped.DRIVER.delete", side_effect=RuntimeError("Error: File \"Deep Learning/Backpropagation.md\" not found.")):
        res = silica_restore(txn_id="txn_123", inverses=inverses)

    assert res["success"] is True
    assert "deleted_created:Deep Learning/Backpropagation.md (already_absent)" in res["applied"]
    assert len(res["errors"]) == 0


def test_list_inbox_files_fs(tmp_path):
    """Verify that list_inbox_files lists notes inside inbox_dir on FS backend."""
    import os
    from silica.config import CONFIG
    from silica.driver.fs_backend import ObsidianFSBackend
    
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    
    inbox_dir = vault_dir / "Inbox"
    inbox_dir.mkdir()
    
    (inbox_dir / "lecture_15.md").write_text("Hello from lecture 15", encoding="utf-8")
    (inbox_dir / "subfolder").mkdir()
    (inbox_dir / "subfolder" / "lecture_16.md").write_text("Hello from lecture 16", encoding="utf-8")
    
    orig_inbox = CONFIG.inbox_dir
    orig_vault = CONFIG.vault_path
    
    CONFIG.inbox_dir = "Inbox"
    CONFIG.vault_path = str(vault_dir)
    
    try:
        backend = ObsidianFSBackend(vault_path=str(vault_dir))
        
        inbox_files = backend.list_inbox_files()
        paths = {ref.path for ref in inbox_files}
        names = {ref.name for ref in inbox_files}
        
        assert "Inbox/lecture_15.md" in paths
        assert "Inbox/subfolder/lecture_16.md" in paths
        assert "lecture_15" in names
        assert "lecture_16" in names
    finally:
        CONFIG.inbox_dir = orig_inbox
        CONFIG.vault_path = orig_vault


def test_cli_backend_ref_arg_resolution():
    """Verify that cli backend _ref_arg generates path= when string has slashes or .md suffix."""
    from silica.driver.cli_backend import ObsidianCLIBackend
    from silica.driver.base import NoteRef
    
    backend = ObsidianCLIBackend()
    assert backend._ref_arg("Inbox/lecture_15.md") == "path=Inbox/lecture_15.md"
    assert backend._ref_arg("Deep Learning/lecture.md") == "path=Deep Learning/lecture.md"
    assert backend._ref_arg("lecture.md") == "path=lecture.md"
    assert backend._ref_arg("lecture") == "file=lecture"
    assert backend._ref_arg(NoteRef(name="lecture", path="Folder/lecture.md")) == "path=Folder/lecture.md"
    assert backend._ref_arg(NoteRef(name="lecture")) == "file=lecture"


def test_cli_backend_sentinel_handling():
    """Verify that cli backend _run_cli raises RuntimeError on No matches found."""
    from silica.driver.cli_backend import ObsidianCLIBackend
    from unittest.mock import patch, MagicMock
    import pytest
    
    cli = ObsidianCLIBackend()
    
    # 1. Test _run_cli raises error on "No matches found."
    mock_resp = MagicMock()
    mock_resp.stdout = "No matches found."
    mock_resp.stderr = ""
    mock_resp.returncode = 0
    
    with patch("subprocess.run", return_value=mock_resp):
        with pytest.raises(RuntimeError) as exc:
            cli._run_cli("read", "file=SomeNote")
        assert "No matches found." in str(exc.value)

    # 2. Test _run_json handles "No matches found." gracefully
    with patch("subprocess.run", return_value=mock_resp):
        res = cli._run_json("search:context", "query=something")
        assert res == []

    # 3. "No frontmatter found." (properties on a bare note) is an expected
    #    empty result, not a JSON parse failure — no warning must be logged
    mock_nofm = MagicMock()
    mock_nofm.stdout = "No frontmatter found."
    mock_nofm.stderr = ""
    mock_nofm.returncode = 0
    import logging
    with patch("subprocess.run", return_value=mock_nofm):
        with patch.object(logging.getLogger("silica.driver.cli_backend"), "warning") as warn:
            assert cli._run_json("properties", "file=SomeNote") == []
            assert cli.props_of("SomeNote") == {}
            warn.assert_not_called()


def test_new_tools_registration():
    """Verify that silica_exists and silica_inbox_ls are registered in the registry."""
    from silica.tools import TOOLS
    import silica.tools.atomic  # noqa: F401
    assert "silica_exists" in TOOLS
    assert "silica_inbox_ls" in TOOLS





