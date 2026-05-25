import tempfile
from pathlib import Path
from silica.router.target_resolver import resolve_ingestion_target
from silica.router.orchestrator import InjectorFSM
from silica.config import CONFIG

def test_resolve_ingestion_target():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Test 1: Empty target
        mode, resolved = resolve_ingestion_target("", tmpdir)
        assert mode == "ATOMIC_FOLDER_MODE"
        assert resolved == ""

        # Test 2: Ends with .md
        mode, resolved = resolve_ingestion_target("Concepts.md", tmpdir)
        assert mode == "FILE_APPEND_MODE"
        assert resolved == "Concepts.md"

        # Test 3: No extension, but .md file exists
        (tmp_path / "MOC.md").touch()
        mode, resolved = resolve_ingestion_target("MOC", tmpdir)
        assert mode == "FILE_APPEND_MODE"
        assert resolved == "MOC.md"

        # Test 4: Existing directory
        (tmp_path / "SubFolder").mkdir()
        mode, resolved = resolve_ingestion_target("SubFolder", tmpdir)
        assert mode == "ATOMIC_FOLDER_MODE"
        assert resolved == "SubFolder"

        # Test 5: Neither exists, defaults to ATOMIC_FOLDER_MODE
        mode, resolved = resolve_ingestion_target("NewFolder", tmpdir)
        assert mode == "ATOMIC_FOLDER_MODE"
        assert resolved == "NewFolder"


def test_fsm_target_properties(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(CONFIG, "vault_path", tmpdir)
        tmp_path = Path(tmpdir)
        
        # Scenario A: Atomic Folder
        fsm_folder = InjectorFSM("Inbox/test.md", "TargetFolder")
        assert fsm_folder.target_mode == "ATOMIC_FOLDER_MODE"
        assert fsm_folder.resolved_target == "TargetFolder"
        
        # Scenario B: File Append
        (tmp_path / "HubFile.md").touch()
        fsm_file = InjectorFSM("Inbox/test.md", "HubFile")
        assert fsm_file.target_mode == "FILE_APPEND_MODE"
        assert fsm_file.resolved_target == "HubFile.md"
