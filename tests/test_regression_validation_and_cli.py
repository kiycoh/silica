from unittest.mock import patch

import pytest

from silica.driver.cli_backend import ObsidianCLIBackend
from silica.kernel.validate import validate_operations


def test_write_rejects_sibling_directory_with_same_prefix(tmp_path):
    """A target_dir prefix match must not allow writes into sibling folders."""
    target_dir = tmp_path / "Dir"
    sibling_dir = tmp_path / "Directory"
    target_dir.mkdir()
    sibling_dir.mkdir()

    ops = [
        {
            "op": "write",
            "path": str(sibling_dir / "Bad.md"),
            "heading": "Bad",
            "source_basename": "inbox.md",
        }
    ]

    with patch("silica.kernel.validate.DRIVER.read_note", side_effect=RuntimeError("not found")):
        validated, rejected = validate_operations(ops, [], str(target_dir))

    assert not validated
    assert len(rejected) == 1
    assert "not in target folder" in rejected[0].reason


def test_cli_backend_reports_missing_obsidian_as_runtime_error(monkeypatch):
    """Missing Obsidian CLI should follow the backend's RuntimeError contract."""

    def missing_binary(*args, **kwargs):
        raise FileNotFoundError("obsidian")

    monkeypatch.setattr("silica.driver.cli_backend.subprocess.run", missing_binary)

    with pytest.raises(RuntimeError, match="Obsidian CLI executable not found"):
        ObsidianCLIBackend()._run_cli("files", "ext=md")
