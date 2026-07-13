"""Regression: cli backend create must mkdir a missing target folder.

Root cause: /nucleate --target=NewDir on the cli backend failed every write —
Obsidian's app.vault.create (and the adapter.write fallback) raise if the
parent folder doesn't exist, and the pipeline deferred the failures silently.
fs_backend.create already does mkdir(parents=True); create now routes through
_ensure_dest_dir (the same seam move() uses) for parity.
"""
import os

from silica.driver.cli_backend import ObsidianCLIBackend


def _detached_backend():
    """An instance without __init__ (no live Obsidian needed)."""
    return ObsidianCLIBackend.__new__(ObsidianCLIBackend)


def test_create_mkdirs_missing_target_folder(tmp_path, monkeypatch):
    be = _detached_backend()
    be._base_path = str(tmp_path)  # pre-cache the vault FS root
    monkeypatch.setattr(be, "_reject_hidden", lambda p: None)
    monkeypatch.setattr(be, "_eval", lambda js: "ok")
    monkeypatch.setattr(be, "_wait_for_content_reflects", lambda *a, **k: None)
    monkeypatch.setattr(be, "_wait_for_resolved_event", lambda *a, **k: None)
    monkeypatch.setattr(be, "_patch_graph_add", lambda *a, **k: None)

    be.create("Concepts/New Area/Note.md", "body")

    assert os.path.isdir(tmp_path / "Concepts" / "New Area")


def test_ensure_dest_dir_skips_root_level_paths(monkeypatch):
    be = _detached_backend()

    def no_eval(*a, **k):
        raise AssertionError("root-level path must not trigger a basePath eval")

    monkeypatch.setattr(be, "_eval", no_eval)
    be._ensure_dest_dir("Note.md")  # no parent → no-op, no eval
