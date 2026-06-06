"""autolink_note: CLI delegates to Obsidian; FS uses the pure kernel."""
from unittest.mock import patch
from silica.driver.fs_backend import ObsidianFSBackend
from silica.driver.cli_backend import ObsidianCLIBackend
from silica.driver.base import NoteRef
import json as _json


def test_fs_autolink_note_uses_kernel(tmp_path, monkeypatch):
    import silica.config
    vault = tmp_path / "v"
    vault.mkdir()
    (vault / "Neural Networks.md").write_text("# Neural Networks\n")
    note = vault / "Topic.md"
    note.write_text("Neural Networks are powerful.\n")
    monkeypatch.setattr(silica.config.CONFIG, "vault_path", str(vault))

    backend = ObsidianFSBackend(vault_path=str(vault))
    backend._rebuild_index()
    added = backend.autolink_note("Topic.md", candidates=["Neural Networks"])
    assert "Neural Networks" in added
    assert "[[Neural Networks]]" in note.read_text()


def test_cli_autolink_note_delegates_via_eval():
    backend = ObsidianCLIBackend(vault_name="t")
    captured = {}

    def fake_run_cli(*args, **kwargs):
        captured["code"] = args[1]
        return '=> ' + _json.dumps({"added": ["Neural Networks"]})

    with patch.object(backend, "_run_cli", side_effect=fake_run_cli):
        added = backend.autolink_note("Topic.md", candidates=["Neural Networks"])

    assert added == ["Neural Networks"]
    code = captured["code"]
    assert "getFileCache" in code
    assert "getFirstLinkpathDest" in code
    assert "generateMarkdownLink" in code


def test_cli_autolink_note_escapes_candidates_and_path():
    backend = ObsidianCLIBackend(vault_name="t")
    captured = {}
    with patch.object(backend, "_run_cli", side_effect=lambda *a, **k: captured.update(code=a[1]) or '=> {"added": []}'):
        backend.autolink_note("a'b.md", candidates=["x'y"])
    assert r"a\'b.md" in captured["code"]


def test_cli_autolink_note_empty_candidates_short_circuits():
    backend = ObsidianCLIBackend(vault_name="t")
    with patch.object(backend, "_run_cli") as run:
        assert backend.autolink_note("Topic.md", candidates=[]) == []
    run.assert_not_called()


def test_silica_autolink_tool_routes_to_driver(monkeypatch):
    import silica.tools.composed as composed
    calls = []

    class FakeDriver:
        def list_files(self):
            return [NoteRef(name="Neural Networks", path="Neural Networks.md")]
        def read_note(self, p):
            return type("NC", (), {"content": "Neural Networks rock."})()
        def autolink_note(self, path, candidates=None):
            calls.append((path, candidates))
            return ["Neural Networks"]

    monkeypatch.setattr(composed, "DRIVER", FakeDriver(), raising=False)
    out = composed.silica_autolink(note_path="Topic.md", use_candidates=False)
    assert calls and calls[0][0] == "Topic.md"
    # use_candidates=False → candidates None → routed with the pre-built title_index
    assert calls[0][1] == ["Neural Networks"]
    assert out.get("total_links_added") == 1
    assert out.get("notes_processed") == 1
