# tests/test_revert_command.py
from silica.ui.commands import command_names
from silica.cli import _handle_direct_shortcut


def test_revert_is_registered():
    assert "/revert" in command_names()


def test_revert_invokes_revert_run(monkeypatch):
    class _FakeStore:
        def last_active_run(self):
            return "RUN1abc"

    monkeypatch.setattr("silica.kernel.undo_journal.get_undo_journal",
                        lambda: _FakeStore())
    calls = {}
    def fake_revert(run_id, **kw):
        calls["run_id"] = run_id
        return {"run_id": run_id, "reverted": ["a.md"], "skipped": [], "errors": []}
    monkeypatch.setattr("silica.kernel.undo_journal.revert_run", fake_revert)

    handled = _handle_direct_shortcut("/revert", [])
    assert handled is True
    assert calls["run_id"] == "RUN1abc"


def test_revert_no_active_run(monkeypatch, capsys):
    class _EmptyStore:
        def last_active_run(self):
            return None

    monkeypatch.setattr("silica.kernel.undo_journal.get_undo_journal",
                        lambda: _EmptyStore())
    handled = _handle_direct_shortcut("/revert", [])
    assert handled is True
    captured = capsys.readouterr()
    # Should print something about nothing to undo
    assert handled is True
