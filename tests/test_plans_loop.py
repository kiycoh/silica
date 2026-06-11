from pathlib import Path

from silica.kernel.plans import VALID_STATUS, iter_plan_notes, status_counts


def _plan(vault: Path, name: str, status: str) -> None:
    p = vault / "plans"
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{name}.md").write_text(
        f"---\ntags:\n  - plan\nstatus: {status}\n---\n\n# {name}\n",
        encoding="utf-8",
    )


def test_status_counts_buckets_by_enum(tmp_path):
    _plan(tmp_path, "a", "todo")
    _plan(tmp_path, "b", "in-progress")
    _plan(tmp_path, "c", "in-progress")
    _plan(tmp_path, "d", "done")
    counts = status_counts(tmp_path)
    assert counts == {"todo": 1, "in-progress": 2, "done": 1}


def test_status_counts_ignores_out_of_enum_status(tmp_path):
    _plan(tmp_path, "a", "todo")
    _plan(tmp_path, "x", "wip")  # not in VALID_STATUS
    counts = status_counts(tmp_path)
    assert counts == {"todo": 1}
    assert "wip" not in counts


def test_status_counts_empty_when_no_plans(tmp_path):
    assert status_counts(tmp_path) == {}


def test_iter_plan_notes_yields_path_and_includes_unstatused(tmp_path):
    p = tmp_path / "plans" / "sub"
    p.mkdir(parents=True)
    # nested note with frontmatter but no status: still yielded
    (p / "nested.md").write_text("---\ntags:\n  - plan\n---\n\n# nested\n", encoding="utf-8")
    # note without any frontmatter: skipped
    (tmp_path / "plans" / "raw.md").write_text("# no frontmatter\n", encoding="utf-8")

    results = list(iter_plan_notes(str(tmp_path)))  # pass a str to lock Path normalization
    assert len(results) == 1
    note_path, data = results[0]
    assert isinstance(note_path, Path)
    assert note_path.name == "nested.md"
    assert isinstance(data, dict)


def test_valid_status_set():
    assert VALID_STATUS == {"todo", "in-progress", "blocked", "done"}


def test_check_plan_status_warns_on_bad_enum():
    from silica.kernel.linter import check_plan_status
    assert check_plan_status({"status": "in-progress"}) == []
    assert check_plan_status({}) == []  # absent → no warning
    assert check_plan_status(None) == []  # None data dict → guarded, no warning
    bad = check_plan_status({"status": "wip"})
    assert len(bad) == 1 and "wip" in bad[0]
    # non-string status (YAML int) is coerced via str() and warned on
    bad_int = check_plan_status({"status": 1})
    assert len(bad_int) == 1 and "1" in bad_int[0]


def test_digest_includes_plans_line(tmp_path, monkeypatch):
    from silica.config import CONFIG
    from silica.planner.progress import ProgressLedger

    p = tmp_path / "plans"
    p.mkdir(parents=True)
    (p / "a.md").write_text("---\nstatus: in-progress\n---\n# a\n", encoding="utf-8")
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))

    ledger = ProgressLedger.new(mode="inject", inputs={})
    digest = ledger.digest()
    assert "PLANS:" in digest
    assert "in-progress" in digest


def test_digest_omits_plans_line_when_no_plans(tmp_path, monkeypatch):
    # No plans/ dir → status_counts is empty → the `if counts:` guard omits the line.
    from silica.config import CONFIG
    from silica.planner.progress import ProgressLedger

    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))

    ledger = ProgressLedger.new(mode="inject", inputs={})
    digest = ledger.digest()
    assert "PLANS:" not in digest


def test_plans_command_handled_and_reports(tmp_path, monkeypatch, capsys):
    from silica.config import CONFIG
    from silica.cli import _handle_direct_shortcut

    p = tmp_path / "plans"
    p.mkdir(parents=True)
    (p / "alpha.md").write_text("---\nstatus: todo\n---\n# alpha\n", encoding="utf-8")
    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))

    handled = _handle_direct_shortcut("/plans", [])
    assert handled is True
    out = capsys.readouterr().out
    assert "alpha" in out
    assert "todo" in out
    # Per-note line must keep the literal [status] bracket — rich would
    # otherwise swallow [todo] as an unknown markup tag (regression guard).
    assert "[todo] alpha" in out


def test_plans_command_no_vault(monkeypatch, capsys):
    from silica.config import CONFIG
    from silica.cli import _handle_direct_shortcut

    monkeypatch.setattr(CONFIG, "vault_path", "")
    handled = _handle_direct_shortcut("/plans", [])
    assert handled is True
    assert "No vault configured" in capsys.readouterr().out


def test_plans_command_no_plans(tmp_path, monkeypatch, capsys):
    from silica.config import CONFIG
    from silica.cli import _handle_direct_shortcut

    monkeypatch.setattr(CONFIG, "vault_path", str(tmp_path))  # no plans/ dir
    handled = _handle_direct_shortcut("/plans", [])
    assert handled is True
    assert "No plans found" in capsys.readouterr().out
