"""Tests for the expires_at temporal-forgetting linter."""
from __future__ import annotations

import datetime
from silica.kernel.linter import check_expires_at


def test_no_expires_at_returns_no_warning():
    data = {"title": "My Note", "tags": ["ai"]}
    warnings = check_expires_at(data)
    assert warnings == []


def test_expires_at_in_future_returns_no_warning():
    future = (datetime.date.today() + datetime.timedelta(days=30)).isoformat()
    data = {"expires_at": future}
    assert check_expires_at(data) == []


def test_expires_at_today_returns_no_warning():
    today = datetime.date.today().isoformat()
    data = {"expires_at": today}
    assert check_expires_at(data) == []


def test_expires_at_in_past_returns_warning():
    past = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    data = {"expires_at": past}
    warnings = check_expires_at(data)
    assert len(warnings) == 1
    assert "expired" in warnings[0].lower()


def test_expires_at_far_past_mentions_date():
    past = "2020-01-01"
    data = {"expires_at": past}
    warnings = check_expires_at(data)
    assert "2020-01-01" in warnings[0]


def test_expires_at_invalid_format_returns_warning():
    data = {"expires_at": "not-a-date"}
    warnings = check_expires_at(data)
    assert len(warnings) == 1
    assert "invalid" in warnings[0].lower() or "expires_at" in warnings[0].lower()


def test_expires_at_none_returns_no_warning():
    data = {"expires_at": None}
    assert check_expires_at(data) == []


def test_check_expires_at_integrated_in_validate_note(tmp_path, monkeypatch):
    """validate_note emits an expires_at warning for an expired note."""
    from silica.kernel import frontmatter
    import datetime

    past = (datetime.date.today() - datetime.timedelta(days=5)).isoformat()
    note_path = tmp_path / "expired_note.md"
    note_path.write_text(
        f"---\ntitle: Expired\nexpires_at: {past}\n---\n\n[[Hub]]\n\nContent here.\n",
        encoding="utf-8",
    )

    from silica.driver.fs_backend import ObsidianFSBackend
    import silica.kernel.linter as linter_mod
    monkeypatch.setattr(linter_mod, "DRIVER", ObsidianFSBackend(str(tmp_path)))

    from silica.kernel.linter import validate_note
    errors, warnings = validate_note(str(note_path), hub=None, op_type=None)
    assert any("expired" in w.lower() for w in warnings), f"warnings={warnings}"
