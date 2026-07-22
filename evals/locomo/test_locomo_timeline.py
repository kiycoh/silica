# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Offline test for the distill+timeline overlay (build_timeline_seed).

Zero LLM: a fixture vault of a few notes with `date` frontmatter, verified for
chronological ordering and — the crux the spec flags — that each emitted
pointer actually resolves the right note through the real driver (a pointer the
agent cannot open would invalidate the experiment).
"""
import re

from silica.driver import DRIVER
from evals.locomo import runner
from evals.longmemeval.runner import bind_vault


def _note(path, *, date=None, session_id=None, body="body"):
    fm = ["---"]
    if session_id is not None:
        fm.append(f'session_id: "{session_id}"')
    if date is not None:
        fm.append(f'date: "{date}"')
    fm += ["source: locomo", "---", "", body, ""]
    path.write_text("\n".join(fm), encoding="utf-8")


def _fixture(tmp_path):
    """Three dated notes, deliberately out of file order vs date order."""
    sess = tmp_path / "sessions"
    sess.mkdir()
    # file order a,b,c ; date order b(May) < a(Jun) < c(Jul)
    _note(sess / "a.md", date="2023-06-15", session_id="session_2", body="june body")
    _note(sess / "b.md", date="2023-05-01", session_id="session_1", body="may body")
    _note(sess / "c.md", date="2023-07-20", session_id="session_3", body="july body")
    return tmp_path


def _pointers(seed):
    """[(date, pointer)] per data row, in emitted order."""
    out = []
    for line in seed.splitlines():
        m = re.search(r"(\d{4}-\d{2}-\d{2}).*\(([^)]+)\)", line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def test_build_timeline_seed_orders_by_date_not_filename(tmp_path):
    seed = runner.build_timeline_seed(_fixture(tmp_path))
    assert seed.startswith("## Timeline")
    dates = [d for d, _ in _pointers(seed)]
    assert dates == ["2023-05-01", "2023-06-15", "2023-07-20"]  # ascending, not a,b,c


def test_build_timeline_seed_pointer_opens_the_right_note(tmp_path):
    vault = _fixture(tmp_path)
    seed = runner.build_timeline_seed(vault)
    bind_vault(vault)
    # Whatever string the parens carry, silica_read_note must open THAT note.
    by_date = dict(_pointers(seed))
    assert DRIVER.read_note(by_date["2023-05-01"]).content.strip().endswith("may body")
    assert DRIVER.read_note(by_date["2023-06-15"]).content.strip().endswith("june body")
    assert DRIVER.read_note(by_date["2023-07-20"]).content.strip().endswith("july body")


def test_build_timeline_seed_skips_undated_without_crashing(tmp_path):
    vault = _fixture(tmp_path)
    _note(vault / "sessions" / "d.md", session_id="session_4", body="undated body")
    seed = runner.build_timeline_seed(vault)  # must not raise
    ptrs = [p for _, p in _pointers(seed)]
    assert len(ptrs) == 3                       # d excluded, the 3 dated ones remain
    assert not any("d" == p.removesuffix(".md") for p in ptrs)


def _capture_messages(monkeypatch):
    """Patch run_agent + vmap; return a list that captures the messages sent."""
    import silica.agent.loop as loop_mod
    from silica.kernel import vault_map

    captured = []
    monkeypatch.setattr(vault_map, "build_vault_map", lambda: "VMAP")

    def fake(messages, model, tool_progress_callback=None, progress=None,
             cancel_token=None, constraints=None, temperature=None):
        captured.append(messages)
        return "ok"

    monkeypatch.setattr(loop_mod, "run_agent", fake)
    return captured


def test_timeline_seed_injected_after_vmap(monkeypatch):
    captured = _capture_messages(monkeypatch)
    runner.answer_question_agent("stub", "q?", "2023-05-09", ("Ann", "Bob"),
                                 timeline_seed="## Timeline\n1. x")
    roles = [m["role"] for m in captured[0]]
    assert roles == ["system", "system", "system", "user"]   # contract, vmap, seed, q
    assert captured[0][1]["content"] == "VMAP"
    assert captured[0][2]["content"] == "## Timeline\n1. x"   # right after the vmap


def test_no_timeline_seed_is_r_baseline(monkeypatch):
    captured = _capture_messages(monkeypatch)
    runner.answer_question_agent("stub", "q?", "2023-05-09", ("Ann", "Bob"))
    roles = [m["role"] for m in captured[0]]
    assert roles == ["system", "system", "user"]             # no seed message = R
