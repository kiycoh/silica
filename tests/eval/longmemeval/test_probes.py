# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Key-drift probes: grouping, capture ceiling, verbatim session ids.

Stores are written straight to disk at the probe's expected location
(`index_dir_for(question_vault)/episodic.json`) with `_SILICA_HOME`
monkeypatched into tmp_path — no capture(), no LLM, no global state."""
from __future__ import annotations

import json
from pathlib import Path

from tests.eval.longmemeval.probes import probe_question, run_probes


def _fact(fid: str, key: str, runs: list[str], status: str = "live") -> dict:
    return {"id": fid, "key": key, "text": f"text {fid}",
            "first_seen": "2026-01-01", "last_seen": "2026-01-01",
            "runs": runs, "supersedes": None, "status": status}


def _write_store(vault: Path, facts: list[dict], monkeypatch, tmp_path: Path) -> None:
    import silica.kernel.paths as paths_mod

    monkeypatch.setattr(paths_mod, "_SILICA_HOME", tmp_path / "silica_home")
    d = paths_mod.index_dir_for(str(vault))
    d.mkdir(parents=True, exist_ok=True)
    (d / "episodic.json").write_text(
        json.dumps({"schema_version": 1, "next_id": len(facts) + 1,
                    "facts": facts}), encoding="utf-8")


def _inst(qid: str, qtype: str, gold: list[str]) -> dict:
    return {"question_id": qid, "question_type": qtype,
            "answer_session_ids": gold}


def test_aggregative_groups_by_two_segment_prefix(tmp_path, monkeypatch):
    from tests.eval.longmemeval.runner import question_vault

    run_root = tmp_path / "run"
    inst = _inst("q1", "multi-session", ["answer_s1", "answer_s2"])
    _write_store(question_vault(run_root, "q1"), [
        _fact("f_0001", "model_kit.gifts", ["answer_s1"]),
        _fact("f_0002", "model_kits.last_project", ["answer_s2"]),
        _fact("f_0003", "model_kit.dead", ["answer_s1"], status="superseded"),
        _fact("f_0004", "user.unrelated", ["other_session"]),
    ], monkeypatch, tmp_path)

    r = probe_question(inst, run_root)
    assert r["captured_sessions"] == 2 and r["gold_sessions"] == 2
    assert r["gold_facts"] == 2          # superseded + non-gold excluded
    assert r["groups"] == 2              # model_kit vs model_kits: the drift
    assert r["best_coverage"] == 1


def test_ku_groups_by_full_key(tmp_path, monkeypatch):
    from tests.eval.longmemeval.runner import question_vault

    run_root = tmp_path / "run"
    inst = _inst("q2", "knowledge-update", ["answer_s1", "answer_s2"])
    _write_store(question_vault(run_root, "q2"), [
        _fact("f_0001", "user.car.model", ["answer_s1"]),
        _fact("f_0002", "user.car.color", ["answer_s2"]),
    ], monkeypatch, tmp_path)

    r = probe_question(inst, run_root)
    # Full-key grouping: same 2-seg prefix but different attributes stay apart.
    assert r["groups"] == 2 and r["best_coverage"] == 1

    inst3 = _inst("q3", "knowledge-update", ["answer_s1", "answer_s2"])
    _write_store(question_vault(run_root, "q3"), [
        _fact("f_0001", "user.car.model", ["answer_s1"]),
        _fact("f_0002", "user.car.model", ["answer_s2"]),
    ], monkeypatch, tmp_path)
    r3 = probe_question(inst3, run_root)
    assert r3["groups"] == 1 and r3["best_coverage"] == 2


def test_answer_prefix_is_part_of_the_id(tmp_path, monkeypatch):
    from tests.eval.longmemeval.runner import question_vault

    run_root = tmp_path / "run"
    inst = _inst("q4", "multi-session", ["answer_s1"])
    # Fact recorded under bare "s1": must NOT count as gold coverage.
    _write_store(question_vault(run_root, "q4"),
                 [_fact("f_0001", "user.car.model", ["s1"])],
                 monkeypatch, tmp_path)

    r = probe_question(inst, run_root)
    assert r["captured_sessions"] == 0 and r["best_coverage"] == 0


def test_missing_store_reports_zero(tmp_path, monkeypatch):
    import silica.kernel.paths as paths_mod

    monkeypatch.setattr(paths_mod, "_SILICA_HOME", tmp_path / "silica_home")
    r = probe_question(_inst("q5", "multi-session", ["answer_s1"]),
                       tmp_path / "run")
    assert r["captured_sessions"] == 0 and r["groups"] == 0
    assert r["best_group"] == "-"


def test_run_probes_filters_to_probed_types(tmp_path, monkeypatch):
    import silica.kernel.paths as paths_mod

    monkeypatch.setattr(paths_mod, "_SILICA_HOME", tmp_path / "silica_home")
    data = [_inst("q1", "multi-session", ["answer_s1"]),
            _inst("q2", "single-session-user", ["answer_s1"]),
            _inst("q3", "knowledge-update", ["answer_s1"])]
    rows = run_probes(data, tmp_path / "run")
    assert [r["question_id"] for r in rows] == ["q1", "q3"]


def test_normalize_merges_plural_split_prefixes(tmp_path, monkeypatch):
    from tests.eval.longmemeval.runner import question_vault

    run_root = tmp_path / "run"
    inst = _inst("q6", "multi-session", ["answer_s1", "answer_s2"])
    _write_store(question_vault(run_root, "q6"), [
        _fact("f_0001", "model_kit.gifts", ["answer_s1"]),
        _fact("f_0002", "model_kits.gifts", ["answer_s2"]),
    ], monkeypatch, tmp_path)

    plain = probe_question(inst, run_root)
    assert plain["groups"] == 2 and plain["best_coverage"] == 1
    merged = probe_question(inst, run_root, normalize=True)
    assert merged["groups"] == 1 and merged["best_coverage"] == 2


def test_normalize_merges_ku_full_key_variants(tmp_path, monkeypatch):
    from tests.eval.longmemeval.runner import question_vault

    run_root = tmp_path / "run"
    inst = _inst("q7", "knowledge-update", ["answer_s1", "answer_s2"])
    _write_store(question_vault(run_root, "q7"), [
        _fact("f_0001", "user.car.model", ["answer_s1"]),
        _fact("f_0002", "user.car.models", ["answer_s2"]),
    ], monkeypatch, tmp_path)

    assert probe_question(inst, run_root)["best_coverage"] == 1
    assert probe_question(inst, run_root, normalize=True)["best_coverage"] == 2
