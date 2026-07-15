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


def test_cluster_merges_keys_sharing_stemmed_tokens(tmp_path, monkeypatch):
    from tests.eval.longmemeval.runner import question_vault

    run_root = tmp_path / "run"
    inst = _inst("q8", "multi-session", ["answer_s1", "answer_s2", "answer_s3"])
    # No shared 2-seg prefix, but token chains connect all three:
    # model/kit joins f1-f2, project joins f2-f3.
    _write_store(question_vault(run_root, "q8"), [
        _fact("f_0001", "model_kit.gifts", ["answer_s1"]),
        _fact("f_0002", "model_kits.last_project", ["answer_s2"]),
        _fact("f_0003", "user.current_project", ["answer_s3"]),
    ], monkeypatch, tmp_path)

    assert probe_question(inst, run_root, normalize=True)["best_coverage"] == 1
    r = probe_question(inst, run_root, cluster=True)
    assert r["groups"] == 1 and r["best_coverage"] == 3


def test_cluster_keeps_unrelated_apart_and_counts_riders(tmp_path, monkeypatch):
    from tests.eval.longmemeval.runner import question_vault

    run_root = tmp_path / "run"
    inst = _inst("q9", "knowledge-update", ["answer_s1", "answer_s2", "answer_s3"])
    _write_store(question_vault(run_root, "q9"), [
        _fact("f_0001", "user.dog.name", ["answer_s1"]),
        _fact("f_0002", "user.dog.age", ["answer_s2"]),
        _fact("f_0003", "user.dog.breed", ["other_session"]),  # non-gold rider
        _fact("f_0004", "user.car.model", ["answer_s3"]),
    ], monkeypatch, tmp_path)

    r = probe_question(inst, run_root, cluster=True)
    assert r["groups"] == 2            # dog cluster vs car cluster
    assert r["best_coverage"] == 2     # dog cluster: s1 + s2
    assert r["best_size"] == 3         # rider fact rides along — blob proxy


def test_cluster_drops_entity_prefix_but_not_topic(tmp_path, monkeypatch):
    from tests.eval.longmemeval.runner import question_vault

    run_root = tmp_path / "run"
    inst = _inst("q10", "multi-session", ["answer_s1", "answer_s2"])
    # user./assistant. prefixes must not keep the same topic apart...
    _write_store(question_vault(run_root, "q10"), [
        _fact("f_0001", "user.laundry.schedule", ["answer_s1"]),
        _fact("f_0002", "assistant.laundry.tips", ["answer_s2"]),
        # ...and must not glue unrelated topics together either.
        _fact("f_0003", "user.dog.name", ["answer_s1"]),
    ], monkeypatch, tmp_path)

    r = probe_question(inst, run_root, cluster=True)
    assert r["groups"] == 2 and r["best_coverage"] == 2


def test_run_probes_passes_cluster_through(tmp_path, monkeypatch):
    import silica.kernel.paths as paths_mod

    monkeypatch.setattr(paths_mod, "_SILICA_HOME", tmp_path / "silica_home")
    rows = run_probes([_inst("q1", "multi-session", ["answer_s1"])],
                      tmp_path / "run", cluster=True)
    assert rows[0]["groups"] == 0


def test_cluster_max_df_breaks_generic_token_glue(tmp_path, monkeypatch):
    from tests.eval.longmemeval.runner import question_vault

    run_root = tmp_path / "run"
    inst = _inst("q11", "multi-session", ["answer_s1", "answer_s2"])
    # "tips" (df=3) glues unrelated topics; "marathon" (df=2) is the real link.
    _write_store(question_vault(run_root, "q11"), [
        _fact("f_0001", "user.marathon.training_tips", ["answer_s1"]),
        _fact("f_0002", "user.marathon.race_date", ["answer_s2"]),
        _fact("f_0003", "user.cooking.tips", ["other_1"]),
        _fact("f_0004", "user.garden.tips", ["other_2"]),
    ], monkeypatch, tmp_path)

    naive = probe_question(inst, run_root, cluster=True)
    assert naive["best_coverage"] == 2 and naive["best_size"] == 4  # blob
    r = probe_question(inst, run_root, cluster=True, max_df=2)
    assert r["best_coverage"] == 2 and r["best_size"] == 2  # rare link kept


def test_best_group_tie_breaks_to_smallest(tmp_path, monkeypatch):
    from tests.eval.longmemeval.runner import question_vault

    run_root = tmp_path / "run"
    inst = _inst("q12", "knowledge-update", ["answer_s1"])
    _write_store(question_vault(run_root, "q12"), [
        _fact("f_0001", "user.a.b", ["answer_s1"]),
        _fact("f_0002", "user.c.d", ["answer_s1"]),
        _fact("f_0003", "user.c.d", ["other_session"]),
    ], monkeypatch, tmp_path)

    r = probe_question(inst, run_root)
    # Both keys cover s1; the smaller group must win the report.
    assert r["best_group"] == "user.a.b" and r["best_size"] == 1


def test_pairwise_probe_reports_product_attachment_groups(tmp_path, monkeypatch):
    from tests.eval.longmemeval.runner import question_vault

    run_root = tmp_path / "run"
    inst = _inst("q13", "knowledge-update", ["answer_s1", "answer_s2"])
    _write_store(question_vault(run_root, "q13"), [
        _fact("f_0001", "user.fitness.tournament.date", ["answer_s1"]),
        _fact("f_0002", "user.tennis_tournament_date", ["answer_s2"]),
        _fact("f_0003", "user.dog.name", ["other_session"]),
    ], monkeypatch, tmp_path)

    # Layer A alone cannot link the drifted pair...
    assert probe_question(inst, run_root, normalize=True)["best_coverage"] == 1
    # ...the product attachment rule can, and best_size stays honest.
    r = probe_question(inst, run_root, pairwise=True)
    assert r["best_coverage"] == 2
    assert r["best_size"] == 2       # the dog fact does not ride along
    assert r["groups"] == 1


def test_regroup_store_writes_groups_in_place_and_is_idempotent(tmp_path, monkeypatch):
    import silica.kernel.paths as paths_mod

    from tests.eval.longmemeval.probes import regroup_store
    from tests.eval.longmemeval.runner import question_vault

    run_root = tmp_path / "run"
    vault = question_vault(run_root, "q14")
    _write_store(vault, [
        _fact("f_0001", "user.fitness.tournament.date", ["answer_s1"]),
        _fact("f_0002", "user.tennis_tournament_date", ["answer_s2"]),
    ], monkeypatch, tmp_path)

    p = paths_mod.index_dir_for(str(vault)) / "episodic.json"
    regroup_store(p)
    facts = {f["id"]: f for f in
             json.loads(p.read_text(encoding="utf-8"))["facts"]}
    assert facts["f_0001"]["group"] == "f_0001"
    assert facts["f_0002"]["group"] == "f_0001"

    regroup_store(p)   # idempotent: replay from None gives the same result
    facts2 = {f["id"]: f for f in
              json.loads(p.read_text(encoding="utf-8"))["facts"]}
    assert facts2 == facts
