# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""End-to-end check for the LongMemEval adapter, fully offline.

Stubs the LLM (answer + judge) so load -> retrieve -> answer -> judge ->
aggregate runs with no network. The stub answer echoes the retrieved context,
and the stub judge says 'yes' iff the gold string appears in it; so a correct
retrieval of the evidence session yields a correct verdict, wiring the whole
pipeline to real behavior.
"""
from __future__ import annotations

import json

import silica.agent.llm as llm_mod
from silica.agent.llm import LLMResponse
from evals.longmemeval import runner


def _session(text):
    return [{"role": "user", "content": text},
            {"role": "assistant", "content": "noted."}]


def _instance():
    # Evidence session s1 holds the gold token; s0/s2 are distractor filler.
    return {
        "question_id": "q_multi_1",
        "question_type": "multi-session",
        "question": "What is the user's dog's name?",
        "answer": "Zephyr",
        "question_date": "2026-05-01",
        "haystack_session_ids": ["s0", "s1", "s2"],
        "haystack_dates": ["2026-01-01", "2026-02-01", "2026-03-01"],
        "haystack_sessions": [
            _session("I moved to Berlin last winter."),
            _session("My dog Zephyr turned three today."),
            _session("The weather has been rainy all week."),
        ],
        "answer_session_ids": ["s1"],
    }


def _abs_instance():
    # Real LongMemEval _abs shape: the gold session id is a synthetic marker
    # (answer_..._abs) that is NOT part of the haystack, so retrieval can never
    # hit it — session recall is undefined for abstention, not 0.0.
    return {
        "question_id": "q_multi_2_abs",
        "question_type": "multi-session",
        "question": "What is the user's cat's name?",
        "answer": "The user never mentioned a cat.",
        "question_date": "2026-05-01",
        "haystack_session_ids": ["s0"],
        "haystack_dates": ["2026-01-01"],
        "haystack_sessions": [_session("I moved to Berlin last winter.")],
        "answer_session_ids": ["answer_deadbeef_abs"],
    }


def _install_stub(monkeypatch, gold="Zephyr"):
    def fake_call_llm(model, messages, **kw):
        prompt = messages[-1]["content"]
        if "Answer yes or no" in prompt:  # judge turn
            return LLMResponse(text="yes" if gold.lower() in prompt.lower() else "no")
        return LLMResponse(text=prompt)  # answer turn: echo the memory context
    monkeypatch.setattr(llm_mod, "call_llm", fake_call_llm)


def test_stuff_pipeline_scores_correct(tmp_path, monkeypatch):
    _install_stub(monkeypatch)
    doc = runner.run([_instance()], tmp_path / "run", model="stub", judge_model="stub",
                     k=10, stuff=True, use_embedder=False, limit=None, verbose=False)
    m = doc["metrics"]
    assert m["overall_accuracy"] == 1.0     # gold session was fed -> judge says yes
    assert m["answerable_n"] == 1
    assert m["by_type"]["multi-session"]["accuracy"] == 1.0
    # --stuff feeds all sessions, so per-question session recall is not computed.
    assert m["session_recall_mean"] is None


def test_retrieval_finds_evidence_session_cooccur(tmp_path, monkeypatch):
    _install_stub(monkeypatch)
    # Real cooccur retrieval (offline): the question shares "dog" with s1 only.
    doc = runner.run([_instance()], tmp_path / "run", model="stub", judge_model="stub",
                     k=2, stuff=False, use_embedder=False, limit=None, verbose=False)
    row = doc["questions"][0]
    assert row["session_recall"] == 1.0     # evidence session s1 retrieved
    assert doc["metrics"]["overall_accuracy"] == 1.0


def test_retrieval_only_is_llm_free(tmp_path, monkeypatch):
    # No stub: any answer/judge call would blow up, proving the mode skips the LLM.
    def boom(*a, **k):
        raise AssertionError("call_llm invoked in --retrieval-only mode")
    monkeypatch.setattr(llm_mod, "call_llm", boom)
    doc = runner.run([_instance()], tmp_path / "run", model="stub", judge_model="stub",
                     k=2, stuff=False, use_embedder=False, retrieval_only=True,
                     limit=None, verbose=False)
    row = doc["questions"][0]
    assert row["correct"] is None                 # nothing graded
    assert row["session_recall"] == 1.0           # cooccur still finds evidence session s1
    m = doc["metrics"]
    assert m["overall_accuracy"] is None          # aggregate tolerates ungraded rows
    assert m["session_recall_mean"] == 1.0        # the retrieval signal survives
    assert m["by_type"]["multi-session"]["session_recall"] == 1.0


def test_abstention_session_recall_is_none_not_zero(tmp_path, monkeypatch):
    # The synthetic gold id can never be retrieved; scoring it 0.0 would drag
    # session_recall_mean down with a false miss on the full 500-q set.
    def boom(*a, **k):
        raise AssertionError("no LLM in retrieval-only")
    monkeypatch.setattr(llm_mod, "call_llm", boom)
    doc = runner.run([_abs_instance()], tmp_path / "run", model="stub", judge_model="stub",
                     k=5, stuff=False, use_embedder=False, retrieval_only=True,
                     limit=None, verbose=False)
    assert doc["questions"][0]["session_recall"] is None
    assert doc["metrics"]["session_recall_mean"] is None


def test_abstention_counted_separately(tmp_path, monkeypatch):
    # Gold cat name is absent; stub judge says 'no' -> abstention handled as its
    # own bucket, kept out of overall answerable accuracy.
    _install_stub(monkeypatch, gold="Whiskers")
    doc = runner.run([_instance(), _abs_instance()], tmp_path / "run",
                     model="stub", judge_model="stub", k=10, stuff=True,
                     use_embedder=False, limit=None, verbose=False)
    m = doc["metrics"]
    assert m["answerable_n"] == 1           # the _abs question excluded from answerable
    assert m["abstention_n"] == 1
    assert m["abstention_accuracy"] == 0.0  # stub judged the abs answer 'no'


def test_note_rendering_carries_date_and_turns():
    note = runner._note("s7", "2026-02-01", _session("hello world"))
    assert 'session_id: "s7"' in note
    assert 'date: "2026-02-01"' in note
    assert "User: hello world" in note and "Assistant: noted." in note


def test_distill_routes_sessions_through_distiller(tmp_path, monkeypatch):
    # --distill sends each session's transcript to the Silica distiller and
    # stores the distilled body (not the verbatim turns), frontmatter intact.
    import silica.kernel.prep_delegation as prep

    payloads = []

    def fake_distiller(payload, target, **kw):
        payloads.append(payload)
        return {"updates": [{"op": "create", "snippet": "Fact: the dog is Zephyr."}]}

    monkeypatch.setattr(prep, "run_distiller", fake_distiller)
    vault = tmp_path / "v"
    runner.bind_vault(vault)
    runner.load_question_vault(vault, _instance(), distill=True)

    note = (vault / "sessions" / "s0001.md").read_text(encoding="utf-8")
    assert "Fact: the dog is Zephyr." in note          # distilled body written
    assert "My dog Zephyr turned three today." not in note  # not the raw transcript
    assert 'session_id: "s1"' in note and 'date: "2026-02-01"' in note
    # question-blind ingest: the eval question never reaches the distiller.
    blob = json.dumps(payloads)
    assert _instance()["question"] not in blob
    assert "My dog Zephyr turned three today." in blob  # session content did reach it


def test_distill_ignores_bodies_on_skip_ops(tmp_path, monkeypatch):
    # Contract says skip ops carry no body, but models attach meta-lines
    # ("no durable facts to extract") anyway — those must never reach the
    # session note, while write/patch bodies still do.
    import silica.kernel.prep_delegation as prep

    monkeypatch.setattr(prep, "run_distiller", lambda payload, target, **kw: {
        "updates": [
            {"op": "skip", "snippet": "Two friends exchange greetings; nothing durable."},
            {"op": "write", "snippet": "Fact: the dog is Zephyr."},
        ]})
    vault = tmp_path / "v"
    runner.bind_vault(vault)
    runner.load_question_vault(vault, _instance(), distill=True)

    note = (vault / "sessions" / "s0001.md").read_text(encoding="utf-8")
    assert "Fact: the dog is Zephyr." in note
    assert "Two friends exchange greetings" not in note


def test_distill_all_skip_bodies_falls_back_to_verbatim(tmp_path, monkeypatch):
    # When skip meta-lines are the ONLY bodies, dropping them must trigger
    # the verbatim fallback, not write an empty/meta note.
    import silica.kernel.prep_delegation as prep

    monkeypatch.setattr(prep, "run_distiller", lambda payload, target, **kw: {
        "updates": [{"op": "skip", "snippet": "Greetings only; skipping."}]})
    vault = tmp_path / "v"
    runner.bind_vault(vault)
    runner.load_question_vault(vault, _instance(), distill=True)

    note = (vault / "sessions" / "s0001.md").read_text(encoding="utf-8")
    assert "My dog Zephyr turned three today." in note   # verbatim preserved
    assert "Greetings only" not in note


def test_distill_falls_back_to_verbatim_when_distiller_yields_nothing(tmp_path, monkeypatch):
    # A distiller error/empty output must not drop the session: keep it verbatim.
    import silica.kernel.prep_delegation as prep

    monkeypatch.setattr(prep, "run_distiller", lambda payload, target, **kw: {"error": "boom"})
    vault = tmp_path / "v"
    runner.bind_vault(vault)
    runner.load_question_vault(vault, _instance(), distill=True)

    note = (vault / "sessions" / "s0001.md").read_text(encoding="utf-8")
    assert "My dog Zephyr turned three today." in note   # verbatim transcript preserved
    assert 'session_id: "s1"' in note


def test_distill_flag_recorded_in_run_config(tmp_path, monkeypatch):
    # The flag threads through run() into the report config so results are labelled.
    import silica.kernel.prep_delegation as prep

    _install_stub(monkeypatch)
    monkeypatch.setattr(prep, "run_distiller",
                        lambda payload, target, **kw: {"updates": [{"snippet": "dog Zephyr"}]})
    doc = runner.run([_instance()], tmp_path / "run", model="stub", judge_model="stub",
                     k=10, stuff=True, use_embedder=False, distill=True, limit=None, verbose=False)
    assert doc["config"]["distill"] is True


def test_bind_vault_scopes_episodic_home_to_question_vault(tmp_path):
    # Isolation: episodic facts land in the per-question vault, while the
    # memory lane keeps abstaining (coincident-vault rule) so the adapter's
    # "personal-memory lane is never passed" guarantee holds.
    from silica.kernel.episodic import episodic_home
    from silica.kernel.memory_lane import memory_vault

    vault = tmp_path / "v"
    vault.mkdir()
    runner.bind_vault(vault)
    assert episodic_home() == vault.resolve()
    assert memory_vault() is None


def _ephemeral_distiller(by_session):
    def fake(payload, target, **kw):
        sid = payload["batches"][0]["inbox_file"]
        out = {"updates": [{"snippet": f"note for {sid}"}]}
        if sid in by_session:
            out["ephemerals"] = by_session[sid]
        return out
    return fake


def test_distill_captures_ephemerals_and_answer_recalls_them(tmp_path, monkeypatch):
    import silica.kernel.prep_delegation as prep

    monkeypatch.setattr(prep, "run_distiller", _ephemeral_distiller({
        "s1": [{"key": "user.dog.name", "text": "My dog is named Zephyr"}],
    }))
    seen = {}

    def capture_llm(model, messages, **kw):
        prompt = messages[-1]["content"]
        if "Answer yes or no" in prompt:
            return LLMResponse(text="yes")
        seen["answer_prompt"] = prompt
        return LLMResponse(text="Zephyr")

    monkeypatch.setattr(llm_mod, "call_llm", capture_llm)
    doc = runner.run([_instance()], tmp_path / "run", model="stub", judge_model="stub",
                     k=10, stuff=True, use_embedder=False, distill=True,
                     limit=None, verbose=False)
    # The answer step got a Personal memory block next to the session notes.
    assert "Personal memory:" in seen["answer_prompt"]
    assert "My dog is named Zephyr" in seen["answer_prompt"]
    # The recalled fact came from gold session s1.
    row = doc["questions"][0]
    assert row["ephemeral_hit"] is True
    assert doc["metrics"]["ephemeral_hit_mean"] == 1.0
    # Headline run is Mem0/Zep-comparable: no evaporation unless opted in.
    assert doc["config"]["episodic_ttl"] == 0


def test_distill_passes_session_date_to_distiller(tmp_path, monkeypatch):
    # F2a: the SESSION date (simulated time) reaches the distiller prompt so
    # relative dates in facts resolve against it, not against today.
    import silica.kernel.prep_delegation as prep

    seen_kwargs = {}

    def spy(payload, target, **kw):
        seen_kwargs[payload["batches"][0]["inbox_file"]] = kw.get("session_date")
        return {"updates": [{"snippet": "note"}]}

    monkeypatch.setattr(prep, "run_distiller", spy)
    _install_stub(monkeypatch)
    runner.run([_instance()], tmp_path / "run", model="stub", judge_model="stub",
               k=10, stuff=True, use_embedder=False, distill=True,
               limit=None, verbose=False)
    assert seen_kwargs["s1"] == "2026-02-01"  # s1's haystack date


def test_key_schema_flag_enforces_keys_through_product_seam(tmp_path, monkeypatch):
    # ADR-0021 lever: --key-schema drops a manifest into the question vault so
    # capture_from_distill -> load_manifest(episodic_home()) -> enforce runs
    # the EXACT product path, no harness-only shortcut.
    import silica.kernel.prep_delegation as prep

    monkeypatch.setattr(prep, "run_distiller", _ephemeral_distiller({
        "s1": [{"key": "dog.name", "text": "My dog is named Zephyr"},
               {"key": "assistant.recipe.oven.temp.exact", "text": "180C"}],
    }))
    _install_stub(monkeypatch)
    doc = runner.run([_instance()], tmp_path / "run", model="stub", judge_model="stub",
                     k=10, stuff=True, use_embedder=False, distill=True,
                     key_schema=True, limit=None, verbose=False)
    vault = runner.question_vault(tmp_path / "run", "q_multi_1")
    assert (vault / "vault.yaml").is_file()
    # run() leaves the last question bound, so store_path() IS the product
    # lookup for this vault's episodic store.
    from silica.kernel.episodic import store_path

    store = json.loads(store_path().read_text())
    keys = {f["key"] for f in store["facts"]}
    assert "user.dog.name" in keys
    assert "assistant.recipe.oven_temp_exact" in keys
    assert doc["config"]["key_schema"] is True


def test_context_assembly_is_the_product_perception(tmp_path, monkeypatch):
    # Perception promotion (spec 2026-07-15): the harness owns no context
    # assembler. The prompt's Memory section must be byte-identical to
    # silica.kernel.perception.perceive().render() on the same vault — the
    # by-construction guarantee that eval and product cannot diverge, checked
    # once end-to-end.
    seen = {}

    def capture_llm(model, messages, **kw):
        prompt = messages[-1]["content"]
        if "Answer yes or no" in prompt:
            return LLMResponse(text="yes")
        seen["answer_prompt"] = prompt
        return LLMResponse(text="Zephyr")

    monkeypatch.setattr(llm_mod, "call_llm", capture_llm)
    inst = _instance()
    runner.run([inst], tmp_path / "run", model="stub", judge_model="stub",
               k=2, stuff=False, use_embedder=False, limit=None, verbose=False)

    from silica.kernel.perception import perceive

    runner.bind_vault(tmp_path / "run" / inst["question_id"])
    p = perceive(inst["question"], now=inst["question_date"], k=2,
                 use_embedder=False, with_facts=False)
    expected = p.render()
    assert expected and expected in seen["answer_prompt"]
    assert "[#1" in expected and "dated 2026-" in expected  # perception headers live


def test_flat_context_keeps_legacy_wall(tmp_path, monkeypatch):
    # --flat-context reproduces the legacy wall-of-prose arm through the same
    # product renderer (windowed=False): full bodies, no rank headers.
    long_note = "filler chatter " * 400 + "the yoga class is on Tuesday evening"
    inst = _instance()
    inst["haystack_sessions"][1] = _session(long_note)
    seen = {}

    def capture_llm(model, messages, **kw):
        prompt = messages[-1]["content"]
        if "Answer yes or no" in prompt:
            return LLMResponse(text="yes")
        seen["answer_prompt"] = prompt
        return LLMResponse(text="ok")

    monkeypatch.setattr(llm_mod, "call_llm", capture_llm)
    runner.run([inst], tmp_path / "run", model="stub", judge_model="stub",
               k=10, stuff=True, use_embedder=False, flat_context=True,
               limit=None, verbose=False)
    ctx = seen["answer_prompt"]
    assert "[dated 2026-02-01]" in ctx and "[#1" not in ctx
    assert ctx.count("filler chatter") == 400          # body uncut in flat mode


def test_perception_flags_recorded_in_config(tmp_path, monkeypatch):
    _install_stub(monkeypatch)
    doc = runner.run([_instance()], tmp_path / "run", model="stub", judge_model="stub",
                     k=10, stuff=True, use_embedder=False, flat_context=True,
                     facts_last=True, limit=None, verbose=False)
    assert doc["config"]["context"] == "flat"
    assert doc["config"]["facts_position"] == "last"
    doc = runner.run([_instance()], tmp_path / "run2", model="stub", judge_model="stub",
                     k=10, stuff=True, use_embedder=False, limit=None, verbose=False)
    assert doc["config"]["context"] == "windowed"
    assert doc["config"]["facts_position"] == "first"


def test_reuse_vaults_skips_distiller_and_rebuilds_index(tmp_path, monkeypatch):
    # Frozen-corpus methodology: --distill re-rolls every note per run (LLM
    # non-determinism), which confounded the run-A/run-B comparison. With
    # reuse=True an existing vault is adopted as-is: no distiller call, notes
    # untouched, and the {rel: {session_id, date}} map rebuilt from frontmatter.
    import silica.kernel.prep_delegation as prep

    calls = []

    def first_pass(payload, target, **kw):
        calls.append(1)
        return {"updates": [{"snippet": "Fact: the dog is Zephyr."}]}

    monkeypatch.setattr(prep, "run_distiller", first_pass)
    vault = tmp_path / "v"
    runner.bind_vault(vault)
    idx1 = runner.load_question_vault(vault, _instance(), distill=True)
    assert calls  # first pass distilled

    def boom(*a, **kw):
        raise AssertionError("distiller called on reuse")

    monkeypatch.setattr(prep, "run_distiller", boom)
    runner.bind_vault(vault)
    idx2 = runner.load_question_vault(vault, _instance(), distill=True, reuse=True)
    assert idx2 == idx1  # session_id/date map survives the round-trip
    note = (vault / "sessions" / "s0001.md").read_text(encoding="utf-8")
    assert "Fact: the dog is Zephyr." in note  # first run's distillation frozen


def test_gold_in_context_direct_token_and_undefined():
    assert runner._gold_in_context("the suburbs",
                                   "she moved to the suburbs last month") is True
    # relaxed hit: every content token present even though the sentence is not
    assert runner._gold_in_context(
        "The Plesiosaur had a blue scaly body.",
        "the image shows a plesiosaur whose body is scaly and blue") is True
    assert runner._gold_in_context("3", "any context at all") is None  # derived gold
    assert runner._gold_in_context(3, "any context at all") is None    # JSON number gold
    assert runner._gold_in_context("Business Administration", "unrelated text") is False


def test_gold_in_context_recorded_per_question(tmp_path, monkeypatch):
    _install_stub(monkeypatch)
    doc = runner.run([_instance()], tmp_path / "run", model="stub", judge_model="stub",
                     k=10, stuff=True, use_embedder=False, limit=None, verbose=False)
    assert doc["questions"][0]["gold_in_context"] is True   # "Zephyr" is in s1
    assert doc["metrics"]["gold_in_context_mean"] == 1.0


def test_personal_memory_block_precedes_sessions(tmp_path, monkeypatch):
    # Layout fix (post-mortem 2026-07-14): the facts block is the densest
    # evidence and must open the context, not trail 10 long session notes.
    import silica.kernel.prep_delegation as prep

    monkeypatch.setattr(prep, "run_distiller", _ephemeral_distiller({
        "s1": [{"key": "user.dog.name", "text": "My dog is named Zephyr"}]}))
    seen = {}

    def capture_llm(model, messages, **kw):
        prompt = messages[-1]["content"]
        if "Answer yes or no" in prompt:
            return LLMResponse(text="yes")
        seen["answer_prompt"] = prompt
        seen["system"] = messages[0]["content"]
        return LLMResponse(text="Zephyr")

    monkeypatch.setattr(llm_mod, "call_llm", capture_llm)
    runner.run([_instance()], tmp_path / "run", model="stub", judge_model="stub",
               k=10, stuff=True, use_embedder=False, distill=True,
               limit=None, verbose=False)
    p = seen["answer_prompt"]
    assert p.index("Personal memory:") < p.index("[#1")   # facts precede ranked sessions
    # The answer model is told the block is reliable memory, not an afterthought.
    assert "Personal memory" in seen["system"]


_META = "A user-generated prompt and assistant response creating a children's book outline."


def test_distill_meta_summary_falls_back_to_verbatim(tmp_path, monkeypatch):
    # A body that DESCRIBES the conversation instead of carrying its facts is
    # distill-loss (post-mortem: Plesiosaur/placeholder notes) — keep verbatim.
    import silica.kernel.prep_delegation as prep

    monkeypatch.setattr(prep, "run_distiller",
                        lambda payload, target, **kw: {"updates": [{"snippet": _META}]})
    vault = tmp_path / "v"
    runner.bind_vault(vault)
    runner.load_question_vault(vault, _instance(), distill=True)
    note = (vault / "sessions" / "s0001.md").read_text(encoding="utf-8")
    assert "My dog Zephyr turned three today." in note
    assert "user-generated prompt" not in note.lower()


def test_distill_drops_meta_snippet_keeps_factual(tmp_path, monkeypatch):
    import silica.kernel.prep_delegation as prep

    monkeypatch.setattr(prep, "run_distiller", lambda payload, target, **kw: {
        "updates": [{"snippet": _META}, {"snippet": "Fact: the dog is Zephyr."}]})
    vault = tmp_path / "v"
    runner.bind_vault(vault)
    runner.load_question_vault(vault, _instance(), distill=True)
    note = (vault / "sessions" / "s0001.md").read_text(encoding="utf-8")
    assert "Fact: the dog is Zephyr." in note
    assert "user-generated prompt" not in note.lower()
    assert "My dog Zephyr turned three today." not in note  # no fallback needed


def test_episodic_ttl_off_by_default_opt_in_drops_old_facts(tmp_path, monkeypatch):
    import silica.kernel.prep_delegation as prep

    # Ephemeral from s0, dated 2026-01-01 — 120 days before question_date.
    monkeypatch.setattr(prep, "run_distiller", _ephemeral_distiller({
        "s0": [{"key": "user.dog.nickname", "text": "The dog goes by Zeph"}],
    }))
    prompts = []

    def capture_llm(model, messages, **kw):
        prompt = messages[-1]["content"]
        if "Answer yes or no" in prompt:
            return LLMResponse(text="yes")
        prompts.append(prompt)
        return LLMResponse(text="ok")

    monkeypatch.setattr(llm_mod, "call_llm", capture_llm)
    common = dict(model="stub", judge_model="stub", k=10, stuff=True,
                  use_embedder=False, distill=True, limit=None, verbose=False)

    doc = runner.run([_instance()], tmp_path / "r1", **common)
    assert "The dog goes by Zeph" in prompts[-1]      # default: TTL off, fact kept
    assert doc["questions"][0]["ephemeral_hit"] is False  # s0 is not a gold session

    doc = runner.run([_instance()], tmp_path / "r2", episodic_ttl=90, **common)
    assert "The dog goes by Zeph" not in prompts[-1]  # ablation: 120d-old fact dropped
    assert doc["config"]["episodic_ttl"] == 90


def test_judge_prompt_selects_per_type_rubric():
    assert "rubric" in runner._judge_instruction("single-session-preference").lower()
    assert "off-by-one" in runner._judge_instruction("temporal-reasoning")
    assert "updated answer" in runner._judge_instruction("knowledge-update")
    assert runner._judge_instruction("multi-session") == runner._JUDGE_BASE


def test_judge_rubric_uses_abstention_template_for_abs():
    # An _abs question must be graded on whether the model correctly abstained,
    # not on whether it echoes the gold string (LongMemEval evaluate_qa.py).
    base = runner._judge_instruction("single-session-user")
    abstain = runner._judge_instruction("single-session-user", is_abs=True)
    assert abstain != base
    assert "unanswerable" in abstain.lower()


def test_judge_retries_empty_reply_before_none(monkeypatch):
    # Empty completions are mostly transient provider drops (HTTP-200-no-text);
    # a verdict must survive one, and only repeated emptiness is a judge failure.
    import types as _types

    calls = []

    def fake(model, messages, **kw):
        calls.append(1)
        return _types.SimpleNamespace(text="" if len(calls) == 1 else "yes")

    monkeypatch.setattr("time.sleep", lambda s: None)  # backoff is not the test
    monkeypatch.setattr("silica.agent.llm.call_llm", fake)
    assert runner.judge("m", "multi-session", "q", "gold", "resp") is True
    assert len(calls) == 2

    calls.clear()
    monkeypatch.setattr("silica.agent.llm.call_llm",
                        lambda *a, **k: _types.SimpleNamespace(text=""))
    assert runner.judge("m", "multi-session", "q", "gold", "resp") is None


def test_pipeline_routes_abs_question_to_abstention_rubric(tmp_path, monkeypatch):
    # The abstention rubric must actually reach the judge: run_instance threads
    # is_abs, not just qtype.
    seen = {}

    def capture(model, messages, **kw):
        prompt = messages[-1]["content"]
        if "Answer yes or no" in prompt:          # judge turn
            seen["judge_prompt"] = prompt
            return LLMResponse(text="yes")
        return LLMResponse(text="I don't have that information.")  # answer turn

    monkeypatch.setattr(llm_mod, "call_llm", capture)
    runner.run([_abs_instance()], tmp_path / "run", model="stub", judge_model="stub",
               k=10, stuff=True, use_embedder=False, limit=None, verbose=False)
    assert "unanswerable" in seen["judge_prompt"].lower()


def test_incremental_checkpoint_written_after_every_question(tmp_path, monkeypatch):
    # A killed/hung run must keep every row scored so far: the metrics doc is
    # rewritten to `out` after each question, marked partial; the returned
    # (final) doc carries no partial marker.
    _install_stub(monkeypatch)
    out = tmp_path / "metrics.json"

    partials_seen_before_next_question = []
    orig = runner.run_instance

    def spy(inst, *a, **kw):
        if out.exists():
            partials_seen_before_next_question.append(
                json.loads(out.read_text())["partial"])
        return orig(inst, *a, **kw)

    monkeypatch.setattr(runner, "run_instance", spy)
    doc = runner.run([_instance(), _abs_instance()], tmp_path / "run",
                     model="stub", judge_model="stub", k=10, stuff=True,
                     use_embedder=False, limit=None, verbose=False, out=out)

    # Question 1's checkpoint was on disk before question 2 started.
    assert partials_seen_before_next_question == ["1/2"]
    on_disk = json.loads(out.read_text())
    assert on_disk["partial"] == "2/2"
    assert len(on_disk["questions"]) == 2
    assert on_disk["metrics"]["overall_accuracy"] is not None
    assert "partial" not in doc


def test_distill_passes_episodic_key_vocabulary(tmp_path, monkeypatch):
    import silica.kernel.paths as paths_mod
    from silica.kernel import prep_delegation as prep
    from evals.longmemeval import runner

    monkeypatch.setattr(paths_mod, "_SILICA_HOME", tmp_path / "silica_home")
    runner.bind_vault(tmp_path / "vault")
    from silica.kernel.episodic import EpisodicStore

    EpisodicStore().capture([{"key": "user.car.model", "text": "Panda"}],
                            run_id="s0", seen="2026-01-01")

    seen_kwargs: dict = {}

    def fake_distiller(payload, target, **kw):
        seen_kwargs.update(kw)
        return {"updates": []}

    monkeypatch.setattr(prep, "run_distiller", fake_distiller)
    runner.distill_session("s1", "2026-01-02", "User: hello")
    substrate = seen_kwargs.get("substrate") or ""
    assert "## Episodic keys" in substrate
    assert "user.car.model" in substrate
