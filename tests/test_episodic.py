# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Episodic memory lane — short-term fact store with supersedes chains and TTL
(docs spec 2026-07-14). Store unit tests use an explicit path; no global state."""
from __future__ import annotations

import json

from silica.kernel.episodic import EpisodicStore, Fact


def _store(tmp_path):
    return EpisodicStore(path=tmp_path / "episodic.json")


def test_capture_new_key_persists_round_trip(tmp_path):
    store = _store(tmp_path)
    store.capture(
        [{"key": "user.dog.name", "text": "Il mio cane si chiama Tom"}],
        run_id="run_a3f2",
        seen="2026-07-14",
    )

    reloaded = EpisodicStore(path=tmp_path / "episodic.json")
    facts = reloaded.live_facts()
    assert len(facts) == 1
    f = facts[0]
    assert f.key == "user.dog.name"
    assert f.text == "Il mio cane si chiama Tom"
    assert f.first_seen == "2026-07-14"
    assert f.last_seen == "2026-07-14"
    assert f.runs == ["run_a3f2"]
    assert f.supersedes is None
    assert f.status == "live"


def test_reinforce_same_key_same_normalized_text(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Tom"}],
                  run_id="run_1", seen="2026-06-10")
    # Same fact, different casing/punctuation — reinforces, no new fact.
    store.capture([{"key": "user.dog.name", "text": "il mio cane si chiama tom!"}],
                  run_id="run_2", seen="2026-07-01")
    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Tom"}],
                  run_id="run_2", seen="2026-07-02")

    facts = store.live_facts()
    assert len(facts) == 1
    f = facts[0]
    assert f.first_seen == "2026-06-10"
    assert f.last_seen == "2026-07-02"
    assert f.runs == ["run_1", "run_2"]  # run_2 appended once


def test_supersede_same_key_different_text_keeps_chain(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Rex"}],
                  run_id="run_1", seen="2026-03-01")
    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Tom"}],
                  run_id="run_2", seen="2026-06-10")

    live = store.live_facts()
    assert len(live) == 1
    head = live[0]
    assert head.text == "Il mio cane si chiama Tom"
    assert head.first_seen == "2026-06-10"

    old = next(f for f in store.facts if f.id == head.supersedes)
    assert old.text == "Il mio cane si chiama Rex"
    assert old.status == "superseded"
    assert old.supersedes is None

    # Chain grows: a third value points at the second.
    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Ugo"}],
                  run_id="run_3", seen="2026-07-01")
    (head2,) = store.live_facts()
    assert head2.supersedes == head.id
    assert next(f for f in store.facts if f.id == head.id).status == "superseded"


class _FakeEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class _BrokenEmbedder:
    def embed(self, texts):
        raise RuntimeError("embedder down")


def test_capture_embeds_new_facts_when_embedder_served(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "user.dog.name", "text": "Tom"}],
                  run_id="r1", seen="2026-07-14", embedder=_FakeEmbedder())
    (f,) = store.live_facts()
    assert f.vec == [1.0, 0.0]


def test_capture_without_embedder_or_broken_embedder_skips_silently(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "a.b", "text": "x"}], run_id="r1", seen="2026-07-14")
    store.capture([{"key": "c.d", "text": "y"}],
                  run_id="r1", seen="2026-07-14", embedder=_BrokenEmbedder())
    assert all(f.vec is None for f in store.live_facts())


def test_episodic_home_resolves_even_when_active_vault_is_memory_vault(tmp_path, monkeypatch):
    """Unlike memory_lane.memory_vault(), episodic_home never abstains."""
    from silica.config import CONFIG
    from silica.kernel.episodic import episodic_home

    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(CONFIG, "memory_vault", str(vault), raising=False)
    monkeypatch.setattr(CONFIG, "vault_path", str(vault), raising=False)
    assert episodic_home() == vault.resolve()


def test_corrupt_store_file_is_quarantined_not_fatal(tmp_path):
    p = tmp_path / "episodic.json"
    p.write_text("{not json", encoding="utf-8")
    store = EpisodicStore(path=p)
    assert store.live_facts() == []
    # Original bytes preserved aside, store restarts empty.
    assert any(".corrupt." in q.name for q in tmp_path.iterdir())


def test_sweep_evaporates_whole_chain_by_head_last_seen(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "user.dog.name", "text": "Rex"}], run_id="r1", seen="2026-01-01")
    store.capture([{"key": "user.dog.name", "text": "Tom"}], run_id="r2", seen="2026-02-01")
    store.capture([{"key": "user.city", "text": "Torino"}], run_id="r2", seen="2026-07-01")

    removed = store.sweep(now="2026-07-14", ttl_days=90)
    # dog chain head last_seen 2026-02-01 is >90d old: head AND superseded
    # ancestor evaporate together; city (13d old) survives.
    assert removed == 1
    assert {f.key for f in store.facts} == {"user.city"}

    # Reloaded store reflects the sweep (sweep persists).
    assert {f.key for f in EpisodicStore(path=store.path).facts} == {"user.city"}


def test_sweep_reinforcement_resets_timer_and_zero_ttl_never_expires(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "user.dog.name", "text": "Tom"}], run_id="r1", seen="2026-01-01")
    store.capture([{"key": "user.dog.name", "text": "Tom"}], run_id="r2", seen="2026-07-10")
    assert store.sweep(now="2026-07-14", ttl_days=90) == 0
    assert len(store.live_facts()) == 1

    store.capture([{"key": "old.fact", "text": "x"}], run_id="r1", seen="2020-01-01")
    assert store.sweep(now="2026-07-14", ttl_days=0) == 0  # 0 = never expire
    assert len(store.live_facts()) == 2


def test_nucleation_candidates_count_distinct_runs_across_chain(tmp_path):
    store = _store(tmp_path)
    # user.dog.name: 3 distinct runs spread over a supersede (Rex r1+r2, Tom r3)
    store.capture([{"key": "user.dog.name", "text": "Rex"}], run_id="r1", seen="2026-06-10")
    store.capture([{"key": "user.dog.name", "text": "Rex"}], run_id="r2", seen="2026-06-20")
    store.capture([{"key": "user.dog.name", "text": "Tom"}], run_id="r3", seen="2026-07-01")
    # user.city: 2 runs only — below threshold
    store.capture([{"key": "user.city", "text": "Torino"}], run_id="r1", seen="2026-06-10")
    store.capture([{"key": "user.city", "text": "Torino"}], run_id="r2", seen="2026-06-20")

    cands = store.nucleation_candidates(min_runs=3)
    assert len(cands) == 1
    c = cands[0]
    assert c.key == "user.dog.name"
    assert c.run_count == 3
    assert c.since == "2026-06-10"


def test_recall_ranks_by_embedding_when_vectors_exist(tmp_path):
    store = _store(tmp_path)

    class _E:
        def embed(self, texts):
            return [[1.0, 0.1] if "cane" in t else [0.1, 1.0] for t in texts]

    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Tom"},
                   {"key": "user.city", "text": "Vivo a Torino"}],
                  run_id="r1", seen="2026-07-14", embedder=_E())

    hits = store.recall("come si chiama il mio cane", query_vec=[1.0, 0.0],
                        k=2, now="2026-07-14")
    assert [h.fact.key for h in hits] == ["user.dog.name", "user.city"]
    assert hits[0].score > hits[1].score


def test_recall_lexical_fallback_without_vectors_and_live_only(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Rex"}],
                  run_id="r1", seen="2026-07-01")
    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Tom"}],
                  run_id="r2", seen="2026-07-10")
    store.capture([{"key": "user.meeting", "text": "Riunione lunedì"}],
                  run_id="r2", seen="2026-07-10")

    hits = store.recall("cane", query_vec=None, k=5, now="2026-07-14")
    # Only the live head of the dog chain matches; superseded Rex never
    # surfaces as its own hit even though its text also matches.
    assert [h.fact.text for h in hits][0] == "Il mio cane si chiama Tom"
    assert all(h.fact.status == "live" for h in hits)
    # Key segments count as lexical signal too.
    hits_by_key = store.recall("dog name", query_vec=None, k=5, now="2026-07-14")
    assert hits_by_key and hits_by_key[0].fact.key == "user.dog.name"


def test_recall_filters_expired_chains_without_mutating_store(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "user.old", "text": "vecchio fatto sul cane"}],
                  run_id="r1", seen="2026-01-01")
    hits = store.recall("cane", query_vec=None, k=5, now="2026-07-14", ttl_days=90)
    assert hits == []
    # Recall never deletes: sweep at digest time is the only deleter.
    assert len(store.facts) == 1


def test_render_includes_chain_history_with_dates(tmp_path):
    from silica.kernel import episodic

    store = _store(tmp_path)
    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Rex"}],
                  run_id="r1", seen="2026-03-01")
    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Tom"}],
                  run_id="r2", seen="2026-06-10")
    store.capture([{"key": "user.city", "text": "Vivo a Torino"}],
                  run_id="r2", seen="2026-06-10")

    hits = store.recall("cane Torino", query_vec=None, k=5, now="2026-07-14")
    text = episodic.render(hits, store=store)
    assert "- [since 2026-06-10] Il mio cane si chiama Tom" in text
    assert "(previously: Il mio cane si chiama Rex, 2026-03-01 to 2026-06-10)" in text
    assert "- [since 2026-06-10] Vivo a Torino" in text
    # No chain for the city fact — no "previously" line for it.
    assert text.count("previously") == 1


def test_render_empty_hits_is_empty_string(tmp_path):
    from silica.kernel import episodic

    assert episodic.render([], store=_store(tmp_path)) == ""


def test_distiller_output_parses_with_and_without_ephemerals():
    from silica.kernel.ops import DistillerOutput

    legacy = DistillerOutput.model_validate({"updates": []})
    assert legacy.ephemerals == []

    doc = DistillerOutput.model_validate({
        "updates": [],
        "ephemerals": [{"key": "user.dog.name", "text": "Il mio cane si chiama Tom"}],
    })
    assert doc.ephemerals[0].key == "user.dog.name"
    assert doc.ephemerals[0].text == "Il mio cane si chiama Tom"


def test_config_episodic_fields_env_overrides(monkeypatch):
    from silica.config import SilicaConfig

    assert SilicaConfig().episodic_ttl_days == 90
    assert SilicaConfig().episodic_nucleation_runs == 3
    monkeypatch.setenv("SILICA_EPISODIC_TTL_DAYS", "0")
    monkeypatch.setenv("SILICA_EPISODIC_NUCLEATION_RUNS", "5")
    cfg = SilicaConfig()
    assert cfg.episodic_ttl_days == 0
    assert cfg.episodic_nucleation_runs == 5


def test_capture_from_distill_routes_ephemerals_and_never_raises(tmp_path, monkeypatch):
    from silica.kernel import episodic

    monkeypatch.setattr(episodic, "store_path", lambda: tmp_path / "episodic.json")
    result = {
        "updates": [],
        "ephemerals": [{"key": "user.dog.name", "text": "Il mio cane si chiama Tom"},
                       {"key": "", "text": "junk ignored"}],
    }
    episodic.capture_from_distill(result, run_id="run_x", seen="2026-07-14")
    (f,) = EpisodicStore(path=tmp_path / "episodic.json").live_facts()
    assert f.key == "user.dog.name"
    assert f.runs == ["run_x"]

    # No ephemerals / broken store: silent no-op, ingest must never fail.
    episodic.capture_from_distill({"updates": []}, run_id="r", seen="2026-07-14")
    monkeypatch.setattr(episodic, "store_path",
                        lambda: (_ for _ in ()).throw(RuntimeError("disk gone")))
    episodic.capture_from_distill(result, run_id="r", seen="2026-07-14")


def test_digest_sweeps_and_lists_nucleation_candidates(tmp_path, monkeypatch):
    import datetime as _dt

    from silica.kernel import episodic
    from silica.kernel.progress import ProgressLedger

    monkeypatch.setattr(episodic, "store_path", lambda: tmp_path / "episodic.json")
    store = EpisodicStore(path=tmp_path / "episodic.json")
    today = _dt.date.today().isoformat()
    for rid in ("r1", "r2", "r3"):
        store.capture([{"key": "user.dog.name", "text": "Tom"}], run_id=rid, seen=today)
    store.capture([{"key": "user.stale", "text": "old"}], run_id="r0", seen="2020-01-01")

    text = ProgressLedger.new(mode="test").digest()
    assert ("episodic candidate: user.dog.name (3 runs since "
            f"{today}) -> consider promoting to a note") in text
    # Sweep ran: the 2020 chain evaporated from the persisted store.
    assert {f.key for f in EpisodicStore(path=tmp_path / "episodic.json").facts} == {"user.dog.name"}


def test_distiller_prompt_routes_ephemerals():
    from silica.kernel.prep_delegation import render_prompt

    prompt = render_prompt(target="Notes", source_text="some english text")
    assert '"ephemerals"' in prompt
    assert "user.dog.name" in prompt  # the canonical key example
    assert "entity.attribute" in prompt


def test_normalize_key_merges_morphological_variants():
    from silica.kernel.episodic import normalize_key

    assert normalize_key("model_kits.gifts") == normalize_key("model_kit.gift")
    assert normalize_key("User.Car.Model") == "user.car.model"
    assert normalize_key("user.cities") == normalize_key("user.city")
    # snowball, not naive strip-s: these survive intact
    assert normalize_key("user.status") == "user.status"
    assert normalize_key("user.address") == "user.address"
    # dots stay segment separators, underscores stay token separators
    assert normalize_key("model_kits.last_project") == "model_kit.last_project"


def test_normalize_key_idempotent():
    from silica.kernel.episodic import normalize_key

    for k in ("model_kits.gifts", "user.preferences.color", "user.cities",
              "assistant.recipe.oven_temp"):
        once = normalize_key(k)
        assert normalize_key(once) == once


def test_capture_links_chain_across_plural_key_variants(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "model_kit.project", "text": "Working on a Spitfire kit"}],
                  run_id="run_1", seen="2026-06-01")
    store.capture([{"key": "model_kits.projects", "text": "Now building a B-29 kit"}],
                  run_id="run_2", seen="2026-06-20")

    live = store.live_facts()
    assert len(live) == 1
    head = live[0]
    assert head.key == "model_kits.projects"   # raw key stored as emitted
    assert head.text == "Now building a B-29 kit"
    old = next(f for f in store.facts if f.id == head.supersedes)
    assert old.key == "model_kit.project" and old.status == "superseded"


def test_capture_reinforces_across_key_variants_when_text_matches(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "model_kit.project", "text": "Building a Spitfire"}],
                  run_id="run_1", seen="2026-06-01")
    store.capture([{"key": "model_kits.project", "text": "building a spitfire!"}],
                  run_id="run_2", seen="2026-06-10")

    live = store.live_facts()
    assert len(live) == 1
    assert live[0].key == "model_kit.project"   # first spelling kept
    assert live[0].runs == ["run_1", "run_2"]
    assert live[0].last_seen == "2026-06-10"


def test_capture_matches_legacy_head_written_before_layer_a(tmp_path):
    # A store written before normalization existed: raw plural head on disk.
    store = _store(tmp_path)
    store.facts.append(Fact(id="f_0001", key="model_kits.gifts",
                            text="Got a B-29 kit", first_seen="2026-05-01",
                            last_seen="2026-05-01", runs=["run_0"]))
    store.next_id = 2
    store.save()

    store.capture([{"key": "model_kit.gifts", "text": "Got a Camaro kit"}],
                  run_id="run_1", seen="2026-06-01")
    live = store.live_facts()
    assert len(live) == 1
    assert live[0].supersedes == "f_0001"


def test_key_vocabulary_lists_live_heads_by_recency_with_cap(tmp_path):
    from silica.kernel.episodic import key_vocabulary

    store = _store(tmp_path)
    store.capture([{"key": "user.dog.name", "text": "Tom"}],
                  run_id="r1", seen="2026-01-01")
    store.capture([{"key": "user.car.model", "text": "Panda"}],
                  run_id="r2", seen="2026-03-01")
    # Supersede user.dog.name: only the head key surfaces, once.
    store.capture([{"key": "user.dog.name", "text": "Rex"}],
                  run_id="r3", seen="2026-04-01")

    assert key_vocabulary(store) == ["user.dog.name", "user.car.model"]
    assert key_vocabulary(store, cap=1) == ["user.dog.name"]


def test_key_vocabulary_section_renders_or_abstains(tmp_path):
    from silica.kernel.episodic import key_vocabulary_section

    store = _store(tmp_path)
    assert key_vocabulary_section(store) is None   # empty store: no section

    store.capture([{"key": "user.car.model", "text": "Panda"}],
                  run_id="r1", seen="2026-01-01")
    section = key_vocabulary_section(store)
    assert section is not None
    assert section.startswith("## Episodic keys")
    assert "user.car.model" in section


def test_key_tokens_stemmed_entity_prefix_dropped():
    from silica.kernel.episodic import key_tokens

    # The probe-proven KU pair shares exactly two stemmed tokens.
    shared = (key_tokens("user.fitness.tournament.date")
              & key_tokens("user.tennis_tournament_date"))
    assert len(shared) == 2
    # Entity prefixes are dropped, never tokens.
    assert not key_tokens("user.laundry.schedule") & {"user", "assist"}
    # Morphological variants and user/assistant prefixes merge.
    assert key_tokens("assistant.laundry.tips") == key_tokens("user.laundry.tip")
    # Single-char tokens are noise, not alphabet.
    assert key_tokens("user.a_b.c") == set()


# ---------------------------------------------------------------------------
# Key schema enforcement (ADR-0021): declared grammar, fold never reject
# ---------------------------------------------------------------------------


def _schema(**kw):
    from silica.kernel.vault_manifest import EpisodicKeySchema

    return EpisodicKeySchema(**kw)


def test_enforce_compliant_key_passes_through():
    from silica.kernel.episodic import enforce_key_schema

    assert enforce_key_schema("user.dog.name", _schema()) == "user.dog.name"


def test_enforce_unknown_prefix_folds_under_default():
    from silica.kernel.episodic import enforce_key_schema

    assert enforce_key_schema("dog.name", _schema()) == "user.dog.name"


def test_enforce_depth_folds_tail_segments():
    from silica.kernel.episodic import enforce_key_schema

    assert (enforce_key_schema("assistant.recipe.oven.temp", _schema())
            == "assistant.recipe.oven_temp")


def test_enforce_prefix_match_is_canonical_stored_form_untouched():
    from silica.kernel.episodic import enforce_key_schema

    # "assistants" stems to the same canonical form as "assistant": it IS a
    # known prefix, and the stored spelling is never rewritten.
    assert enforce_key_schema("assistants.diet", _schema()) == "assistants.diet"


def test_enforce_prepend_then_depth_fold_compose():
    from silica.kernel.episodic import enforce_key_schema

    # Unknown prefix adds a segment, so the tail folds to honor max_depth.
    assert (enforce_key_schema("weather.city.today", _schema())
            == "user.weather.city_today")


def test_enforce_is_idempotent():
    from silica.kernel.episodic import enforce_key_schema

    schema = _schema()
    for raw in ("dog.name", "assistant.recipe.oven.temp", "weather.city.today",
                "user.dog.name"):
        once = enforce_key_schema(raw, schema)
        assert enforce_key_schema(once, schema) == once


def test_enforce_drops_empty_segments():
    from silica.kernel.episodic import enforce_key_schema

    assert enforce_key_schema("user..dog.name", _schema()) == "user.dog.name"


def test_enforce_honors_custom_schema_values():
    from silica.kernel.episodic import enforce_key_schema

    schema = _schema(prefixes=("team",), default_prefix="team", max_depth=2)
    assert enforce_key_schema("velocity.sprint", schema) == "team.velocity_sprint"
    assert enforce_key_schema("team.velocity", schema) == "team.velocity"


def test_capture_with_schema_stores_enforced_key(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "dog.name", "text": "Il mio cane si chiama Tom"}],
                  run_id="run_1", seen="2026-07-16", schema=_schema())
    facts = store.live_facts()
    assert len(facts) == 1
    assert facts[0].key == "user.dog.name"


def test_capture_without_schema_stores_raw_key(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "assistant.recipe.oven.temp", "text": "180 gradi"}],
                  run_id="run_1", seen="2026-07-16")
    assert store.live_facts()[0].key == "assistant.recipe.oven.temp"


def test_capture_from_distill_loads_schema_from_memory_vault(tmp_path, monkeypatch):
    from silica.kernel import episodic

    vault = tmp_path / "memvault"
    vault.mkdir()
    (vault / "vault.yaml").write_text(
        "conventions:\n  episodic_keys: {}\n", encoding="utf-8")
    monkeypatch.setattr(episodic, "episodic_home", lambda: vault)
    monkeypatch.setattr(episodic, "store_path", lambda: tmp_path / "episodic.json")

    result = {"ephemerals": [{"key": "dog.name", "text": "Si chiama Tom"}]}
    episodic.capture_from_distill(result, run_id="run_1", seen="2026-07-16")
    (f,) = EpisodicStore(path=tmp_path / "episodic.json").live_facts()
    assert f.key == "user.dog.name"


def test_capture_from_distill_without_schema_block_keeps_raw_key(tmp_path, monkeypatch):
    from silica.kernel import episodic

    vault = tmp_path / "memvault"
    vault.mkdir()  # no vault.yaml at all
    monkeypatch.setattr(episodic, "episodic_home", lambda: vault)
    monkeypatch.setattr(episodic, "store_path", lambda: tmp_path / "episodic.json")

    result = {"ephemerals": [{"key": "dog.name", "text": "Si chiama Tom"}]}
    episodic.capture_from_distill(result, run_id="run_1", seen="2026-07-16")
    (f,) = EpisodicStore(path=tmp_path / "episodic.json").live_facts()
    assert f.key == "dog.name"


def test_capture_schema_converges_raw_variant_onto_existing_chain(tmp_path):
    store = _store(tmp_path)
    store.capture([{"key": "user.dog.name", "text": "Il mio cane si chiama Tom"}],
                  run_id="run_1", seen="2026-07-01", schema=_schema())
    # Same fact arrives with a prefixless key: enforcement folds it onto the
    # same chain instead of starting a parallel one.
    store.capture([{"key": "dog.name", "text": "Il mio cane si chiama Tom"}],
                  run_id="run_2", seen="2026-07-16", schema=_schema())
    facts = store.live_facts()
    assert len(facts) == 1
    assert facts[0].runs == ["run_1", "run_2"]

