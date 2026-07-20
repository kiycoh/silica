# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Answer-time perception (silica/kernel/perception.py).

perceive() is the single assembly of recalled memory into model context —
the LongMemEval harness consumes this same function, so these tests cover the
product behavior the eval numbers are attributed to: per-note query-densest
window, rank/evidence/date headers, facts-first episodic block, degraded legs.
All offline: co-occurrence retrieval only, no embedder, no reranker.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _bind(vault: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point CONFIG/DRIVER at a fresh fs vault; singletons reset per test."""
    import silica.driver
    import silica.kernel.cooccurrence as cooc_mod
    import silica.kernel.embed as embed_mod
    from silica.config import CONFIG

    vault.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(CONFIG, "vault_path", str(vault))
    monkeypatch.setattr(CONFIG, "memory_vault", str(vault))  # coincident: lane abstains
    monkeypatch.setattr(CONFIG, "backend", "fs")
    monkeypatch.setattr(silica.driver, "_driver", None)
    embed_mod.clear()
    cooc_mod.clear()


def _write(rel: str, date: str, body: str) -> None:
    from silica.driver import DRIVER

    DRIVER.create(rel, f'---\ndate: "{date}"\n---\n\n{body}\n')


def _index() -> None:
    from silica.tools.graph import silica_cooccurrence_refresh

    silica_cooccurrence_refresh(force=True)


LONG_BODY = ("filler chatter " * 400) + "the yoga class is on Tuesday evening " \
            + ("more filler " * 400)


def test_best_window_is_public():
    # The harness used to import the private name; the seam is public now.
    from silica.kernel.rerank import _best_window, best_window

    assert best_window is _best_window


def test_perceive_windows_bodies_under_rank_evidence_date_headers(tmp_path, monkeypatch):
    _bind(tmp_path / "v", monkeypatch)
    _write("sessions/a.md", "2026-01-01", LONG_BODY)
    _write("sessions/b.md", "2026-02-02", "short note about cooking pasta")
    _index()
    from silica.kernel.perception import perceive

    p = perceive("when is my yoga class?", now="2026-05-01", k=2,
                 window_chars=200, use_embedder=False)
    assert p.blocks, "cooccur leg should retrieve the yoga note"
    top = p.blocks[0]
    assert top.path == "sessions/a"
    assert "yoga class is on Tuesday" in top.excerpt      # query-densest window
    assert len(top.excerpt) <= 200                        # the wall was cut
    assert "date:" not in top.excerpt                     # frontmatter stripped
    assert top.evidence                                   # per-leg provenance survives

    ctx = p.render()
    assert f"[#1 | {top.evidence} | dated 2026-01-01]" in ctx
    assert "yoga class is on Tuesday" in ctx


def test_render_flat_returns_whole_bodies_without_rank_headers(tmp_path, monkeypatch):
    _bind(tmp_path / "v", monkeypatch)
    _write("sessions/a.md", "2026-01-01", LONG_BODY)
    _index()
    from silica.kernel.perception import perceive

    p = perceive("when is my yoga class?", now="2026-05-01", k=1,
                 window_chars=200, use_embedder=False)
    flat = p.render(windowed=False)
    assert "[dated 2026-01-01]" in flat
    assert "[#1" not in flat
    assert "filler chatter" in flat and "more filler" in flat  # body uncut


# --- multi-window perception (multi-window spec 2026-07-15) -----------------

# Gold tokens (Tuesday/Thursday) sit BEFORE the query terms: on density ties the
# earliest window wins, so terms-first phrasing would cut the trailing gold — the
# adjacency risk the spec's arm A/B comparison measures, not this unit's concern.
TWO_FACT_BODY = ("filler chatter " * 40) + "on Tuesday evening we go to the yoga class " \
                + ("filler chatter " * 40) + "on Thursday evening we moved the yoga class " \
                + ("filler chatter " * 40)


def test_perceive_multi_window_excerpt_joins_with_elision_marker(tmp_path, monkeypatch):
    _bind(tmp_path / "v", monkeypatch)
    _write("sessions/a.md", "2026-01-01", TWO_FACT_BODY)
    _index()
    from silica.kernel.perception import perceive

    p = perceive("when is my yoga class?", now="2026-05-01", k=1,
                 window_chars=150, windows=2, use_embedder=False)
    ex = p.blocks[0].excerpt
    assert "\n[…]\n" in ex
    assert "Tuesday" in ex and "Thursday" in ex
    assert ex.index("Tuesday") < ex.index("Thursday")  # document order survives
    assert len(ex) <= 2 * 150 + len("\n[…]\n")


def test_perceive_default_single_window_is_unchanged(tmp_path, monkeypatch):
    # windows=1 (the default) must keep the prompt surface byte-identical:
    # no marker, excerpt == best_window of the body.
    _bind(tmp_path / "v", monkeypatch)
    _write("sessions/a.md", "2026-01-01", TWO_FACT_BODY)
    _index()
    from silica.kernel.perception import perceive
    from silica.kernel.rerank import best_window

    p = perceive("when is my yoga class?", now="2026-05-01", k=1,
                 window_chars=150, use_embedder=False)
    b = p.blocks[0]
    assert "[…]" not in b.excerpt
    assert b.excerpt == best_window(b.body, "when is my yoga class?", 150)


def test_perceive_multi_window_short_body_passes_whole(tmp_path, monkeypatch):
    _bind(tmp_path / "v", monkeypatch)
    _write("sessions/a.md", "2026-01-01", "short note about the yoga class")
    _index()
    from silica.kernel.perception import perceive

    p = perceive("yoga class?", now="2026-05-01", k=1,
                 window_chars=200, windows=2, use_embedder=False)
    b = p.blocks[0]
    assert b.excerpt == b.body
    assert "[…]" not in b.excerpt


def _seed_fact(key="user.dog.name", text="My dog is named Zephyr",
               run_id="s1", seen="2026-01-01") -> None:
    from silica.kernel.episodic import EpisodicStore

    EpisodicStore().capture([{"key": key, "text": text}], run_id=run_id, seen=seen)


def test_facts_block_first_by_default_last_on_request(tmp_path, monkeypatch):
    _bind(tmp_path / "v", monkeypatch)
    _write("sessions/a.md", "2026-01-01", "we talked about my dog at the park")
    _index()
    _seed_fact()
    from silica.kernel.perception import perceive

    p = perceive("What is my dog's name?", now="2026-05-01", k=5,
                 use_embedder=False, episodic_ttl_days=0)
    assert p.facts_block.startswith("Personal memory:")
    assert "Zephyr" in p.facts_block
    assert p.fact_chains and p.fact_chains[0][0].runs == ["s1"]  # telemetry chain

    ctx = p.render()
    assert ctx.index("Personal memory:") < ctx.index("[#1")
    tail = p.render(facts_first=False)
    assert tail.index("[#1") < tail.index("Personal memory:")


def test_empty_episodic_store_yields_no_facts_block(tmp_path, monkeypatch):
    _bind(tmp_path / "v", monkeypatch)
    _write("sessions/a.md", "2026-01-01", "we talked about my dog at the park")
    _index()
    from silica.kernel.perception import perceive

    p = perceive("What is my dog's name?", now="2026-05-01", k=5, use_embedder=False)
    assert p.facts_block == ""
    assert "Personal memory" not in p.render()


def test_with_facts_false_skips_episodic_recall(tmp_path, monkeypatch):
    _bind(tmp_path / "v", monkeypatch)
    _write("sessions/a.md", "2026-01-01", "we talked about my dog at the park")
    _index()
    _seed_fact()
    from silica.kernel.perception import perceive

    p = perceive("What is my dog's name?", now="2026-05-01", k=5,
                 use_embedder=False, with_facts=False)
    assert p.facts_block == "" and not p.fact_hits


def test_paths_override_skips_retrieval_keeps_order(tmp_path, monkeypatch):
    # --stuff arm: assemble the given notes in order, no index needed at all.
    _bind(tmp_path / "v", monkeypatch)
    _write("sessions/a.md", "2026-01-01", "alpha body")
    _write("sessions/b.md", "2026-02-02", "beta body")
    from silica.kernel.perception import perceive

    p = perceive("anything", now="2026-05-01", use_embedder=False,
                 paths=["sessions/b", "sessions/a"])
    assert [b.path for b in p.blocks] == ["sessions/b", "sessions/a"]
    assert all(b.evidence == "" for b in p.blocks)
    ctx = p.render()
    assert "[#1 | dated 2026-02-02]" in ctx   # no evidence segment, no double pipe
    assert "[#2 | dated 2026-01-01]" in ctx


def test_note_without_frontmatter_does_not_crash_perceive(tmp_path, monkeypatch):
    """A body-only note (no frontmatter) must assemble cleanly. Product notes
    written by the FSM write path can lack frontmatter; frontmatter.split then
    returns data=None and _read_dated_body used to crash on data.get (found by
    the LoCoMo e2e leg: perceive died mid-run at question 173/199)."""
    _bind(tmp_path / "v", monkeypatch)
    from silica.driver import DRIVER
    DRIVER.create("memory/plain.md", "just a body, no frontmatter at all\n")
    from silica.kernel.perception import perceive

    p = perceive("anything", now="2026-05-01", use_embedder=False,
                 paths=["memory/plain"])
    assert [b.path for b in p.blocks] == ["memory/plain"]
    ctx = p.render()
    assert "just a body" in ctx
    assert "[#1]" in ctx           # no date segment, no crash


def test_unreadable_paths_are_skipped_rank_stays_dense(tmp_path, monkeypatch):
    _bind(tmp_path / "v", monkeypatch)
    _write("sessions/a.md", "2026-01-01", "alpha body")
    from silica.kernel.perception import perceive

    p = perceive("anything", now="2026-05-01", use_embedder=False,
                 paths=["missing/nope", "sessions/a"])
    assert [b.path for b in p.blocks] == ["sessions/a"]
    assert "[#1 | dated 2026-01-01]" in p.render()


def test_silica_recall_tool_returns_context_and_paths(tmp_path, monkeypatch):
    _bind(tmp_path / "v", monkeypatch)
    _write("sessions/a.md", "2026-01-01", LONG_BODY)
    _index()
    from silica.tools.graph import silica_recall

    out = silica_recall(query="when is my yoga class?", k=5)
    assert "yoga class is on Tuesday" in out["context"]
    assert out["notes"] == ["sessions/a"]
    assert out["facts"] == 0


def test_use_recall_weights_false_ignores_populated_store(tmp_path, monkeypatch):
    """Default off: a populated recall_weights store must not change output."""
    _bind(tmp_path / "v", monkeypatch)
    _write("sessions/a.md", "2026-01-01", "short note about cooking pasta")
    _write("sessions/b.md", "2026-02-02", "short note about hiking trails")
    _index()
    from silica.kernel import recall_weights
    from silica.kernel.perception import perceive

    recall_weights.bump(["sessions/b"])  # store populated, flag stays off
    p = perceive("pasta", now="2026-05-01", k=2, use_embedder=False)
    assert not any("recall:" in b.evidence for b in p.blocks)


def test_use_recall_weights_true_resurfaces_bumped_note(tmp_path, monkeypatch):
    """Flag on: a bumped note surfaces with recall: evidence via the RRF leg."""
    _bind(tmp_path / "v", monkeypatch)
    _write("sessions/a.md", "2026-01-01", "short note about cooking pasta")
    _write("sessions/b.md", "2026-02-02", "short note about hiking trails")
    _index()
    from silica.kernel import recall_weights
    from silica.kernel.perception import perceive

    recall_weights.bump(["sessions/b"])
    p = perceive("pasta", now="2026-05-01", k=2, use_embedder=False,
                 use_recall_weights=True)
    assert any(b.path == "sessions/b" and "recall:" in b.evidence for b in p.blocks)
