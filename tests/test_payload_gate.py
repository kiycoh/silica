"""Recurrence-gated note creation — classify_action + build_payload wiring.

Spec (hermes-coherence §4): llm-wiki rule "create a page when a concept appears
in 2+ sources OR is central; never for passing mentions". Two signals, both
already computed upstream, gate the no-collision path of classify_action:
  - recurrence: cross-file (CROSSDEDUP merge count) or intra-file (match count);
  - structural centrality: concept sourced from author markup (_seed_structural
    → ConceptCandidate.confidence == "EXTRACTED" → recon "structural_concepts").

Knob: CONFIG.min_recurrence_for_create. Default 1 = gate inert = today's
behavior bit-identical. The gate NEVER touches collision verdicts.
"""
from __future__ import annotations

import pytest

from silica.config import CONFIG
from silica.kernel.payload import build_concept_entry, build_payload, classify_action


# ---------------------------------------------------------------------------
# classify_action — knob=1 (default): bit-identical to the pre-gate function
# ---------------------------------------------------------------------------

class TestClassifyActionDefaultKnob:
    def test_new_concept_creates(self):
        assert classify_action(None, True) == "create"

    def test_no_collision_not_new_skips(self):
        assert classify_action(None, False) == "skip"

    def test_title_match_enriches(self):
        assert classify_action({"best_match": "title", "total_hits": 1}, False) == "enrich"

    def test_three_hits_reviews(self):
        assert classify_action({"best_match": "body", "total_hits": 3}, False) == "review"

    def test_two_body_hits_likely_skip(self):
        assert classify_action({"best_match": "body", "total_hits": 2}, False) == "likely_skip"

    def test_gate_inert_at_default_knob_regardless_of_signals(self):
        """knob=1 → the recurrence/structural signals are never even consulted."""
        assert classify_action(
            None, True,
            recurrence_count=0, is_structural=False, min_recurrence_for_create=1,
        ) == "create"


# ---------------------------------------------------------------------------
# classify_action — knob=2: the gate
# ---------------------------------------------------------------------------

class TestClassifyActionGate:
    def test_single_mention_non_structural_gated_to_likely_skip(self):
        assert classify_action(
            None, True,
            recurrence_count=1, is_structural=False, min_recurrence_for_create=2,
        ) == "likely_skip"

    def test_structural_single_mention_still_creates(self):
        """Centrality (author markup) alone satisfies the llm-wiki rule."""
        assert classify_action(
            None, True,
            recurrence_count=1, is_structural=True, min_recurrence_for_create=2,
        ) == "create"

    def test_recurrent_concept_still_creates(self):
        assert classify_action(
            None, True,
            recurrence_count=2, is_structural=False, min_recurrence_for_create=2,
        ) == "create"

    @pytest.mark.parametrize("collision,expected", [
        (None, "skip"),
        ({"best_match": "title", "total_hits": 1}, "enrich"),
        ({"best_match": "body", "total_hits": 3}, "review"),
        ({"best_match": "body", "total_hits": 2}, "likely_skip"),
    ])
    def test_collision_paths_untouched_by_gate(self, collision, expected):
        """The gate lives strictly on the in_new_concepts path — collision tiers
        (and the τ_low/τ_high routing downstream of them) are out of scope."""
        assert classify_action(
            collision, False,
            recurrence_count=1, is_structural=False, min_recurrence_for_create=2,
        ) == expected


# ---------------------------------------------------------------------------
# build_payload wiring — signals flow from the recon report to classify_action
# ---------------------------------------------------------------------------

class _FakeDriver:
    def __init__(self, notes: dict[str, str]):
        self._notes = notes

    def read_note(self, name):
        from silica.driver.base import NoteContent, NoteRef
        if name not in self._notes:
            raise RuntimeError(f"not found: {name}")
        return NoteContent(ref=NoteRef(name=name, path=name), content=self._notes[name])


def _hints(payload: dict) -> dict[str, str]:
    return {
        c["name"]: c["action_hint"]
        for b in payload["batches"]
        for c in b["concepts"]
    }


@pytest.fixture
def fake_vault(monkeypatch):
    """One inbox note: 'entropia' twice (recurrent), 'retropropagazione' once."""
    import silica.kernel.payload as payload_mod
    body = (
        "L'entropia misura il disordine del sistema; l'entropia cresce sempre.\n\n"
        "La retropropagazione viene citata di passaggio una volta sola.\n"
    )
    monkeypatch.setattr(payload_mod, "DRIVER", _FakeDriver({"inbox/a.md": body}))
    return body


def _report(**over) -> dict:
    base = {
        "file": "inbox/a.md",
        "collisions": [],
        "new_concepts": ["entropia", "retropropagazione"],
    }
    base.update(over)
    return base


class TestBuildPayloadDefaultKnob:
    def test_knob_1_every_new_concept_creates(self, fake_vault, monkeypatch):
        monkeypatch.setattr(CONFIG, "min_recurrence_for_create", 1, raising=False)
        hints = _hints(build_payload([_report()], window=450))
        assert hints == {"entropia": "create", "retropropagazione": "create"}


class TestBuildPayloadGateKnob2:
    def test_single_intra_file_mention_gated(self, fake_vault, monkeypatch):
        monkeypatch.setattr(CONFIG, "min_recurrence_for_create", 2, raising=False)
        hints = _hints(build_payload([_report()], window=450))
        assert hints["retropropagazione"] == "likely_skip"   # 1 mention, not structural

    def test_intra_file_recurrence_creates(self, fake_vault, monkeypatch):
        monkeypatch.setattr(CONFIG, "min_recurrence_for_create", 2, raising=False)
        hints = _hints(build_payload([_report()], window=450))
        assert hints["entropia"] == "create"                 # 2 intra-file mentions

    def test_structural_concept_survives(self, fake_vault, monkeypatch):
        """recon's structural_concepts (from _seed_structural markup) bypasses the gate."""
        monkeypatch.setattr(CONFIG, "min_recurrence_for_create", 2, raising=False)
        report = _report(structural_concepts=["retropropagazione"])
        hints = _hints(build_payload([report], window=450))
        assert hints["retropropagazione"] == "create"

    def test_cross_file_recurrence_creates(self, fake_vault, monkeypatch):
        """CROSSDEDUP's concept_recurrence (merge count) bypasses the gate."""
        monkeypatch.setattr(CONFIG, "min_recurrence_for_create", 2, raising=False)
        report = _report(concept_recurrence={"retropropagazione": 2})
        hints = _hints(build_payload([report], window=450))
        assert hints["retropropagazione"] == "create"

    def test_collision_entries_untouched(self, fake_vault, monkeypatch):
        monkeypatch.setattr(CONFIG, "min_recurrence_for_create", 2, raising=False)
        report = _report(
            new_concepts=[],
            collisions=[{
                "name": "entropia", "total_hits": 1, "best_match": "title",
                "hits": [{"path": "inbox/a.md", "count": 1, "in_title": True}],
            }],
        )
        hints = _hints(build_payload([report], window=450))
        assert hints["entropia"] == "enrich"


class TestBuildConceptEntryLegacyFallback:
    def test_collision_none_not_new_stays_create(self, fake_vault):
        """Pre-gate behavior: collision=None + in_new_concepts=False hardcoded 'create'
        (never routed through classify_action). Must stay bit-identical."""
        entry = build_concept_entry(
            name="entropia", inbox_content=fake_vault,
            collision=None, in_new_concepts=False, window=450,
        )
        assert entry["action_hint"] == "create"
