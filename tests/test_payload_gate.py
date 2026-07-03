"""classify_action + build_payload wiring — action hints for the distiller payload."""
from __future__ import annotations

import pytest

from silica.kernel.payload import build_concept_entry, build_payload, classify_action


# ---------------------------------------------------------------------------
# classify_action
# ---------------------------------------------------------------------------

class TestClassifyAction:
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


# ---------------------------------------------------------------------------
# build_payload wiring
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
    """One inbox note: 'entropia' twice, 'retropropagazione' once."""
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


class TestBuildPayload:
    def test_every_new_concept_creates(self, fake_vault):
        hints = _hints(build_payload([_report()], window=450))
        assert hints == {"entropia": "create", "retropropagazione": "create"}

    def test_collision_entries_get_collision_hints(self, fake_vault):
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
        """collision=None + in_new_concepts=False is hardcoded 'create'
        (never routed through classify_action). Must stay bit-identical."""
        entry = build_concept_entry(
            name="entropia", inbox_content=fake_vault,
            collision=None, in_new_concepts=False, window=450,
        )
        assert entry["action_hint"] == "create"
