"""Tests for silica.kernel.recon — concept filtering via the DomainOverlay seam.

Concept *extraction* now lives in silica.kernel.keyphrase (YAKE); recon keeps the
overlay-driven *filter* (`is_concept`) applied to every candidate, plus the
collision-ranking helpers. These tests guard the domain knowledge in the overlays
(which headings/words are noise) against the live filter.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from silica.kernel.overlay import DEFAULT_OVERLAY

_EXAMPLE_OVERLAYS = (
    Path(__file__).resolve().parent.parent / "examples" / "overlays"
)


@pytest.fixture
def it_overlay():
    """Load the Italian-academic example overlay."""
    path = _EXAMPLE_OVERLAYS / "it-academic.yaml"
    if not path.exists():
        pytest.skip(f"examples overlay not found: {path}")
    from silica.kernel.overlay import load_overlay
    return load_overlay(path)


# ---------------------------------------------------------------------------
# is_concept — noise rejected (default overlay)
# ---------------------------------------------------------------------------

class TestIsConceptFiltersNoise:
    @pytest.mark.parametrize("phrase", [
        "Chapter 3: Introduction",   # noise pattern ^(Chapter|Lesson|Exercise)\b[:\s]
        "Summary",                   # structural-noise word
        "the",                       # stopword
        "Resources:",                # trailing colon
        "What is recursion?",        # question
        "AI",                        # below MIN_LEN
        "NB: important",             # ^[A-Z]{2,6}:\s noise prefix
    ])
    def test_rejected(self, phrase):
        from silica.kernel.recon import is_concept
        assert not is_concept(phrase, overlay=DEFAULT_OVERLAY)


# ---------------------------------------------------------------------------
# is_concept — real concepts kept
# ---------------------------------------------------------------------------

class TestIsConceptKeepsConcepts:
    @pytest.mark.parametrize("phrase", ["Backpropagation", "Gradient Descent", "PID"])
    def test_kept_default(self, phrase):
        from silica.kernel.recon import is_concept
        assert is_concept(phrase, overlay=DEFAULT_OVERLAY)

    def test_italian_overlay_filters_noise(self, it_overlay):
        from silica.kernel.recon import is_concept
        assert not is_concept("Capitolo 3: Reti Neurali", overlay=it_overlay)
        assert not is_concept("unipa", overlay=it_overlay)  # vault stopword

    def test_italian_overlay_keeps_concepts(self, it_overlay):
        from silica.kernel.recon import is_concept
        assert is_concept("Reti Neurali", overlay=it_overlay)
        assert is_concept("Backpropagation", overlay=it_overlay)  # extends default


# ---------------------------------------------------------------------------
# is_concept — overlay argument honoured
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# silica_recon — degraded (embedder-down) extraction defers uncorroborated concepts
# ---------------------------------------------------------------------------

class _FakeDriver:
    """Driver stub: serves one note body, vault search finds nothing (all concepts new)."""
    def __init__(self, body: str):
        self._body = body

    def read_note(self, ref):
        from silica.driver.base import NoteContent, NoteRef
        return NoteContent(ref=NoteRef(name="note", path="inbox/note.md"), content=self._body)

    def search_context(self, query):
        return []


# Heading is 4 words → YAKE (n=3) can't produce it → _seed_structural prepends it,
# so the corroborated concept survives the MIN_CONCEPTS=1 cutoff. Body is long
# enough (k = tokens // 20 ≥ 2) for at least one prose-only (INFERRED) concept too.
_STRUCTURAL = "knowledge graph memory system"
_RECON_BODY = (
    "# Knowledge Graph Memory System\n\n"
    "The planning agent stores memory in the graph and retrieves planning context "
    "across many tasks and domains. Memory recall improves planning, and the agent "
    "reasons over stored knowledge for later planning tasks and decision making. "
    "The system indexes past episodes so the planner can resume work from memory reliably."
)


class TestReconDeferral:
    """Embedder is None here (autouse _no_recon_embedder) → degraded extraction."""

    def test_defers_uncorroborated_when_flag_on(self, monkeypatch):
        import silica.tools.pipeline as pipe
        from silica.config import CONFIG
        monkeypatch.setattr(pipe, "DRIVER", _FakeDriver(_RECON_BODY))
        monkeypatch.setattr(CONFIG, "defer_uncorroborated_concepts", True, raising=False)

        res = pipe.silica_recon("inbox/note.md")

        admitted = set(res["new_concepts"]) | {c["name"] for c in res["collisions"]}
        deferred = set(res["deferred_concepts"])
        assert _STRUCTURAL in admitted    # structural → corroborated → admitted now
        assert deferred                          # single-signal concepts held back
        assert not (admitted & deferred)         # admitted XOR deferred — never both

    def test_admits_everything_when_flag_off(self, monkeypatch):
        """Default: embedder-free vaults must not defer (no later pass is coming)."""
        import silica.tools.pipeline as pipe
        from silica.config import CONFIG
        monkeypatch.setattr(pipe, "DRIVER", _FakeDriver(_RECON_BODY))
        monkeypatch.setattr(CONFIG, "defer_uncorroborated_concepts", False, raising=False)

        res = pipe.silica_recon("inbox/note.md")

        assert res["deferred_concepts"] == []    # nothing held back
        admitted = set(res["new_concepts"]) | {c["name"] for c in res["collisions"]}
        assert _STRUCTURAL in admitted     # uncorroborated concepts still admitted


class TestIsConceptOverlayArg:
    def test_explicit_overlay_used_over_active(self):
        """is_concept uses an explicitly passed overlay, not get_active_overlay."""
        from silica.kernel.overlay import DomainOverlay
        import re
        block_bp = DomainOverlay(
            stopwords=frozenset(),
            noise_patterns=(re.compile(r"^Backpropagation$", re.IGNORECASE),),
        )
        from silica.kernel.recon import is_concept
        assert not is_concept("Backpropagation", overlay=block_bp)
        assert is_concept("Backpropagation", overlay=DEFAULT_OVERLAY)

    def test_explicit_stopword_overlay(self):
        """is_concept filters a word that is a stopword only in the explicit overlay."""
        from silica.kernel.overlay import DomainOverlay
        custom_overlay = DomainOverlay(
            stopwords=frozenset({"neuralnetwork"}),
            noise_patterns=(),
        )
        from silica.kernel.recon import is_concept
        assert not is_concept("neuralnetwork", overlay=custom_overlay)

    def test_none_overlay_uses_active(self, monkeypatch):
        """is_concept(s, overlay=None) falls back to get_active_overlay()."""
        from silica.kernel.overlay import DomainOverlay
        sentinel = DomainOverlay(
            stopwords=frozenset({"sentinel_word"}),
            noise_patterns=(),
        )
        monkeypatch.setattr("silica.kernel.recon.get_active_overlay", lambda: sentinel)
        from silica.kernel.recon import is_concept
        assert not is_concept("sentinel_word")
