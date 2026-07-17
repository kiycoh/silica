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

_BUNDLED_OVERLAYS = (
    Path(__file__).resolve().parent.parent / "silica" / "overlays"
)


@pytest.fixture
def it_overlay():
    """Load the bundled Italian overlay."""
    path = _BUNDLED_OVERLAYS / "italian.yaml"
    if not path.exists():
        pytest.skip(f"bundled overlay not found: {path}")
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

    def search_context_batch(self, queries):
        return {q: [] for q in queries}


class _BatchSpyDriver:
    """Driver stub: batch returns one external hit per query; counts call types."""
    def __init__(self, body: str):
        self._body = body
        self.batch_calls = 0
        self.single_calls = 0

    def read_note(self, ref):
        from silica.driver.base import NoteContent, NoteRef
        return NoteContent(ref=NoteRef(name="note", path="inbox/note.md"), content=self._body)

    def search_context(self, query):
        self.single_calls += 1
        return []

    def search_context_batch(self, queries):
        self.batch_calls += 1
        from silica.driver.base import Hit, NoteRef
        ref = NoteRef(name="Other", path="vault/Other.md")
        return {q: [Hit(ref=ref, line=1, snippet=q)] for q in queries}


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


class _InboxHitDriver(_BatchSpyDriver):
    """Every query hits ONLY an Inbox note — a staging file indexed like any
    vault note. Real incident 2026-07-17: the SVM-book Inbox folder became the
    expected collision for half of Lezione 1's concepts, dooming every op."""
    def search_context_batch(self, queries):
        self.batch_calls += 1
        from silica.driver.base import Hit, NoteRef
        ref = NoteRef(name="01-intro", path="Inbox/svm-book/01-intro.md")
        return {q: [Hit(ref=ref, line=1, snippet=q)] for q in queries}


class TestReconBatch:
    def test_inbox_hits_never_become_collisions(self, monkeypatch):
        """An Inbox-only hit is not a collision: validate rejects every Inbox
        target, so registering one poisons the payload's vault_collision and
        the distiller can never produce an acceptable op. The concept must be
        classified as new instead."""
        import silica.tools.pipeline as pipe
        from silica.config import CONFIG
        monkeypatch.setattr(CONFIG, "defer_uncorroborated_concepts", False, raising=False)
        drv = _InboxHitDriver(_RECON_BODY)
        monkeypatch.setattr(pipe, "DRIVER", drv)

        res = pipe.silica_recon("inbox/note.md")

        assert res["collisions"] == []
        assert res["new_concepts"]

    def test_recon_uses_batch_search_once(self, monkeypatch):
        """Hot path issues ONE batch call (N->1) and never per-concept search."""
        import silica.tools.pipeline as pipe
        from silica.config import CONFIG
        monkeypatch.setattr(CONFIG, "defer_uncorroborated_concepts", False, raising=False)
        drv = _BatchSpyDriver(_RECON_BODY)
        monkeypatch.setattr(pipe, "DRIVER", drv)

        res = pipe.silica_recon("inbox/note.md")

        assert drv.batch_calls == 1            # one eval for all concepts
        assert drv.single_calls == 0           # no per-concept rescan anymore
        assert res["new_concepts"] == []       # every concept collided
        assert res["collisions"]               # collisions reported from batch hits


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


# ---------------------------------------------------------------------------
# strip_math (migrated to the kernel/text seam, C1) — LaTeX scrubbed from the
# extraction body (notes stay intact)
# ---------------------------------------------------------------------------

class TestStripMath:
    def test_strips_display_and_inline_spans(self):
        from silica.kernel.text import strip_math
        out = strip_math(
            r"prosa $$\sum_{i} x_i$$ poi $\mathbb{R}$ e \[ \int f \] e \( \alpha \) fine"
        )
        for junk in ("sum", "mathbb", "int", "alpha"):
            assert junk not in out
        assert "prosa" in out and "fine" in out

    def test_strips_residual_commands_outside_spans(self):
        from silica.kernel.text import strip_math
        out = strip_math(r"il vettore \mathbf{w} ha norma \leq uno")
        for junk in ("mathbf", "leq"):
            assert junk not in out
        assert "vettore" in out and "norma" in out and "uno" in out

    def test_leaves_prose_untouched_and_is_pure(self):
        from silica.kernel.text import strip_math
        src = "La rete neurale calcola il gradiente."
        out = strip_math(src)
        assert out == src                      # no math -> content unchanged
        assert src == "La rete neurale calcola il gradiente."  # input not mutated
