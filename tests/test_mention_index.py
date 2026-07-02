"""Fix 4 — mention index scales to large vaults (linear, not O(N²·L)).

The mention index maps each note title to the notes whose body mentions it
(BACKLINK candidate generation). The old build substring-checked every title
against every note body — O(N²·L), ~32s projected at 10k notes.

The linear build is a *first-word-anchored substring* inverted index: titles are
bucketed by their first word; a note's body is matched only against titles whose
first word appears as a word in that body, then the full title is substring-
verified. This keeps morphology/suffix recall (plural "networks" still matches
title "Network") while dropping mid-word false positives ("ros" inside "across").
"""
from __future__ import annotations

import time

from silica.driver.fs_backend import ObsidianFSBackend


def _vault(tmp_path, notes: dict[str, str]):
    for name, body in notes.items():
        (tmp_path / f"{name}.md").write_text(body, encoding="utf-8")
    return ObsidianFSBackend(str(tmp_path))


def test_word_anchored_mention_found(tmp_path):
    b = _vault(tmp_path, {
        "Neural Network": "A note defining the concept.",
        "Essay": "This essay discusses the neural network in depth.",
    })
    assert "Essay.md" in b.mentions_of("Neural Network")


def test_morphology_suffix_still_matches(tmp_path):
    """Plural in the body still matches the singular title (substring verify)."""
    b = _vault(tmp_path, {
        "Network": "title note",
        "Essay": "We compare several networks here.",
    })
    assert "Essay.md" in b.mentions_of("Network")


def test_midword_false_positive_dropped(tmp_path):
    """'ros' must NOT match inside 'across' — the first word must be a real word."""
    b = _vault(tmp_path, {
        "Ros": "title note",
        "Geo": "We walked across the field.",
    })
    assert "Geo.md" not in b.mentions_of("Ros")


def test_multiword_phrase_contiguity_preserved(tmp_path):
    """Multi-word titles still require the contiguous phrase (substring verify)."""
    b = _vault(tmp_path, {
        "Neural Network": "title note",
        "Scattered": "neural systems and a separate network module",  # not contiguous
        "Contiguous": "the neural network powers it",
    })
    hits = b.mentions_of("Neural Network")
    assert "Contiguous.md" in hits
    assert "Scattered.md" not in hits


def test_single_char_first_word_title_matched(tmp_path):
    """A title whose first token is one char (e.g. 'C#') is still found."""
    b = _vault(tmp_path, {
        "C#": "title note",
        "Essay": "I love C# programming.",
    })
    assert "Essay.md" in b.mentions_of("C#")


def test_large_vault_builds_near_linear(tmp_path):
    """Scaling guard at the 10k-note target.

    Measured: linear build ≈1.1s at N=10000 (456ms at 4000 — grows linearly).
    The old O(N²·L) build extrapolates to ~15-30s at 10000. The 5s bound clears
    the linear build with ~5x headroom while a quadratic regression blows it.
    """
    import random
    words = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda "
             "neural network gradient tensor vector matrix cluster node edge").split()
    n = 10000
    for i in range(n):
        body = " ".join(random.choice(words) for _ in range(60))
        if i == 0:
            body += " see also Neural Network for context"
        (tmp_path / f"Concept {i}.md").write_text(f"{body}\n", encoding="utf-8")
    (tmp_path / "Neural Network.md").write_text("the title note\n", encoding="utf-8")

    b = ObsidianFSBackend(str(tmp_path))
    t0 = time.perf_counter()
    b._ensure_index()
    dt = time.perf_counter() - t0

    assert dt < 5.0, f"mention index build too slow ({dt:.1f}s) — quadratic regression?"
    assert "Concept 0.md" in b.mentions_of("Neural Network")  # correct at scale
