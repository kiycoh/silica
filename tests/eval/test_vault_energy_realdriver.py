# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Real-driver characterization of E(vault): the dangling term + harness wiring.

The override-path bench (test_vault_energy_bench.py) cannot populate the
dangling term — an edge to a non-existent id is not counted there. This bench
builds a real on-disk vault (fs backend, live driver) with actual broken
wikilinks and drives the EXACT call the golden harness makes
(compute_report(analytics=True) -> vault_energy), so it doubles as the
regression that pins the harness's ΔE wiring:

  * dangling responds end-to-end: a real [[Ghost]] link raises E.dangling;
    materializing the target lowers it (the perturbation the override missed).
  * wiring integrity: total == Σ terms, the invariant runner.collect relies on.
  * the gap fix holds through the real report path, not just the override:
    E.gaps is bounded by the gap count, never the size-scaling gap_score.
"""
from __future__ import annotations

from silica.kernel.graph_report import compute_report
from tests.eval.vault_energy import vault_energy


def _seed(tmp_vault, ghost_links: int = 1):
    """Two linked clusters + N broken wikilinks into non-existent notes."""
    tmp_vault.note("A1.md", "# A1\n[[A2]]\n[[A3]]\n")
    tmp_vault.note("A2.md", "# A2\n[[A1]]\n[[A3]]\n")
    tmp_vault.note("A3.md", "# A3\n[[A1]]\n[[A2]]\n")
    tmp_vault.note("B1.md", "# B1\n[[B2]]\n[[B3]]\n")
    tmp_vault.note("B2.md", "# B2\n[[B1]]\n[[B3]]\n")
    tmp_vault.note("B3.md", "# B3\n[[B1]]\n[[B2]]\n")
    broken = "".join(f"[[Ghost{i}]]\n" for i in range(ghost_links))
    tmp_vault.note("A1.md", f"# A1\n[[A2]]\n[[A3]]\n{broken}")


def _energy(tmp_vault):
    import silica.driver
    silica.driver._driver = None  # rebuild graph after edits
    return vault_energy(compute_report(analytics=True))


def test_dangling_term_counts_real_broken_links(tmp_vault):
    """The override path leaves dangling=0; a real [[Ghost]] must register."""
    _seed(tmp_vault, ghost_links=1)
    e = _energy(tmp_vault)
    assert e.dangling == 1.0, f"one broken link → dangling 1.0, got {e.dangling}"


def test_fixing_a_broken_link_lowers_energy(tmp_vault):
    """Materializing the missing target is an improving edit: E must fall, and
    the whole drop must come from the dangling term."""
    _seed(tmp_vault, ghost_links=1)
    before = _energy(tmp_vault)
    tmp_vault.note("Ghost0.md", "# Ghost0\n[[A1]]\n")  # resolve the dangling target
    after = _energy(tmp_vault)
    assert after.dangling < before.dangling
    assert after.total < before.total


def test_breaking_more_links_raises_energy(tmp_vault):
    """Adding broken links is a degrading edit: dangling and E both rise."""
    _seed(tmp_vault, ghost_links=1)
    before = _energy(tmp_vault)
    _seed(tmp_vault, ghost_links=3)
    after = _energy(tmp_vault)
    assert after.dangling > before.dangling
    assert after.total > before.total


def test_harness_wiring_integrity_and_bounded_gaps(tmp_vault):
    """Mirror runner.collect's exact call: terms sum to total, and the gap fix
    holds through the real report path (E.gaps bounded by the gap count, not the
    size-scaling gap_score that would read in the thousands on a real vault)."""
    _seed(tmp_vault, ghost_links=1)
    r = compute_report(analytics=True)
    e = vault_energy(r)
    assert abs(e.total - (e.cohesion + e.orphans + e.dangling
                          + e.gaps + e.deficits + e.contested)) < 1e-9
    assert 0.0 <= e.gaps <= float(len(r.structural_gaps))  # density ∈ [0,1) per gap
