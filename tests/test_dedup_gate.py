# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

from silica.capabilities.dedup import passes_dedup_gate


def test_high_cosine_similar_size_passes():
    assert passes_dedup_gate(0.90, incoming_len=500, candidate_len=600) is True


def test_size_guard_rejects_spoke_in_hub():
    # High cosine but the "candidate" is a 10x-larger hub -> not a merge pair.
    assert passes_dedup_gate(0.90, incoming_len=200, candidate_len=5000) is False


def test_below_threshold_rejected():
    assert passes_dedup_gate(0.70, incoming_len=500, candidate_len=600) is False


def test_threshold_boundary_inclusive():
    assert passes_dedup_gate(0.85, incoming_len=500, candidate_len=600) is True
