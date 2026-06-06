from __future__ import annotations

import pytest

from silica.router.overlay import merge_overlay, OverlayError


def _base():
    return {
        "name": "injector",
        "gates": {"rejection_rate_max": 0.10, "graph_regression": "forbid_new_orphans"},
        "phases": [
            {"id": "recon", "kind": "mechanical", "tool": "silica_recon"},
            {"id": "payload", "kind": "mechanical", "tool": "silica_payload", "partition_if_over": 7},
            {"id": "distill", "kind": "semantic", "worker": "distiller", "max_workers": 7},
        ],
    }


def test_empty_overlay_is_identity():
    base = _base()
    assert merge_overlay(base, {}) == base


def test_gate_threshold_override():
    out = merge_overlay(_base(), {"gates": {"rejection_rate_max": 0.05}})
    assert out["gates"]["rejection_rate_max"] == 0.05
    # untouched gate keys survive
    assert out["gates"]["graph_regression"] == "forbid_new_orphans"


def test_per_phase_param_override_by_id():
    out = merge_overlay(_base(), {"phases": [{"id": "payload", "partition_if_over": 4}]})
    payload = next(p for p in out["phases"] if p["id"] == "payload")
    assert payload["partition_if_over"] == 4
    # other phase params and order preserved
    assert payload["tool"] == "silica_payload"
    assert [p["id"] for p in out["phases"]] == ["recon", "payload", "distill"]


def test_unknown_phase_id_is_rejected():
    # adding a phase is a control-flow change → needs compiler C (deferred)
    with pytest.raises(OverlayError):
        merge_overlay(_base(), {"phases": [{"id": "translate", "kind": "semantic"}]})


def test_reordering_is_rejected():
    # an overlay cannot change phase order
    reordered = {"phases": [{"id": "distill"}, {"id": "recon"}]}
    with pytest.raises(OverlayError):
        merge_overlay(_base(), reordered)


def test_does_not_mutate_base():
    base = _base()
    merge_overlay(base, {"gates": {"rejection_rate_max": 0.01}})
    assert base["gates"]["rejection_rate_max"] == 0.10
