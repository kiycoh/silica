# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""merge_overlay — apply a Domain Pack (overlay) onto a base recipe.

A Domain Pack adapts a recipe by overriding gate thresholds and per-phase
parameters. It may NOT add, remove, or reorder phases — that is a control-flow
change requiring the recipe->FSM compiler (subsystem C, deferred). This function
is the single place that boundary is enforced. See ADR-0005.
"""
from __future__ import annotations

import copy
from typing import Any


class OverlayError(ValueError):
    """An overlay tried to do something only the compiler (C) may do."""


def merge_overlay(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    if not overlay:
        return result

    # gates: shallow key override
    if "gates" in overlay:
        result.setdefault("gates", {}).update(overlay["gates"] or {})

    # phases: override params of existing ids only; same order enforced
    if "phases" in overlay:
        base_ids = [p.get("id") for p in result.get("phases", [])]
        base_by_id = {p.get("id"): p for p in result.get("phases", [])}
        seen: list[str] = []
        for ov_phase in overlay["phases"] or []:
            pid = ov_phase.get("id")
            if pid not in base_by_id:
                raise OverlayError(
                    f"overlay phase '{pid}' is not in the base recipe; adding/"
                    f"removing phases is a control-flow change (needs compiler C)"
                )
            base_by_id[pid].update({k: v for k, v in ov_phase.items() if k != "id"})
            seen.append(pid)
        # reject reordering: overlay phase ids must appear in base order
        if seen != [pid for pid in base_ids if pid in seen]:
            raise OverlayError("overlay may not reorder phases (needs compiler C)")

    # top-level scalars (name, inputs, ...) — override but never 'phases'/'gates'
    for k, v in overlay.items():
        if k not in ("gates", "phases"):
            result[k] = v

    return result
