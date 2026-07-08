# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Static adapter registry — ADR-0014.

A list, not a plugin system (the ADR's scope line): N sources = N entries,
edited here. Dispatch is first-match over matches(); `enabled` (from the
vault manifest) filters which adapters participate for the current vault.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

from silica.sources.base import SourceAdapter
from silica.sources.code import CODE
from silica.sources.prose import PROSE

logger = logging.getLogger(__name__)

ALL_ADAPTERS: tuple[SourceAdapter, ...] = (PROSE, CODE)


def enabled_adapters(enabled: Sequence[str] | None = None) -> list[SourceAdapter]:
    """Adapters participating in dispatch. enabled=None → all registered."""
    if enabled is None:
        return list(ALL_ADAPTERS)
    known = {a.name for a in ALL_ADAPTERS}
    for name in enabled:
        if name not in known:
            logger.warning("vault manifest lists unknown source %r — ignored", name)
    return [a for a in ALL_ADAPTERS if a.name in enabled]


def adapter_for(target: str, enabled: Sequence[str] | None = None) -> SourceAdapter | None:
    for adapter in enabled_adapters(enabled):
        if adapter.matches(target):
            return adapter
    return None


def stage(adapter: SourceAdapter, target: str) -> dict:
    """read → to_stub → write, for terminal-lane stubs; status dict out.

    Distill-lane stubs are NOT written here — the Injector FSM owns that
    lane (ADR-0013); the caller forwards the target to the agent instead.
    """
    from silica.driver import DRIVER

    try:
        item = adapter.read(target)
        stub = adapter.to_stub(item)
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    if stub.lane != "terminal":
        return {"status": "distill", "target": target}
    DRIVER.create(stub.note_path, stub.body)
    return {"status": "ok", "note_path": stub.note_path, "meta": dict(item.meta)}
