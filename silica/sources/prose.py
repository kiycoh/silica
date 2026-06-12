"""Prose source adapter — markdown/.txt notes, the Obsidian-native source.

Distill lane: the Injector FSM owns destination and write; this adapter
only classifies, reads (sanitized, zero-trust frontier per ADR-0009) and
declares the lane. read() is total: unreadable targets yield an empty
RawItem rather than raising, because the FSM re-reads through the driver
anyway and dispatch must not block a batch on one bad path.
"""
from __future__ import annotations

from pathlib import Path

from silica.config import CONFIG
from silica.kernel.sanitize import strip_degenerate_runs
from silica.sources.base import GroundedStub, RawItem, Staleness

_EXTS = (".md", ".txt")


class ProseAdapter:
    name = "prose"

    def matches(self, target: str) -> bool:
        return target.lower().endswith(_EXTS)

    def read(self, target: str) -> RawItem:
        p = Path(target)
        if not p.is_absolute():
            vault = (CONFIG.vault_path or "").strip()
            p = (Path(vault) / target) if vault else (Path.cwd() / target)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return RawItem(target=target, text="")
        return RawItem(target=target, text=strip_degenerate_runs(text))

    def to_stub(self, item: RawItem) -> GroundedStub:
        return GroundedStub(lane="distill", body=item.text)

    def staleness(self, note_path: str) -> Staleness:
        return "fresh"  # native prose has no external source to drift from


PROSE = ProseAdapter()
