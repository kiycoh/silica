"""CapabilityBounds — execution bounds that constrain what a bounded sub-agent may write.

A bounded sub-agent (dedup, refiner) is allowed to *write*, but only within a
strictly bounded envelope.  The framework — not the model — decides:

  * which op-types are permitted (`allowed_ops`),
  * which note paths it may touch (`target_predicate` + `forbidden_paths`),
  * that no information is lost on a rewrite (`content_guard`).

`CapabilityBounds.enforce()` runs BEFORE the writer: any op outside the envelope
is dropped with a reason, so a small/eager model can never escalate beyond its
bounds.  The kept ops still flow through the normal validate→snapshot→write→lint
micro-gate.

Design note: enforcement is mechanical and deterministic.  The model only ever
proposes; the bounds dispose.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable

from silica.kernel.ast import WIKILINK_TARGET_RE
from silica.kernel.ops import Op, OpType


def _wikilinks(text: str) -> set[str]:
    """Return the set of wikilink targets in `text` (case-insensitive, trimmed)."""
    return {m.strip().lower() for m in WIKILINK_TARGET_RE.findall(text or "") if m.strip()}


def _norm_path(path: str | None) -> str:
    """Canonical comparison key for a vault path: posix, no .md, lowercase."""
    if not path:
        return ""
    return path.replace("\\", "/").removesuffix(".md").lower()


def make_no_info_loss_guard(floor_ratio: float = 0.85) -> Callable[[Op, str], str | None]:
    """Build a content_guard enforcing anti-deletion on a rewrite.

    Rejects an overwrite/patch when the new body drops any wikilink present in
    the original, or shrinks the note below `floor_ratio` of its original length.
    Returns a rejection reason string, or None when the op is acceptable.
    """
    def guard(op: Op, original: str) -> str | None:
        new = op.content if op.content is not None else (op.snippet or "")
        old_links = _wikilinks(original)
        new_links = _wikilinks(new)
        missing = old_links - new_links
        if missing:
            return f"info-loss: dropped wikilink(s) {sorted(missing)}"
        old_len = len(original.strip())
        new_len = len(new.strip())
        if old_len and new_len < floor_ratio * old_len:
            return (
                f"info-loss: body shrank to {new_len} chars "
                f"(< {floor_ratio:.0%} of {old_len})"
            )
        return None

    return guard


def make_link_addition_guard() -> Callable[[Op, str], str | None]:
    """Build a content_guard requiring a patch/overwrite to ADD at least one wikilink.

    Used by the orphan connector: a de-orphaning op that introduces no link is a
    no-op and is rejected, so the orphan is reported unresolved rather than
    silently "fixed".
    """
    def guard(op: Op, original: str) -> str | None:
        added = op.content if op.content is not None else (op.snippet or "")
        if not _wikilinks(added):
            return "orphan repair added no wikilink"
        return None

    return guard


@dataclass(frozen=True)
class CapabilityBounds:
    """Execution bounds for a bounded sub-agent."""

    name: str
    allowed_ops: frozenset[OpType]
    # path → True if the sub-agent may touch it. Receives the raw op path.
    target_predicate: Callable[[str], bool] = field(default=lambda _p: True)
    # exact vault paths that are never touchable (e.g. the run hub).
    forbidden_paths: frozenset[str] = frozenset()
    # optional semantic guard for rewrites: (op, original_content) → reason|None.
    content_guard: Callable[[Op, str], str | None] | None = None

    def allows_path(self, path: str | None) -> bool:
        norm = _norm_path(path)
        if not norm:
            return False
        forbidden_norms = {_norm_path(p) for p in self.forbidden_paths}
        if norm in forbidden_norms:
            return False
        # Bare-name forbidden entries (no "/" or "\") may be matched by the
        # incoming path's basename — e.g. hub="Concepts" blocks "notes/Concepts.md".
        # Only apply basename expansion for bare entries to avoid false positives
        # where a note named "Foo.md" is blocked by hub="Foo" even when the hub
        # is actually a different full path like "other/Foo".
        bare_forbidden = {
            _norm_path(p)
            for p in self.forbidden_paths
            if "/" not in p and "\\" not in p
        }
        if bare_forbidden and _norm_path(os.path.basename(path or "")) in bare_forbidden:
            return False
        return bool(self.target_predicate(path or ""))

    def enforce(
        self,
        ops: list[Op],
        *,
        read_note: Callable[[str], str] | None = None,
    ) -> tuple[list[Op], list[dict]]:
        """Split `ops` into (kept, rejected) according to the envelope.

        `read_note(path) -> str` supplies the original note body for the
        content_guard; defaults to the live DRIVER.  rejected entries are
        {"op": <dict>, "reason": <str>} so the caller can log/defer them.
        """
        kept: list[Op] = []
        rejected: list[dict] = []

        guarded_ops = {OpType.overwrite, OpType.patch}

        for op in ops:
            # Explicit no-ops always pass through untouched.
            if op.op == OpType.skip:
                kept.append(op)
                continue

            if op.op not in self.allowed_ops:
                rejected.append({
                    "op": op.model_dump(),
                    "reason": f"op '{op.op.value}' not permitted by bounds '{self.name}'",
                })
                continue

            path = op.touched_ref()
            if not self.allows_path(path):
                rejected.append({
                    "op": op.model_dump(),
                    "reason": f"target '{path}' outside bounds '{self.name}'",
                })
                continue

            if self.content_guard is not None and op.op in guarded_ops:
                original = self._read_original(path, read_note)
                reason = self.content_guard(op, original)
                if reason is not None:
                    rejected.append({
                        "op": op.model_dump(),
                        "reason": f"{reason} (bounds '{self.name}')",
                    })
                    continue

            kept.append(op)

        return kept, rejected

    @staticmethod
    def _read_original(path: str | None, read_note: Callable[[str], str] | None) -> str:
        if not path:
            return ""
        if read_note is not None:
            try:
                return read_note(path)
            except Exception:
                return ""
        # Fall back to the live driver (best-effort; missing note → empty).
        try:
            from silica.driver import DRIVER
            return DRIVER.read_note(path).content or ""
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# Presets — the bounds instances used by the in-pipeline sub-agents.
# ---------------------------------------------------------------------------

def dedup_bounds(larger_path: str, *, hub: str | None = None) -> CapabilityBounds:
    """Dedup bounds: append-only into the LARGER note of a borderline pair.

    The only permitted action is a `patch` against `larger_path`.  The model may
    never overwrite, delete, or create notes, and never touch the hub.  Which note
    is "larger" is decided mechanically by the framework (via ofm.metrics), not by
    the model.
    """
    larger_key = _norm_path(larger_path)
    forbidden = frozenset({hub} if hub else set())
    return CapabilityBounds(
        name="dedup",
        allowed_ops=frozenset({OpType.patch}),
        target_predicate=lambda p: _norm_path(p) == larger_key,
        forbidden_paths=forbidden,
    )


def refiner_bounds(
    target_path: str,
    *,
    hub: str | None = None,
    floor_ratio: float = 0.85,
) -> CapabilityBounds:
    """Refiner bounds: stylistic overwrite of one note, with anti-info-loss.

    Permits a single `overwrite` of `target_path` only if the rewrite preserves
    every wikilink and stays above `floor_ratio` of the original length.
    """
    target_key = _norm_path(target_path)
    forbidden = frozenset({hub} if hub else set())
    return CapabilityBounds(
        name="refiner",
        allowed_ops=frozenset({OpType.overwrite}),
        target_predicate=lambda p: _norm_path(p) == target_key,
        forbidden_paths=forbidden,
        content_guard=make_no_info_loss_guard(floor_ratio),
    )


def orphan_bounds(orphan_path: str, *, hub: str | None = None) -> CapabilityBounds:
    """Connector bounds: append-only patch into the orphan note that ADDS a link.

    Permits a single `patch` against `orphan_path` whose body introduces at least
    one wikilink (de-orphaning it).  Never overwrites, deletes, or creates.
    """
    orphan_key = _norm_path(orphan_path)
    forbidden = frozenset({hub} if hub else set())
    return CapabilityBounds(
        name="orphan",
        allowed_ops=frozenset({OpType.patch}),
        target_predicate=lambda p: _norm_path(p) == orphan_key,
        forbidden_paths=forbidden,
        content_guard=make_link_addition_guard(),
    )
