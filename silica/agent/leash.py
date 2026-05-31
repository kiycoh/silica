"""The Leash — a capability envelope that keeps a sub-agent on a tight rein.

A leashed sub-agent (dedup, refiner) is allowed to *write*, but only within a
strictly bounded envelope.  The framework — not the model — decides:

  * which op-types are permitted (`allowed_ops`),
  * which note paths it may touch (`target_predicate` + `forbidden_paths`),
  * that no information is lost on a rewrite (`content_guard`),
  * how far it may explore before being reined in (`max_turns`, `timeout_s`,
    `context_budget_chars`).

`Leash.enforce()` runs BEFORE the writer: any op outside the envelope is dropped
with a reason, so a small/eager model can never escalate beyond its leash.  The
kept ops still flow through the normal validate→snapshot→write→lint micro-gate.

Design note: enforcement is mechanical and deterministic.  The model only ever
proposes; the leash disposes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from silica.kernel.ops import Op, OpType

# Wikilink extraction for the anti-info-loss guard: matches [[Target]] and
# [[Target|alias]] (the target before the first pipe is what matters for links).
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)")


def _wikilinks(text: str) -> set[str]:
    """Return the set of wikilink targets in `text` (case-insensitive, trimmed)."""
    return {m.strip().lower() for m in _WIKILINK_RE.findall(text or "") if m.strip()}


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
class Leash:
    """A bounded capability envelope for a leashed sub-agent."""

    name: str
    allowed_ops: frozenset[OpType]
    # path → True if the sub-agent may touch it. Receives the raw op path.
    target_predicate: Callable[[str], bool] = field(default=lambda _p: True)
    # exact vault paths that are never touchable (e.g. the run hub).
    forbidden_paths: frozenset[str] = frozenset()
    # optional semantic guard for rewrites: (op, original_content) → reason|None.
    content_guard: Callable[[Op, str], str | None] | None = None
    # exploration caps
    max_turns: int = 6
    timeout_s: float = 120.0
    context_budget_chars: int = 8000

    def allows_path(self, path: str | None) -> bool:
        key = _norm_path(path)
        if not key:
            return False
        if key in {_norm_path(p) for p in self.forbidden_paths}:
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
                    "reason": f"op '{op.op.value}' not permitted by leash '{self.name}'",
                })
                continue

            path = op.touched_ref()
            if not self.allows_path(path):
                rejected.append({
                    "op": op.model_dump(),
                    "reason": f"target '{path}' outside leash '{self.name}'",
                })
                continue

            if self.content_guard is not None and op.op in guarded_ops:
                original = self._read_original(path, read_note)
                reason = self.content_guard(op, original)
                if reason is not None:
                    rejected.append({
                        "op": op.model_dump(),
                        "reason": f"{reason} (leash '{self.name}')",
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
# Presets — the two leashes used by the in-pipeline sub-agents.
# ---------------------------------------------------------------------------

def dedup_leash(larger_path: str, *, hub: str | None = None) -> Leash:
    """Dedup envelope: append-only into the LARGER note of a borderline pair.

    The only permitted action is a `patch` against `larger_path`.  The model may
    never overwrite, delete, or create notes, and never touch the hub.  Which note
    is "larger" is decided mechanically by the framework (via ofm.metrics), not by
    the model.
    """
    larger_key = _norm_path(larger_path)
    forbidden = frozenset({hub} if hub else set())
    return Leash(
        name="dedup",
        allowed_ops=frozenset({OpType.patch}),
        target_predicate=lambda p: _norm_path(p) == larger_key,
        forbidden_paths=forbidden,
    )


def refiner_leash(
    target_path: str,
    *,
    hub: str | None = None,
    floor_ratio: float = 0.85,
) -> Leash:
    """Refiner envelope: stylistic overwrite of one note, with anti-info-loss.

    Permits a single `overwrite` of `target_path` only if the rewrite preserves
    every wikilink and stays above `floor_ratio` of the original length.
    """
    target_key = _norm_path(target_path)
    forbidden = frozenset({hub} if hub else set())
    return Leash(
        name="refiner",
        allowed_ops=frozenset({OpType.overwrite}),
        target_predicate=lambda p: _norm_path(p) == target_key,
        forbidden_paths=forbidden,
        content_guard=make_no_info_loss_guard(floor_ratio),
    )


def orphan_leash(orphan_path: str, *, hub: str | None = None) -> Leash:
    """Connector envelope: append-only patch into the orphan note that ADDS a link.

    Permits a single `patch` against `orphan_path` whose body introduces at least
    one wikilink (de-orphaning it).  Never overwrites, deletes, or creates.
    """
    orphan_key = _norm_path(orphan_path)
    forbidden = frozenset({hub} if hub else set())
    return Leash(
        name="orphan",
        allowed_ops=frozenset({OpType.patch}),
        target_predicate=lambda p: _norm_path(p) == orphan_key,
        forbidden_paths=forbidden,
        content_guard=make_link_addition_guard(),
    )
