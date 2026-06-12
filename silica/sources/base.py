"""SourceAdapter contract — ADR-0014: sources are adapters, not vault types.

One seam that does not grow with the number of sources: each knowledge
source (prose, code, future zotero) implements this Protocol; everything
downstream (CORRELATE, relatedness, collision routing, enrich, undo) stays
the single shared pipeline. Zero-trust (ADR-0009): `read()` is the ingress
frontier — any adapter text that can reach a prompt must be sanitized in
`read()` or, for derived text, in `to_stub()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

# Lane (ADR-0013): "terminal" = mechanical vault-terminal write (no FSM);
# "distill" = full Injector FSM pipeline. The adapter decides, in to_stub.
Lane = Literal["terminal", "distill"]

Staleness = Literal["fresh", "stale", "unknown"]


@dataclass(frozen=True)
class RawItem:
    """Content read at the ingress frontier, plus adapter-specific metadata."""

    target: str                                  # user-supplied target (path; future: scheme)
    text: str                                    # content; prompt-bound text must be sanitized
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundedStub:
    """Draft + grounding produced by an adapter; declares its pipeline lane."""

    lane: Lane
    note_path: str = ""                          # vault-relative destination ("" → lane decides)
    body: str = ""                               # full markdown incl. grounding frontmatter


@runtime_checkable
class SourceAdapter(Protocol):
    """The whole contract. Anything not shared by every source stays out
    (in frontmatter), per the ADR's anti-bloat discipline."""

    name: str

    def matches(self, target: str) -> bool: ...

    def read(self, target: str) -> RawItem: ...

    def to_stub(self, item: RawItem) -> GroundedStub: ...

    def staleness(self, note_path: str) -> Staleness: ...
