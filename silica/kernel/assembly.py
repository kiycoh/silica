# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Read-time assembly (IWE-informed): reconstitute atomic notes into assembled
context at recall time. Pure module — no I/O, no LLM. Callers inject edge and
body readers so the logic is unit-testable without a vault.

Three ported ideas (spec 2026-07-21): directional expansion (1.1), squash by
hub (1.3), breadcrumb (2.3). Ranking is NOT touched here: seeds arrive already
ranked/reranked; neighbours enter as periphery and are never re-ranked.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Unit:
    """One assemblable piece of text: a ranked seed or a periphery neighbour."""
    path: str
    text: str
    is_seed: bool
    rank: int


@dataclass
class Truncation:
    """What the budget kept vs dropped (IWE Truncation style)."""
    kept: int = 0
    dropped: list[str] = field(default_factory=list)


def fill_budget(
    seeds: list[Unit], periphery: list[Unit], *, budget: int
) -> tuple[list[Unit], Truncation]:
    """Seeds first (never trimmed), then periphery by ascending rank until the
    char ceiling. Metering is by len(text) — chars, no tokenizer (spec Dec. 4).
    """
    kept: list[Unit] = list(seeds)
    used = sum(len(u.text) for u in kept)
    dropped: list[str] = []
    for u in sorted(periphery, key=lambda x: x.rank):
        if used + len(u.text) <= budget:
            kept.append(u)
            used += len(u.text)
        else:
            dropped.append(u.path)
    return kept, Truncation(kept=len(kept), dropped=dropped)


_ATX = re.compile(r"^(#{1,6})(\s)", re.MULTILINE)


@dataclass
class AssembledBlock:
    """One rendered block: a squashed hub group, or a lone (breadcrumbed) seed."""
    hub: str | None
    breadcrumb: str
    text: str
    members: list[str] = field(default_factory=list)


def relevel_headers(body: str, shift: int) -> str:
    """Deepen every ATX heading by `shift` levels (capped at H6)."""
    if shift <= 0:
        return body

    def _bump(m: re.Match) -> str:
        level = min(len(m.group(1)) + shift, 6)
        return "#" * level + m.group(2)

    return _ATX.sub(_bump, body)


def squash(
    units: list[Unit],
    hub_of: dict[str, str | None],
    breadcrumb_of: dict[str, str],
) -> list[AssembledBlock]:
    """Group co-hub units into single ordered blocks; lone units stay separate.

    A hub with >= 2 members: one block, "# Hub" header + each member re-leveled
    one level down, in rank order, breadcrumbed. A lone member (or hub=None): its
    own block, breadcrumb-prefixed, text unchanged (the degenerate case — spec
    1.3 "a single seed means no squash").
    """
    by_hub: dict[str | None, list[Unit]] = {}
    for u in sorted(units, key=lambda x: x.rank):
        by_hub.setdefault(hub_of.get(u.path), []).append(u)

    blocks: list[AssembledBlock] = []
    for hub, members in by_hub.items():
        if hub is not None and len(members) >= 2:
            crumb = breadcrumb_of.get(members[0].path, hub)
            parts = [f"# {hub}"]
            for m in members:
                parts.append(relevel_headers(m.text, 1))
            blocks.append(AssembledBlock(
                hub=hub,
                breadcrumb=crumb,
                text=(f"{crumb}\n\n" if crumb else "") + "\n\n".join(parts),
                members=[m.path for m in members],
            ))
        else:
            for m in members:
                crumb = breadcrumb_of.get(m.path, "")
                blocks.append(AssembledBlock(
                    hub=None,
                    breadcrumb=crumb,
                    text=(f"{crumb}\n\n" if crumb else "") + m.text,
                    members=[m.path],
                ))
    # Stable order: by the best (lowest) member rank so ranking intent survives.
    rank_of = {u.path: u.rank for u in units}
    blocks.sort(key=lambda b: min(rank_of[p] for p in b.members))
    return blocks


ASSEMBLY_BUDGET_CHARS = 12000  # ponytail: placeholder ceiling, tuned in A/B


@dataclass
class Neighbors:
    parent: str | None
    children: list[str]
    related: list[str]
    edges: list[str]


@dataclass
class Caps:
    parent: int = 1
    children: int = 1
    related: int = 1
    edges: int = 1


@dataclass
class AssemblyResult:
    blocks: list[AssembledBlock]
    truncation: Truncation


_MAX_CHAIN = 6  # breadcrumb walk bound (cycle guard)


def _breadcrumb(path: str, neighbors_of: Callable[[str], "Neighbors"]) -> str:
    """spoke -> parent -> hub path, walked up the parent chain (bounded)."""
    chain = [path]
    seen = {path}
    cur = path
    for _ in range(_MAX_CHAIN):
        parent = neighbors_of(cur).parent
        if not parent or parent in seen:
            break
        chain.append(parent)
        seen.add(parent)
        cur = parent
    return " > ".join(reversed(chain))  # hub > ... > spoke


def assemble(
    seed_paths: list[str],
    *,
    neighbors_of: Callable[[str], Neighbors],
    body_of: Callable[[str], str],
    caps: Caps = Caps(),
    budget: int = ASSEMBLY_BUDGET_CHARS,
) -> AssemblyResult:
    """Directional 1-hop expansion + budget + squash. See module docstring."""
    if not seed_paths:
        return AssemblyResult(blocks=[], truncation=Truncation())

    seed_set = set(seed_paths)
    seeds = [Unit(path=p, text=body_of(p), is_seed=True, rank=i)
             for i, p in enumerate(seed_paths)]

    periphery: list[Unit] = []
    seen: set[str] = set(seed_set)
    prank = 0
    for p in seed_paths:
        n = neighbors_of(p)
        directions = (
            ([n.parent] if n.parent else [])[: caps.parent],
            n.children[: caps.children],
            n.related[: caps.related],
            n.edges[: caps.edges],
        )
        for group in directions:
            for np in group:
                if np in seen:
                    continue
                seen.add(np)
                periphery.append(Unit(path=np, text=body_of(np),
                                      is_seed=False, rank=prank))
                prank += 1

    kept, trunc = fill_budget(seeds, periphery, budget=budget)
    hub_of = {u.path: neighbors_of(u.path).parent for u in kept}
    crumb_of = {u.path: _breadcrumb(u.path, neighbors_of) for u in kept}
    blocks = squash(kept, hub_of, crumb_of)
    return AssemblyResult(blocks=blocks, truncation=trunc)
