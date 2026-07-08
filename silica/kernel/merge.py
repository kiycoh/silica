# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""3-way merge with conflict callout (Tier 2, Item 7, ADR-0007 soft-failure sink).

When a write lands on a note that was modified concurrently (base != current),
we inject an Obsidian danger callout rather than silently overwriting.  The
incoming content is still written; the callout gives the vault owner a clear
signal to review the merge manually.

Terminology:
    base     — content at snapshot time (what we expected)
    current  — content on disk when the write lands
    incoming — what the op wants to write
"""
from __future__ import annotations

CONFLICT_CALLOUT_HEADER = "> [!danger] Conflitto Semantico"

_CALLOUT_BODY = """\
> [!danger] Conflitto Semantico
> This note was modified concurrently. Review and merge the sections below manually.

"""


def detect_conflict(base: str | None, current: str | None) -> bool:
    """Return True iff base and current both exist and differ."""
    if base is None or current is None:
        return False
    return base != current


def inject_conflict_callout(content: str) -> str:
    """Prepend the conflict callout to content (idempotent)."""
    if CONFLICT_CALLOUT_HEADER in content:
        return content
    return _CALLOUT_BODY + content


def three_way_merge(
    base: str | None,
    current: str | None,
    incoming: str,
) -> tuple[str, bool]:
    """Merge incoming content, injecting a conflict callout if base != current.

    Returns:
        (merged_content, had_conflict)
    """
    if detect_conflict(base, current):
        return inject_conflict_callout(incoming), True
    return incoming, False
