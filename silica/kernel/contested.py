# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Contested-claims layer (spec-hermes-coherence §1).

A contradiction is neither a duplicate nor a new concept: it is recorded on
the existing note (frontmatter flag + warning callout) and kept visible until
a human resolves it. Pure functions over note text — no I/O, no LLM.
"""
from __future__ import annotations

from silica.kernel import frontmatter

CONTESTED_KEY = "contested"
CONTRADICTIONS_KEY = "contradictions"


def mark_contested(content: str, source_ref: str) -> str:
    """Set `contested: true` and append `source_ref` to `contradictions:`.

    Idempotent on source_ref. A note without frontmatter gains a minimal one;
    a note with unparseable YAML is returned unchanged (never destroy what we
    cannot round-trip).
    """
    data, raw, body = frontmatter.split(content)
    if data is None:
        if raw is not None:  # frontmatter present but broken YAML
            return content
        data, body = {}, content
    refs = list(data.get(CONTRADICTIONS_KEY) or [])
    if source_ref in refs:
        return content
    data[CONTESTED_KEY] = True
    data[CONTRADICTIONS_KEY] = refs + [source_ref]
    return frontmatter.dump(data, body)


def clear_contested(content: str) -> str:
    """Remove the `contested`/`contradictions` flag from a note's frontmatter.

    No-op when the note is not contested; a note with unparseable YAML is
    returned unchanged (mirror of `mark_contested`).
    """
    data, raw, body = frontmatter.split(content)
    if data is None or not data.get(CONTESTED_KEY):
        return content
    data.pop(CONTESTED_KEY, None)
    data.pop(CONTRADICTIONS_KEY, None)
    return frontmatter.dump(data, body)


def contested_refs(content: str) -> list[str]:
    """The note's `contradictions:` entries; [] when not contested."""
    data, _, _ = frontmatter.split(content)
    if not data or not data.get(CONTESTED_KEY):
        return []
    return list(data.get(CONTRADICTIONS_KEY) or [])


def contested_callout(claim: str, source_basename: str) -> str:
    """The warning callout recording a conflicting claim, with provenance."""
    quoted = "\n".join(f"> {line}".rstrip() for line in claim.strip().splitlines())
    return (
        f"> [!warning] Contradiction — from {source_basename}\n"
        f"{quoted}\n"
        f">\n"
        f"> Conflicts with this note. Unresolved."
    )
