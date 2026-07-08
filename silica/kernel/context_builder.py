# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Context assembler for single-checkpoint LLM calls (Phase 2).

Guarantees that each LLM invocation receives exactly:
  1. ledger_digest — compact (<500 token) run summary.
  2. checkpoint payload — the input slice for this specific step.
  3. substrate — optional pre-fetched candidates from embeddings/graph (Phase 3+).

Nothing outside these three sources is forwarded to the model.
The function is pure (no I/O, no side-effects) and is safe to call from tests
without any external dependencies.
"""
from __future__ import annotations

import json


def build_context(
    checkpoint_id: str,
    payload: dict | str | None = None,
    ledger_digest: str | None = None,
    substrate: str | None = None,
) -> str:
    """Return a formatted context string for one checkpoint's LLM call.

    Args:
        checkpoint_id: identifier of the current step (used as section heading).
        payload: the checkpoint's own input data (dict → JSON-serialised).
        ledger_digest: compact run summary from ProgressLedger.digest().
        substrate: optional pre-fetched context from embeddings / graph (Phase 3).

    Returns:
        A formatted string ready for inclusion in a user message.
        Empty string if all inputs are None / empty.
    """
    parts: list[str] = []

    if ledger_digest and ledger_digest.strip():
        parts.append("## Run Context\n" + ledger_digest.strip())

    if substrate and substrate.strip():
        parts.append("## Related Notes (candidates)\n" + substrate.strip())

    if payload is not None:
        if isinstance(payload, str):
            payload_str = payload
        else:
            payload_str = json.dumps(payload, ensure_ascii=False, indent=2)
        parts.append(f"## Checkpoint: {checkpoint_id}\n" + payload_str)

    return "\n\n".join(parts)
