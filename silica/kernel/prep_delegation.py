"""Prepare and run Distiller delegation for the Injector pipeline.

This module ports `build_tasks()` from Hermes prep_delegation.py as a pure
function, and adds `run_distiller()` which calls the LLM directly via
`call_llm()` (stateless, single-turn, no tool use).

The protocol template uses {TARGET} as the only substitution. PAYLOAD_PATH
is passed as a file reference in the task context, not inlined into the prompt.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Distiller prompt template — vendored at install time
_PROMPT_PATH = Path(__file__).resolve().parent.parent / "workers" / "distiller_prompt.txt"


def _load_prompt() -> str:
    if not _PROMPT_PATH.exists():
        raise FileNotFoundError(f"Distiller prompt not found: {_PROMPT_PATH}")
    return _PROMPT_PATH.read_text(encoding="utf-8")


def render_prompt(target: str, hub: str | None = None) -> str:
    """Render the distiller prompt with TARGET substitution."""
    body = _load_prompt()
    body = body.replace("{TARGET}", target)
    if hub:
        body = body.replace("{HUB_NAME}", hub)
    return body


def payload_checksum(payload_json: str) -> str:
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def run_distiller(payload: dict, target: str, hub: str | None = None) -> dict:
    """Call the Distiller LLM (single-turn) for one payload chunk.

    Args:
        payload: the payload dict (schema_version + batches)
        target: vault-relative target directory for new notes
        hub: optional [[Hub]] note name

    Returns:
        parsed dict with {"updates": [...]} or {"error": ...}
    """
    from silica.agent.providers import get_provider
    from silica.config import CONFIG
    from silica.kernel.ops import DistillerOutput
    from silica.kernel.sanitize import parse_json

    prompt_text = render_prompt(target=target, hub=hub)
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    checksum = payload_checksum(payload_json)

    # Build a single-turn user message: protocol + payload inline.
    # For S2.3 (one worker), we inline the payload directly.
    # S3.1 (fan-out) passes payload-by-pointer via file reference.
    user_message = (
        f"{prompt_text}\n\n"
        f"---\n"
        f"## Payload (SHA-256: {checksum})\n\n"
        f"{payload_json}"
    )

    logger.info("Calling Distiller LLM (payload checksum %s)", checksum[:12])

    try:
        provider = get_provider(CONFIG)
        response = provider.call_llm(
            messages=[{"role": "user", "content": user_message}],
            tools=None,
            response_schema=DistillerOutput,
            max_tokens=4000,
        )
    except Exception as e:
        logger.warning("Distiller provider call failed, falling back to litellm: %s", e)
        from silica.agent.llm import call_llm
        response = call_llm(
            model=CONFIG.model,
            messages=[{"role": "user", "content": user_message}],
            tools=None,
            max_tokens=4000,
        )

    if response.finish_reason == "length":
        logger.error("Distiller call hit maximum tokens limit (generation cut off)")
        return {"error": "Distiller call hit maximum tokens limit (generation cut off)"}

    raw_output = response.text or ""
    if not raw_output.strip():
        return {"error": "Distiller returned empty response"}

    try:
        parsed, _ = parse_json(raw_output, strict=False)
    except Exception as e:
        return {"error": f"Distiller output JSON parse failed: {e}", "raw": raw_output[:500]}

    if not isinstance(parsed, dict) or "updates" not in parsed:
        return {"error": "Distiller output missing 'updates' key", "raw": raw_output[:500]}

    logger.info("Distiller produced %d updates", len(parsed["updates"]))
    return parsed
