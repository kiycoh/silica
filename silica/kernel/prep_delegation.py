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
from pathlib import Path

logger = logging.getLogger(__name__)

# Distiller prompt template — vendored at install time
_PROMPT_PATH = Path(__file__).resolve().parent.parent / "capabilities" / "prompts" / "distiller_prompt.txt"


def _load_prompt() -> str:
    if not _PROMPT_PATH.exists():
        raise FileNotFoundError(f"Distiller prompt not found: {_PROMPT_PATH}")
    return _PROMPT_PATH.read_text(encoding="utf-8")


def render_prompt(target: str, hub: str | None = None) -> str:
    """Render the distiller prompt with TARGET/LANGUAGE/MAX_TAGS substitution.

    LANGUAGE and MAX_TAGS come from the active vault's `conventions:` block
    (silica/kernel/vault_manifest.py) — single source shared with
    `ofm.ofm_lint`'s max-tags check. A vault without a manifest gets today's
    defaults (Italian / 3), so this is bit-identical when unconfigured.
    """
    from silica.kernel.vault_manifest import get_active_manifest

    body = _load_prompt()
    body = body.replace("{TARGET}", target)
    if hub:
        body = body.replace("{HUB_NAME}", hub)
    conventions = get_active_manifest().conventions
    body = body.replace("{LANGUAGE}", conventions.language)
    body = body.replace("{MAX_TAGS}", str(conventions.max_tags))
    return body


def payload_checksum(payload_json: str) -> str:
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


# Floor for the computed output budget. If the prompt is so large that no
# meaningful headroom remains, we still ask for at least this much rather than a
# negative/zero value — the API call will surface the real problem instead of us
# silently requesting nonsense.
_MIN_DISTILLER_OUTPUT_TOKENS = 1024


def estimate_prompt_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 chars/token), rounded up.

    Intentionally provider-agnostic: we only need a conservative figure to
    leave output headroom, not an exact tokenizer count.
    """
    return (len(text) + 3) // 4


def compute_distiller_max_tokens(
    prompt_text: str,
    *,
    context_window: int,
    safety_margin: int,
    ceiling: int = 0,
) -> int:
    """Size the output budget to the real prompt and the model's context window.

    `max_tokens = min(ceiling?, context_window - prompt_tokens - safety_margin)`,
    floored at `_MIN_DISTILLER_OUTPUT_TOKENS`.

    `ceiling <= 0` means "no manual cap" — use all available headroom. This
    removes the artificial 32k ceiling that was truncating dense batches while
    still letting an operator pin a hard limit via DISTILLER_MAX_TOKENS.
    """
    available = context_window - estimate_prompt_tokens(prompt_text) - safety_margin
    available = max(_MIN_DISTILLER_OUTPUT_TOKENS, available)
    if ceiling and ceiling > 0:
        return min(ceiling, available)
    return available


def salvage_distiller_json(raw: str) -> dict | None:
    """Recover the complete `updates` entries from a truncated distiller response.

    The distiller emits one large `{"main_thematic_axes": [...], "updates": [...]}`
    object. When generation is cut off mid-array the whole document is
    unparseable, but every element BEFORE the truncation point is valid JSON.
    This scans the `updates` array element-by-element and keeps every object that
    parses cleanly, discarding only the final half-written one.

    Returns `{"main_thematic_axes": [...], "updates": [...]}` with at least one
    recovered update, or None when nothing complete can be salvaged.
    """
    decoder = json.JSONDecoder()

    axes: list = []
    ax_key = raw.find('"main_thematic_axes"')
    if ax_key != -1:
        ax_bracket = raw.find("[", ax_key)
        if ax_bracket != -1:
            try:
                axes, _ = decoder.raw_decode(raw, ax_bracket)
            except ValueError:
                axes = []

    up_key = raw.find('"updates"')
    if up_key == -1:
        return None
    arr = raw.find("[", up_key)
    if arr == -1:
        return None

    updates: list = []
    i = arr + 1
    n = len(raw)
    while i < n:
        while i < n and raw[i] in " \t\r\n,":
            i += 1
        if i >= n or raw[i] == "]":
            break
        try:
            obj, end = decoder.raw_decode(raw, i)
        except ValueError:
            break  # trailing object is truncated — stop here
        if isinstance(obj, dict):
            updates.append(obj)
        i = end

    if not updates:
        return None
    return {"main_thematic_axes": axes if isinstance(axes, list) else [], "updates": updates}


def run_distiller(
    payload: dict,
    target: str,
    hub: str | None = None,
    ledger_digest: str | None = None,
    steer_context: str | None = None,
    substrate: str | None = None,
) -> dict:
    """Call the Distiller LLM (single-turn) for one payload chunk.

    Args:
        payload: the payload dict (schema_version + batches)
        target: vault-relative target directory for new notes
        hub: optional [[Hub]] note name
        ledger_digest: compact run summary injected as context header (Phase 2)
        steer_context: corrective steering note injected when re-attempting after
            rejection (Phase 6). States why the previous output was rejected.

    Returns:
        parsed dict with {"updates": [...]} or {"error": ...}
    """
    from silica.agent.providers import get_provider
    from silica.config import CONFIG
    from silica.kernel.context_builder import build_context
    from silica.kernel.ops import DistillerOutput
    from silica.kernel.sanitize import parse_json

    prompt_text = render_prompt(target=target, hub=hub)
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    checksum = payload_checksum(payload_json)

    # Assemble context through the context assembler (Phase 2 rails).
    # Only ledger_digest + the checkpoint payload reach the model — no other
    # vault content is forwarded here.
    ctx = build_context(
        checkpoint_id="distill",
        payload=payload,
        ledger_digest=ledger_digest,
        substrate=substrate,
    )

    steer_section = (
        f"\n\n## STEERING CORRECTION (attempt {steer_context.split('|attempt=')[1].split('|')[0] if '|attempt=' in steer_context else '?'})\n"
        f"{steer_context}\n\n"
        f"Please revise your output to avoid the issues described above.\n"
        if steer_context else ""
    )
    user_message = (
        f"{prompt_text}\n\n"
        f"---\n"
        f"Payload SHA-256: {checksum}\n\n"
        f"{ctx}"
        f"{steer_section}"
    )

    logger.info("Calling Distiller LLM (payload checksum %s)", checksum[:12])

    # #2: size the output budget to the real prompt + model context window
    # instead of a fixed ceiling. DISTILLER_MAX_TOKENS, when set, is treated as
    # an optional hard cap; unset/0 means "use all available headroom".
    context_window = int(os.getenv("MODEL_CONTEXT_WINDOW", "262144"))
    safety_margin = int(os.getenv("DISTILLER_TOKEN_SAFETY_MARGIN", "2048"))
    ceiling = int(os.getenv("DISTILLER_MAX_TOKENS", "0"))
    max_tokens = compute_distiller_max_tokens(
        user_message,
        context_window=context_window,
        safety_margin=safety_margin,
        ceiling=ceiling,
    )
    logger.info(
        "Distiller output budget: %d tokens (window=%d, prompt≈%d, margin=%d, ceiling=%s)",
        max_tokens, context_window, estimate_prompt_tokens(user_message), safety_margin,
        ceiling if ceiling > 0 else "none",
    )

    try:
        provider = get_provider(CONFIG, role="worker")
        response = provider.call_llm(
            messages=[{"role": "user", "content": user_message}],
            tools=None,
            response_schema=DistillerOutput,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.warning("Distiller provider call failed, falling back to litellm: %s", e)
        from silica.agent.llm import call_llm
        response = call_llm(
            model=CONFIG.model,
            messages=[{"role": "user", "content": user_message}],
            tools=None,
            max_tokens=max_tokens,
        )

    raw_output = response.text or ""
    if not raw_output.strip():
        return {"error": "Distiller returned empty response"}

    # #1: a truncated response (finish_reason == "length", or any malformed
    # trailing object) must not kill the whole batch. Try a clean parse first;
    # on failure, salvage every complete `updates` entry from the valid prefix.
    try:
        parsed, _ = parse_json(raw_output, strict=False)
    except Exception as e:
        salvaged = salvage_distiller_json(raw_output)
        if salvaged and salvaged.get("updates"):
            logger.warning(
                "Distiller output truncated/malformed (%s); salvaged %d complete "
                "update(s) from the valid prefix — batch continues with partial set",
                "length-limit" if response.finish_reason == "length" else "parse-error",
                len(salvaged["updates"]),
            )
            parsed = salvaged
        else:
            return {"error": f"Distiller output JSON parse failed: {e}", "raw": raw_output[:500]}

    if not isinstance(parsed, dict) or "updates" not in parsed:
        return {"error": "Distiller output missing 'updates' key", "raw": raw_output[:500]}

    logger.info("Distiller produced %d updates", len(parsed["updates"]))
    return parsed
