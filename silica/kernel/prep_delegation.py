# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Prepare and run Distiller delegation for the Injector pipeline.

This module ports `build_tasks()` from Hermes prep_delegation.py as a pure
function, and adds `run_distiller()` which calls the LLM directly via
`call_llm()` (stateless, single-turn, no tool use).

The protocol template uses {TARGET} as the only substitution. PAYLOAD_PATH
is passed as a file reference in the task context, not inlined into the prompt.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import typing
from pathlib import Path

logger = logging.getLogger(__name__)

# Distiller prompt template — vendored at install time
_PROMPT_PATH = Path(__file__).resolve().parent.parent / "capabilities" / "prompts" / "distiller_prompt.txt"
# Shared anti-slop fragment, appended to every body-writing prompt (refine/enrich too).
_ANTI_SLOP_PATH = _PROMPT_PATH.parent / "_anti_slop.txt"


def _load_prompt() -> str:
    if not _PROMPT_PATH.exists():
        raise FileNotFoundError(f"Distiller prompt not found: {_PROMPT_PATH}")
    return _PROMPT_PATH.read_text(encoding="utf-8")


def render_prompt(target: str, hub: str | None = None, source_text: str = "",
                  session_date: str = "", language: str | None = None) -> str:
    """Render the distiller prompt with TARGET/LANGUAGE/MAX_TAGS substitution.

    `session_date` (F2a): the date the SOURCE session/document happened — not
    necessarily today (eval passes simulated time; dated documents pass their
    own date). Empty ⇒ "unknown", and the prompt rule keeps source wording.

    MAX_TAGS comes from the active vault's `conventions:` block
    (silica/kernel/vault_manifest.py) — single source shared with
    `ofm.ofm_lint`'s max-tags check. A vault without a manifest gets today's
    default (3), so this is bit-identical when unconfigured.

    LANGUAGE precedence: explicit `language` arg > declared
    `conventions.language` > per-call detection from `source_text`. An explicit
    arg is pinned once per file at PAYLOAD so the rendered template is
    byte-identical across a file's chunks/steer retries (cache-stable prefix);
    `conventions.language` unset (None) means "follow the source document's
    language" — detected from `source_text` (capped to 4000 chars, enough
    signal without scanning whole PDFs). A declared `conventions.language` is
    translation intent and always wins over detection. The {LANGUAGE}
    placeholder always receives a concrete language name ("Italian",
    "English", ...), never None.
    """
    from silica.kernel import language as lang_mod
    from silica.kernel.vault_manifest import get_active_manifest

    body = _load_prompt()
    body = body.replace("{TARGET}", target)
    if hub:
        body = body.replace("{HUB_NAME}", hub)
    conventions = get_active_manifest().conventions
    # Cache-stable prefix: an explicit `language` (pinned once per file at
    # PAYLOAD) wins over per-call detection, so the rendered template is
    # byte-identical across all chunks and steer retries of a file.
    lang_name = language or conventions.language or lang_mod.display_name(
        lang_mod.detect(source_text[:4000])
    )
    body = body.replace("{LANGUAGE}", lang_name)
    body = body.replace("{MAX_TAGS}", str(conventions.max_tags))
    body = body.replace("{SESSION_DATE}", session_date.strip() or "unknown")
    # F1b: vault-declared capture rules. Empty ⇒ the placeholder line vanishes
    # entirely (consume its trailing newline), so an unconfigured vault renders
    # bit-identically to before this existed.
    rules = (conventions.capture_rules or "").strip()
    body = body.replace(
        "{CAPTURE_RULES}\n",
        f"## Vault capture rules\n{rules}\n\n" if rules else "",
    )
    if _ANTI_SLOP_PATH.exists():  # ponytail: optional fragment, missing file must not break nucleation
        body += "\n\n" + _ANTI_SLOP_PATH.read_text(encoding="utf-8")
    return body


def _payload_sample_text(payload: dict, limit: int = 4000) -> str:
    """Concatenate inbox excerpts from the payload as a source-language sample.

    Used only when `conventions.language` is unset — detecting the dominant
    language of the batch's own inbox content is cheap and enough signal for
    the {LANGUAGE} placeholder; capped early so we never build a huge string
    for a single detect() call.
    """
    parts: list[str] = []
    total = 0
    for batch in payload.get("batches", []):
        for concept in batch.get("concepts", []):
            excerpt = concept.get("inbox_excerpt") if isinstance(concept, dict) else None
            if excerpt:
                parts.append(excerpt)
                total += len(excerpt)
                if total >= limit:
                    return "\n".join(parts)[:limit]
    return "\n".join(parts)


# Per-string cap when echoing a rejected op back to the model: enough to
# recognize the op, not enough to blow up the prompt with full note bodies.
_STEER_ECHO_MAX_CHARS = 280


def _truncate_op_echo(obj, limit: int = _STEER_ECHO_MAX_CHARS):
    """Deep-copy `obj` with long strings truncated and empty fields dropped."""
    if isinstance(obj, str):
        return obj if len(obj) <= limit else obj[:limit] + f"… [truncated, {len(obj)} chars total]"
    if isinstance(obj, dict):
        return {k: _truncate_op_echo(v, limit) for k, v in obj.items()
                if v is not None and v != "" and v != []}
    if isinstance(obj, list):
        return [_truncate_op_echo(v, limit) for v in obj]
    return obj


def render_steer_feedback(
    rejected: list[dict],
    *,
    attempt: int,
    max_attempts: int,
    accepted: list[dict] | None = None,
    partial: bool = False,
    ungrounded: list[dict] | None = None,
) -> str:
    """Structured per-op steering feedback for a re-delegation attempt.

    Paper-aligned (PDDL-INSTRUCT): the corrective prompt echoes the previous
    output with a per-op verdict and the validator's detailed reason, instead
    of a flat concatenation of reasons — detailed feedback measurably beats
    binary feedback in verifier-guided refinement loops.

    Args:
        rejected: validator rejection entries, each `{"op": {...}, "reason": str}`
            (entries missing `op` degrade to a reason-only line).
        attempt/max_attempts: steer-arc position, shown in the header.
        accepted: validated op dicts from the same pass (partial steer) —
            listed so the model does not re-emit them.
        partial: True when the payload was filtered to the rejected concepts.
        ungrounded: span-grounding findings on ACCEPTED ops (`{"heading",
            "path", "spans"}`) — advisory only, the gate stays warn-only; the
            retry must not introduce more content untraceable to the payload.
    """
    accepted = accepted or []
    lines = [
        f"## STEERING CORRECTION (attempt {attempt}/{max_attempts})",
        f"Your previous output was validated: {len(accepted)} op(s) ACCEPTED, "
        f"{len(rejected)} op(s) REJECTED.",
    ]
    if partial:
        lines.append(
            "The payload in this message now contains ONLY the concepts whose ops "
            "were rejected; the accepted ops are already being written."
        )
    else:
        lines.append("Regenerate the full output, fixing every rejected op below.")

    if accepted:
        lines.append("\n### Accepted ops (do NOT re-emit these)")
        for op in accepted:
            if isinstance(op, dict):
                lines.append(f"- [{op.get('op', '?')}] {op.get('path') or op.get('heading', '?')}")

    for i, r in enumerate(rejected, 1):
        if not isinstance(r, dict):
            continue
        op = r.get("op") if isinstance(r.get("op"), dict) else None
        label = f"[{op.get('op', '?')}] \"{op.get('title') or op.get('heading', '?')}\"" if op else "(op not recorded)"
        lines.append(f"\n### Rejected op {i} — {label}")
        lines.append(f"Verdict: REJECTED — {r.get('reason') or 'no reason recorded'}")
        if op:
            lines.append("Your op was:")
            lines.append("```json")
            lines.append(json.dumps(_truncate_op_echo(op), ensure_ascii=False, indent=2))
            lines.append("```")

    if ungrounded:
        lines.append(
            "\n### Grounding warnings (accepted, but fix the habit)\n"
            "These accepted ops contain spans NOT traceable to any payload excerpt. "
            "They were not rejected, but your corrected ops must only carry facts "
            "grounded in inbox_excerpt:"
        )
        for u in ungrounded:
            if not isinstance(u, dict):
                continue
            spans = " | ".join(s[:80] for s in u.get("spans", [])[:3])
            lines.append(f"- \"{u.get('heading', '?')}\" ({u.get('path', '?')}): {spans}")

    lines.append(
        "\n### Instructions\n"
        "For EVERY rejected op above, re-emit a corrected op that fixes exactly "
        "the violated constraint stated in its verdict. Do not introduce new "
        "concepts and do not re-emit accepted ops."
    )
    return "\n".join(lines)


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


def _call_with_deadline(fn, seconds: float):
    """Run fn() under a wall-clock deadline; raise TimeoutError past it.

    The transport read-timeout cannot bound a hung distiller call: OpenRouter
    trickles keep-alive bytes while "processing", and every byte resets httpx's
    per-chunk read timer, so a dead upstream holds the socket open forever.
    Only real elapsed time is trustworthy here.
    """
    # ponytail: daemon thread leaks on timeout (dies with socket/process);
    # acceptable for a single-turn call, revisit if calls pile up.
    box: dict = {}

    def _run():
        try:
            box["value"] = fn()
        except Exception as e:
            box["error"] = e

    t = threading.Thread(target=_run, daemon=True, name="distiller-call")
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise TimeoutError(f"distiller call exceeded {seconds:.0f}s wall-clock deadline")
    if "error" in box:
        raise box["error"]
    return box["value"]


def run_distiller(
    payload: dict,
    target: str,
    hub: str | None = None,
    ledger_digest: str | None = None,
    steer_context: str | None = None,
    substrate: str | None = None,
    session_date: str = "",
    language: str | None = None,
    escalate: bool = False,
) -> dict:
    """Call the Distiller LLM (single-turn) for one payload chunk.

    Args:
        payload: the payload dict (schema_version + batches)
        target: vault-relative target directory for new notes
        hub: optional [[Hub]] note name
        ledger_digest: compact run summary injected as context header (Phase 2)
        steer_context: corrective steering note injected when re-attempting after
            rejection (Phase 6). States why the previous output was rejected.
        escalate: route this call to the escalation model (steer retries; Tier 2 cascade).

    Returns:
        parsed dict with {"updates": [...]} or {"error": ...}
    """
    from silica.agent.providers import get_provider
    from silica.config import CONFIG
    from silica.kernel.context_builder import build_context
    from silica.kernel.ops import DistillerOutput
    from silica.kernel.sanitize import parse_json

    prompt_text = render_prompt(target=target, hub=hub,
                                source_text=_payload_sample_text(payload),
                                session_date=session_date, language=language)
    # Assemble context through the context assembler (Phase 2 rails).
    # Only ledger_digest + the checkpoint payload reach the model — no other
    # vault content is forwarded here.
    ctx = build_context(
        checkpoint_id="distill",
        payload=payload,
        ledger_digest=ledger_digest,
        substrate=substrate,
    )

    # steer_context arrives fully rendered (render_steer_feedback), header included.
    steer_section = f"\n\n{steer_context}\n" if steer_context else ""
    ctx_text = f"---\n{ctx}"
    # Budget arithmetic runs on the same concatenation as the old single-message
    # prompt, so token sizing is unchanged by the split.
    budget_text = f"{prompt_text}\n\n{ctx_text}{steer_section}"

    # Cache-stable layout ("Don't Break the Cache", arXiv 2601.06007): the
    # per-file-stable template is a system block with a cache_control marker;
    # dynamic content (ctx) is user part 1 with a second marker so steer
    # retries reuse template+ctx prefill; steer is an appended trailing part.
    # Non-caching upstreams ignore the markers harmlessly.
    user_parts: list[dict[str, typing.Any]] = [
        {"type": "text", "text": ctx_text, "cache_control": {"type": "ephemeral"}}
    ]
    if steer_section:
        user_parts.append({"type": "text", "text": steer_section})
    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": prompt_text, "cache_control": {"type": "ephemeral"}}
        ]},
        {"role": "user", "content": user_parts},
    ]

    logger.info("Calling Distiller LLM%s", " (escalated)" if escalate else "")

    # #2: size the output budget to the real prompt + model context window
    # instead of a fixed ceiling. Window and output cap come from the live
    # provider (LM Studio /api/v0/models, OpenRouter /api/v1/models);
    # MODEL_CONTEXT_WINDOW / DISTILLER_MAX_TOKENS stay as explicit operator
    # overrides, 262144 as the last-resort default when the provider is
    # unreachable or the model unmapped.
    context_window = int(os.getenv("MODEL_CONTEXT_WINDOW", "0"))
    ceiling = int(os.getenv("DISTILLER_MAX_TOKENS", "0"))
    if not context_window or not ceiling:
        from silica.agent.providers import model_limits
        # Same fallback chain as get_provider for the active role.
        if escalate:
            w_provider, w_model = (CONFIG.distill_escalation_provider,
                                   CONFIG.distill_escalation_model)
        else:
            w_provider, w_model = CONFIG.worker_provider, CONFIG.worker_model
        if not w_provider or not w_model:
            w_provider, w_model = CONFIG.provider, CONFIG.model
        window, out_cap = model_limits(w_provider, w_model)
        context_window = context_window or window or 262144
        ceiling = ceiling or out_cap
    safety_margin = int(os.getenv("DISTILLER_TOKEN_SAFETY_MARGIN", "2048"))
    max_tokens = compute_distiller_max_tokens(
        budget_text,
        context_window=context_window,
        safety_margin=safety_margin,
        ceiling=ceiling,
    )
    logger.info(
        "Distiller output budget: %d tokens (window=%d, prompt≈%d, margin=%d, ceiling=%s)",
        max_tokens, context_window, estimate_prompt_tokens(budget_text), safety_margin,
        ceiling if ceiling > 0 else "none",
    )

    deadline = float(os.getenv("DISTILLER_TIMEOUT", "300"))
    try:
        provider = get_provider(CONFIG, role="escalation" if escalate else "worker")
        response = _call_with_deadline(lambda: provider.call_llm(
            messages=messages,
            tools=None,
            response_schema=DistillerOutput,
            max_tokens=max_tokens,
            # The distiller pin is tied to the worker model's provider routing;
            # an escalated call must not inherit it.
            openrouter_provider=None if escalate else CONFIG.openrouter_provider_distiller,
        ), deadline)
    except Exception as e:
        logger.warning("Distiller provider call failed, falling back to litellm: %s", e)
        from silica.agent.llm import call_llm
        response = _call_with_deadline(lambda: call_llm(
            model=(CONFIG.distill_escalation_model or CONFIG.model) if escalate else CONFIG.model,
            messages=messages,
            tools=None,
            max_tokens=max_tokens,
            response_format=DistillerOutput,
            openrouter_provider=None if escalate else CONFIG.openrouter_provider_distiller,
        ), deadline)

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
