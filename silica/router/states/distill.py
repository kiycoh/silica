# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Injector distillation states: DELEGATE, SANITIZE, VALIDATE.

Handler bodies for InjectorFSM, extracted from orchestrator.py: each function
takes the FSM instance and mutates its context/state exactly as the former
method did. Patchable collaborators (DRIVER, CONFIG, tools, load_ops, time)
are resolved through the orchestrator module namespace (orch.X) so tests that
patch silica.router.orchestrator.* keep working.
"""
from __future__ import annotations

import logging
import os
import re
import hashlib
from typing import TYPE_CHECKING

from silica.router import orchestrator as orch

if TYPE_CHECKING:
    from silica.router.orchestrator import InjectorFSM

logger = logging.getLogger(__name__)

# Emitted by the C3 title gate in validate_operations (band 2).
_NEAR_TITLE_RE = re.compile(r"near_title candidate='([^']*)' path='([^']*)'")


def _enqueue_near_title_dedups(fsm: "InjectorFSM", rejected_raw: list) -> None:
    """Fuzzy-band title rejections become live dedup WorkItems (C3 reuses C2).

    The op is already parked in the deferred bundle by _defer_ops; this hands
    the same pair to the dedup judge so the verdict is routed in-run — retry
    stays the exception. Best-effort: no queue (ad-hoc validate) → no-op.
    """
    wq = getattr(fsm, "work_queue", None)
    if wq is None:
        return
    from silica.kernel.workqueue import WorkItem

    for r in rejected_raw:
        if not isinstance(r, dict):
            continue
        m = _NEAR_TITLE_RE.search(r.get("reason", "") or "")
        if not m:
            continue
        op = r.get("op", {}) or {}
        try:
            wq.enqueue(WorkItem(
                kind="dedup",
                target_path=m.group(2),
                context={
                    "concept": op.get("heading", ""),
                    "excerpt": op.get("snippet", ""),
                    "candidate": m.group(1),
                    "score": 0.0,
                    "inbox_file": fsm.inbox_file,
                    "hub": fsm.hub,
                    "content_hash": fsm._current_content_hash,
                    "target_dir": fsm.target_dir,
                },
                reason=r.get("reason", "near_title"),
            ))
        except Exception as _qe:
            logger.debug("VALIDATE: failed to enqueue near_title dedup item: %s", _qe)


def _enqueue_short_snippet_expands(fsm: "InjectorFSM", rejected_raw: list) -> None:
    """«snippet too short» rejections become live expand WorkItems.

    The op is already parked in the deferred bundle by _defer_ops; the expand
    worker re-prompts the LLM with the concept's inbox excerpt (max 2 attempts)
    and commits through the same gate — retry stays the exception. Called only
    for PARTIAL rejections: an all-rejected chunk re-delegates via the steer
    arc, and racing both would author the same notes twice. Best-effort: no
    queue (ad-hoc validate) → no-op.
    """
    wq = getattr(fsm, "work_queue", None)
    if wq is None:
        return
    from silica.kernel.workqueue import WorkItem

    excerpts: dict[str, str] = {}
    try:
        chunk = fsm._chunks[fsm._current_chunk_idx]
        for batch in chunk.get("batches", []):
            for c in batch.get("concepts", []):
                if isinstance(c, dict) and c.get("name"):
                    excerpts[c["name"]] = c.get("inbox_excerpt", "") or ""
    except Exception as _ce:
        logger.debug("EXPAND: excerpt lookup failed (items will skip): %s", _ce)

    for r in rejected_raw:
        if not isinstance(r, dict):
            continue
        reason = r.get("reason", "") or ""
        if not reason.startswith("snippet too short"):
            continue
        op = r.get("op", {}) or {}
        try:
            wq.enqueue(WorkItem(
                kind="expand",
                target_path=op.get("path", ""),
                context={
                    "op": op,
                    "excerpt": excerpts.get(op.get("heading", ""), ""),
                    "reason": reason,
                    "inbox_file": fsm.inbox_file,
                    "hub": fsm.hub,
                    "content_hash": fsm._current_content_hash,
                    "target_dir": fsm.target_dir,
                },
                reason=reason,
            ))
        except Exception as _qe:
            logger.debug("VALIDATE: failed to enqueue expand item: %s", _qe)


def _inject_graph_ctx(chunk: dict, vault_ctx: dict) -> dict:
    """Return a shallow-enriched copy of chunk with graph_context added to concepts.

    For every concept that has a vault_collision, adds a graph_context field:
        {"cluster_id": int, "hub": str|None, "is_hub": bool}
    Concepts without a vault_collision get graph_context=null.
    The original chunk dict is never mutated.
    """
    if not vault_ctx:
        return chunk

    enriched_batches = []
    for batch in chunk.get("batches", []):
        enriched_concepts = []
        for concept in batch.get("concepts", []):
            if not isinstance(concept, dict):
                enriched_concepts.append(concept)
                continue
            collision = concept.get("vault_collision") or {}
            cpath = collision.get("path", "").removesuffix(".md") if collision else ""
            gctx = vault_ctx.get(cpath)
            graph_context = (
                {
                    "cluster_id": gctx["cluster_id"],
                    "hub": gctx["hub"],
                    "is_hub": gctx["is_hub"],
                }
                if gctx and cpath
                else None
            )
            enriched_concepts.append({**concept, "graph_context": graph_context})
        enriched_batches.append({**batch, "concepts": enriched_concepts})
    return {**chunk, "batches": enriched_batches}


def handle_delegate(fsm: "InjectorFSM") -> None:
    from silica.kernel.prep_delegation import run_distiller

    fsm._get_chunks_from_context_if_empty()

    if not fsm._chunks or fsm._current_chunk_idx >= len(fsm._chunks):
        raise RuntimeError("No chunks available for iterative processing.")

    current_chunk = fsm._chunks[fsm._current_chunk_idx]
    idx = fsm._current_chunk_idx

    # Content-addressed idempotency (Phase 2): if this chunk was already
    # processed in a prior run with identical input, skip DELEGATE→SANITIZE→VALIDATE
    # and reuse the persisted knowledge-block ops file.
    #
    # Use the pre-COLLISION hash stored by _handle_collision so the key is
    # based on the original source input, not the vault-state-dependent
    # post-COLLISION chunk (which changes when resumed after a partial run).
    import json as _json
    chunk_hash = fsm.context.get(f"chunk_{idx}_input_hash") or hashlib.sha256(
        _json.dumps(current_chunk, sort_keys=True).encode()
    ).hexdigest()
    saved_ops_path = fsm.progress.is_checkpoint_done(fsm._chunk_task_id("validate"), chunk_hash)
    if saved_ops_path and os.path.exists(saved_ops_path):
        logger.info(
            "DELEGATE chunk %d: content-addressed hit (hash=%s…) — skipping to SNAPSHOT",
            idx,
            chunk_hash[:8],
        )
        fsm.context["ops_path"] = saved_ops_path
        fsm.state = orch.InjectorState.SNAPSHOT
        return

    logger.info(f"--- DISTILLING BATCH {idx + 1}/{len(fsm._chunks)} ---")
    fsm._progress_note(fsm._chunk_task_id("distill"), "distill", "running")

    # Assemble compact ledger digest for LLM context (Phase 2 rails).
    # Include the RunManifest so the distiller knows what was injected in prior chunks.
    ledger_digest: str | None = None
    try:
        ledger_digest = fsm.progress.digest(manifest=fsm.manifest)
    except Exception:
        pass

    # Phase 6: pass steering correction if VALIDATE sent us back here
    steer_context: str | None = fsm.context.get(f"chunk_{idx}_steer_context")
    if steer_context:
        logger.info("DELEGATE chunk %d: re-attempt with steering correction", idx)

    # Enrich the payload with graph context (cluster/hub/is_hub) for concepts
    # that have a vault_collision.  The distiller uses this to understand
    # structural importance of the matched note.  Original chunk is not modified.
    vault_ctx = fsm.context.get("vault_graph_ctx", {})
    enriched_chunk = _inject_graph_ctx(current_chunk, vault_ctx) if vault_ctx else current_chunk

    # Build per-chunk substrate: semantically close vault notes that are not
    # yet directly linked to run notes — candidates for `parent` and wikilinks.
    # Also surface any cleared parent forward-references from earlier chunks.
    substrate: str | None = None
    try:
        from silica.kernel.run_substrate import build_substrate
        substrate = build_substrate(
            enriched_chunk,
            manifest_titles=fsm.manifest.titles(),
            cleared_parents=fsm.context.get("run_cleared_parents"),
            hub_names=[
                v["hub"].rsplit("/", 1)[-1] for v in vault_ctx.values()
                if isinstance(v, dict) and v.get("is_hub") and v.get("hub")
            ],
        )
    except Exception as _sub_e:
        logger.debug("DELEGATE: substrate build failed (non-fatal): %s", _sub_e)

    try:
        chunk_result = run_distiller(
            payload=enriched_chunk,
            target=fsm.target_dir,
            hub=fsm.hub,
            ledger_digest=ledger_digest,
            steer_context=steer_context,
            substrate=substrate,
        )
        if "error" in chunk_result:
            fsm._progress_note(fsm._chunk_task_id("distill"), "distill", "failed", error=chunk_result["error"])
            raise RuntimeError(f"Distiller error on batch {idx}: {chunk_result['error']}")

        # Episodic lane: route personal/ephemeral facts to the short-term
        # store. Never fails the ingest (capture_from_distill swallows).
        from silica.kernel.episodic import capture_from_distill
        capture_from_distill(
            chunk_result,
            run_id=fsm.progress.run_id,
            seen=fsm.progress.started_at[:10],
        )

        distiller_path = fsm._make_tmp(chunk_result)
        fsm._chunk_ctx["distiller_output_path"] = distiller_path
        # Store chunk hash for knowledge-block write at VALIDATE
        fsm.context[f"chunk_{idx}_hash"] = chunk_hash
        fsm._progress_note(fsm._chunk_task_id("distill"), "distill", "done", output_ref=distiller_path)
        fsm._transition_success()

    except Exception as e:
        raise RuntimeError(f"Critical failure delegating batch {idx}: {e}")


def handle_sanitize(fsm: "InjectorFSM") -> None:
    idx = fsm._current_chunk_idx
    fsm._progress_note(fsm._chunk_task_id("sanitize"), "sanitize", "running")
    res = orch.silica_sanitize(fsm._chunk_ctx["distiller_output_path"])
    if "error" in res:
        fsm._progress_note(fsm._chunk_task_id("sanitize"), "sanitize", "failed", error=res["error"])
        raise RuntimeError(f"Sanitize failed: {res['error']}")
    fsm._chunk_ctx["sanitized"] = res
    fsm._progress_note(fsm._chunk_task_id("sanitize"), "sanitize", "done")
    fsm._transition_success()


def handle_validate(fsm: "InjectorFSM") -> None:
    idx = fsm._current_chunk_idx
    fsm._progress_note(fsm._chunk_task_id("validate"), "validate", "running")
    sanitized = fsm._chunk_ctx["sanitized"]["parsed"]
    ops_raw = sanitized.get("updates", sanitized) if isinstance(sanitized, dict) else sanitized
    if not isinstance(ops_raw, list):
        ops_raw = [ops_raw]

    # Merge collision-routed patch ops (Phase 5): prepend so they go through
    # the same validate→snapshot→write path as distiller-generated ops.
    collision_ops = fsm.context.get(f"chunk_{idx}_collision_ops", [])
    if collision_ops:
        ops_raw = list(collision_ops) + list(ops_raw)

    # Cohesion pass: inject sibling cross-references into write ops' related[]
    # before validation so the links land in the written frontmatter.
    # Scope: same-chunk siblings only (cross-chunk handled by AUTOLINK/BACKLINK).
    from silica.kernel.cohesion import cohesion_pass
    ops_raw = cohesion_pass(ops_raw)

    ops_path = fsm._make_tmp(ops_raw)

    fsm._get_chunks_from_context_if_empty()

    payload_paths: list[str] = []
    if fsm._chunks and fsm._current_chunk_idx < len(fsm._chunks):
        payload_paths.append(fsm._make_tmp(fsm._chunks[fsm._current_chunk_idx]))
    else:
        # Fallback to general payload if _chunks is not populated
        payload_data = fsm.context.get("payload", {})
        if "chunks" in payload_data:
            for chunk in payload_data["chunks"]:
                payload_paths.append(fsm._make_tmp(chunk))
        elif "payload" in payload_data:
            payload_paths.append(fsm._make_tmp(payload_data["payload"]))

    res = orch.silica_validate_ops(
        ops_path,
        payload_paths=payload_paths,
        target_dir=fsm.target_dir,
        hub=fsm.hub,
    )

    if "error" in res:
        raise RuntimeError(f"Validate failed: {res['error']}")

    fsm.context["validate"] = res

    # Span-grounding warnings (warn-only, never a rejection) → run ledger,
    # persisted as <run_dir>/warnings.json alongside the orphan warnings.
    # Doubles as calibration data for the gate's thresholds.
    if fsm.warning_ledger is not None:
        for u in res.get("ungrounded", []):
            fsm.warning_ledger.add(
                u.get("path") or "",
                "ungrounded_span",
                f"{u.get('heading', '')}: " + " | ".join(s[:60] for s in u.get("spans", [])),
            )

    max_rate = fsm._get_recipe_gate("rejection_rate_max", 0.10)

    if orch.CONFIG.verbose:
        total_ops = res.get("validated_count", 0) + res.get("rejected_count", 0)
        logger.info(
            "[DEBUG VALIDATE Gate]: Success: %s | Total evaluated ops: %d | Validated (accepted): %d | Rejected: %d | Rejection Rate: %.1f%% (Max Allowed: %.1f%%)",
            res.get("success"),
            total_ops,
            res.get("validated_count", 0),
            res.get("rejected_count", 0),
            res.get("rejection_rate", 0) * 100,
            max_rate * 100,
        )
        for r in res.get("rejected_ops", []):
            op = r.get("op", {}) if isinstance(r, dict) else {}
            label = op.get("path") or op.get("heading") or "?"
            logger.info(
                "VALIDATE reject: [%s] %s — %s",
                op.get("type", "?"), label, r.get("reason", "(no reason given)"),
            )

    if res.get("validated_count", 0) == 0 and res.get("rejected_count", 0) == 0:
        logger.info("VALIDATE: no actionable ops (all skip) — short-circuit to CLEANUP")
        fsm.context["final_status"] = "no_ops"
        fsm._chunk_ctx["ops_path"] = ops_path
        fsm._progress_note(fsm._chunk_task_id("validate"), "validate", "done")
        fsm.state = orch.InjectorState.CLEANUP
        return

    # Persist rejected ops to the deferred store so the model can retry them
    # later without re-running the expensive RECON → DELEGATE cycle.
    rejected_raw = res.get("rejected_ops", [])
    if rejected_raw:
        deferred_ops = [
            r.get("op", r) if isinstance(r, dict) and "op" in r else r
            for r in rejected_raw
        ]
        rejection_reasons = {
            (r.get("op", {}).get("path") or r.get("op", {}).get("heading") or "?"): r.get("reason", "")
            for r in rejected_raw if isinstance(r, dict)
        }
        if fsm._defer_ops(deferred_ops, rejection_reasons, phase="VALIDATE"):
            logger.warning(
                "VALIDATE: %d op(s) rejected and saved to deferred store (hash=%s…). "
                "Use silica_deferred_retry to attempt them later.",
                len(rejected_raw),
                fsm._current_content_hash[:8],
            )
        _enqueue_near_title_dedups(fsm, rejected_raw)
        if res.get("validated_count", 0) > 0:
            # Partial rejection only: with 0 validated the steer arc below
            # re-delegates the whole chunk — expand would race it.
            _enqueue_short_snippet_expands(fsm, rejected_raw)

    # Accumulate cleared parent references across chunks.
    # These are prospective links (parent notes not yet in vault) that the
    # distiller can anticipate in subsequent chunks or future runs.
    cleared = res.get("cleared_parents", [])
    if cleared:
        fsm.context.setdefault("run_cleared_parents", []).extend(cleared)
        logger.debug("VALIDATE: %d parent reference(s) cleared to hub fallback (tracked as forward refs)", len(cleared))

    # Unresolved inline wikilinks are kept verbatim as dangling forward-refs (no
    # rejection) and accumulated so later chunks / future runs can anticipate them.
    cleared_links = res.get("cleared_links", [])
    if cleared_links:
        fsm.context.setdefault("run_cleared_links", []).extend(cleared_links)
        logger.debug("VALIDATE: %d unresolved wikilink(s) kept as forward refs", len(cleared_links))

    rejection_rate = res.get("rejection_rate", 0)
    if rejection_rate >= max_rate:
        logger.warning(
            "VALIDATE: rejection rate %.1f%% exceeds threshold %.1f%% — continuing with %d validated op(s).",
            rejection_rate * 100,
            max_rate * 100,
            res.get("validated_count", 0),
        )

    # Abort only when no validated ops remain — partial success is fine.
    if res.get("validated_count", 0) == 0:
        # Phase 6 steering arc: re-delegate with rejection reason injected (max 2 attempts).
        steer_attempts = fsm.context.get(f"chunk_{idx}_steer_attempts", 0)
        _max_steer = fsm._get_recipe_gate("max_steer_attempts", 2)
        if steer_attempts < _max_steer:
            steer_attempts += 1
            fsm.context[f"chunk_{idx}_steer_attempts"] = steer_attempts
            # Build a short rejection summary to inject as corrective context.
            rejected_raw = res.get("rejected_ops", [])
            reasons = "; ".join(
                r.get("reason", "") for r in rejected_raw if isinstance(r, dict) and r.get("reason")
            )
            steer_msg = (
                f"|attempt={steer_attempts}|"
                f" All {res.get('rejected_count', '?')} ops were rejected."
                f" Reasons: {reasons or 'no reason provided'}."
                f" Produce valid ops that satisfy the pipeline constraints."
            )
            fsm.context[f"chunk_{idx}_steer_context"] = steer_msg
            logger.warning(
                "VALIDATE: steer attempt %d/%d for chunk %d — re-delegating with correction.",
                steer_attempts, _max_steer, idx,
            )
            fsm._progress_note(fsm._chunk_task_id("validate"), "validate", "running",
                                error=f"steer {steer_attempts}/{_max_steer}")
            try:
                fsm.progress.set_status(  # type: ignore[union-attr]
                    fsm._chunk_task_id("distill"), "running",
                    error=f"steer attempt {steer_attempts}"
                )
            except Exception:
                pass
            fsm.state = orch.InjectorState.DELEGATE
            return
        # Exhausted steering budget → defer and short-circuit.
        logger.warning("VALIDATE: steer budget exhausted (%d/%d) — deferring chunk %d.", steer_attempts, _max_steer, idx)
        fsm._chunk_ctx["abort_reason"] = "All ops rejected — nothing to write"
        fsm.context["final_status"] = "no_ops"
        fsm._chunk_ctx["ops_path"] = ops_path
        fsm._progress_note(fsm._chunk_task_id("validate"), "validate", "done")
        fsm.state = orch.InjectorState.CLEANUP
        return

    # Knowledge-block consolidation (Phase 2): persist the validated ops to
    # a stable path in the run directory so they survive tmp cleanup and
    # enable content-addressed skip on re-runs.
    chunk_hash = fsm.context.get(f"chunk_{idx}_hash", "")
    kb_path: str = ops_path  # fallback to tmp if save fails
    if chunk_hash:
        try:
            kb_path = fsm._save_knowledge_block(idx, ops_path)
        except Exception as _kb_e:
            logger.debug("Knowledge-block save failed (non-fatal): %s", _kb_e)

    fsm._chunk_ctx["ops_path"] = kb_path
    fsm._progress_note(
        f"chunk_{fsm._current_chunk_idx}_validate",
        "validate",
        "done",
        output_ref=kb_path,
        content_hash=chunk_hash or None,
    )
    fsm._transition_success()
