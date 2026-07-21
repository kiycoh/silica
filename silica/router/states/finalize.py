# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Injector terminal states: LINT (gate), CLEANUP, ROLLBACK.

Handler bodies for InjectorFSM, extracted from orchestrator.py: each function
takes the FSM instance and mutates its context/state exactly as the former
method did. Patchable collaborators (DRIVER, CONFIG, tools, load_ops, time)
are resolved through the orchestrator module namespace (orch.X) so tests that
patch silica.router.orchestrator.* keep working.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from silica.router import orchestrator as orch

if TYPE_CHECKING:
    from silica.router.orchestrator import InjectorFSM

logger = logging.getLogger(__name__)


from silica.kernel.ops import OpType


def handle_lint(fsm: "InjectorFSM") -> None:
    fsm._progress_note(fsm._chunk_task_id("lint"), "lint", "running")
    try:
        ops = orch.load_ops(fsm._chunk_ctx["ops_path"])
    except Exception as e:
        raise RuntimeError(f"LINT: failed to read ops: {e}")

    touched = [
        (op.touched_ref(), op.op.value if op.op else "", op.hub or "")
        for op in ops
        if op.touched_ref() and op.op not in (OpType.delete, OpType.skip)
    ]

    for path, op_type, hub in touched:
        res = orch.silica_lint(path, op_type=op_type or "", hub=hub or "")
        if orch.CONFIG.verbose:
            logger.info(
                "[DEBUG LINT Gate]: File: %s | Type: %s | Hub: %s | Success: %s | Errors: %s",
                path,
                op_type,
                hub,
                res["success"],
                res.get("errors", []),
            )
        if not res["success"]:
            fsm._chunk_ctx["abort_reason"] = f"Lint failed for {path}: {res['errors']}"
            fsm._progress_note(fsm._chunk_task_id("lint"), "lint", "failed", error=fsm._chunk_ctx["abort_reason"])
            fsm.state = orch.InjectorState.ROLLBACK
            return

    # S3.2: Run graph-diff check
    regression_rule = fsm._get_recipe_gate("graph_regression", "forbid_new_orphans")
    if regression_rule != "allow":
        if fsm._pre_graph is None:
            fsm._chunk_ctx["abort_reason"] = "Graph regression gate failed: pre-write snapshot is missing"
            fsm._progress_note(fsm._chunk_task_id("lint"), "lint", "failed", error=fsm._chunk_ctx["abort_reason"])
            fsm.state = orch.InjectorState.ROLLBACK
            return
        try:
            from silica.driver.base import NoteRef
            snapshot_domain_dicts = fsm._chunk_ctx.get("snapshot_domain", [])
            if snapshot_domain_dicts:
                snapshot_domain = [NoteRef(**d) for d in snapshot_domain_dicts]
            else:
                # Fallback to touched refs if snapshot_domain is missing
                snapshot_domain = []
                for op in ops:
                    path = op.touched_ref()
                    if path:
                        name = os.path.splitext(os.path.basename(path))[0]
                        snapshot_domain.append(NoteRef(name=name, path=path))
            
            post_graph = orch.DRIVER.graph_snapshot(snapshot_domain)
            from silica.kernel.graph_diff import check_graph_regression
            
            created_paths = fsm._txn.created_paths if fsm._txn else []
            # Fold only chunks appended since the last LINT into the run-scoped stem
            # union (B8: was an O(chunks × concepts) rescan on every LINT). A chunk
            # collision-collapses after it is first folded, but its pre-collapse stems
            # only widen the allowed set — never manufacture a false regression.
            try:
                from silica.kernel.templates import slugify
                chunks = getattr(fsm, "_chunks", [])
                for chunk in chunks[fsm._run_concept_stems_n:]:
                    for batch in chunk.get("batches", []):
                        for concept in batch.get("concepts", []):
                            name = concept.get("name")
                            if name:
                                stem = os.path.splitext(os.path.basename(name))[0].lower()
                                fsm._run_concept_stems.add(stem)
                                fsm._run_concept_stems.add(slugify(stem))
                fsm._run_concept_stems_n = len(chunks)
            except Exception as _ce:
                logger.debug("Failed to extract run concept stems for graph check: %s", _ce)

            deferred_stems = set(fsm._chunk_ctx.get("deferred_stems", []))
            deferred_stems |= fsm._run_concept_stems

            success, errors = check_graph_regression(
                fsm._pre_graph, post_graph, created_paths, frozenset(deferred_stems)
            )

            if orch.CONFIG.verbose:
                logger.info(
                    "[DEBUG Graph Regression Gate]: Pre-write graph size: %d nodes | Post-write graph size: %d nodes | Rule: %s | Result: %s",
                    len(fsm._pre_graph.link_counts) if fsm._pre_graph and fsm._pre_graph.link_counts else 0,
                    len(post_graph.link_counts) if post_graph and post_graph.link_counts else 0,
                    regression_rule,
                    "PASSED" if success else f"FAILED: {errors}"
                )

            if not success:
                orphan_errors = [e for e in errors if e.startswith("Unplanned orphans")]
                drift_errors = [e for e in errors if e.startswith("Backlink drift")]
                blocking_errors = [
                    e for e in errors
                    if not e.startswith("Unplanned orphans") and not e.startswith("Backlink drift")
                ]
                if drift_errors:
                    logger.warning(
                        "[Graph Regression Gate]: Backlink drift (non-blocking): %s",
                        "; ".join(drift_errors),
                    )
                if orphan_errors:
                    logger.warning(
                        "[Graph Regression Gate]: Orphan warning (non-blocking): %s",
                        "; ".join(orphan_errors),
                    )
                    # Record run-created notes that ended this chunk orphaned.
                    # Acted on (if still orphaned) at end of run, not now —
                    # AUTOLINK/BACKLINK or a later chunk may yet connect them.
                    if fsm.warning_ledger is not None:
                        try:
                            from silica.kernel.graph_diff import normalize_ref
                            post_orphan_keys = {normalize_ref(r) for r in post_graph.orphans}
                            detail = "; ".join(orphan_errors)
                            for op in ops:
                                p = op.touched_ref()
                                if not p or op.op != OpType.write:
                                    continue
                                name = os.path.splitext(os.path.basename(p))[0]
                                if normalize_ref(NoteRef(name=name, path=p)) in post_orphan_keys:
                                    fsm.warning_ledger.add(p, "orphan", detail)
                        except Exception as _we:
                            logger.debug("orphan warning record failed (non-fatal): %s", _we)
                if blocking_errors:
                    reason = f"Graph regression gate failed: {'; '.join(blocking_errors)}"
                    logger.warning("[Graph Regression Gate]: Blocking errors (triggering rollback): %s", "; ".join(blocking_errors))
                    fsm._chunk_ctx["abort_reason"] = reason
                    fsm._progress_note(fsm._chunk_task_id("lint"), "lint", "failed", error=reason)
                    fsm.state = orch.InjectorState.ROLLBACK
                    return
        except Exception as e:
            logger.error("Failed to perform graph-diff check: %s", e)
            fsm._chunk_ctx["abort_reason"] = f"Graph regression gate error during check: {e}"
            fsm._progress_note(fsm._chunk_task_id("lint"), "lint", "failed", error=fsm._chunk_ctx["abort_reason"])
            fsm.state = orch.InjectorState.ROLLBACK
            return

    fsm._progress_note(fsm._chunk_task_id("lint"), "lint", "done")
    fsm._transition_success()


def _log_nucleate_completion(fsm: "InjectorFSM", fi: int, source_file: str) -> None:
    """Append one line to the vault's human journal (log.md).

    Pure projection of state WRITE/VALIDATE already recorded — the manifest
    (new/patch counts) and the deferred store (deferred count) — onto the
    log.md line shape. No new computation. Idempotent per (run_id, source
    file): a multi-file run shares one run_id and fires this once per file,
    so each file needs its own line, while a resume of the same run must not
    duplicate any (dedup_key). Best-effort and must never block CLEANUP.
    """
    try:
        from silica.kernel.run_log import append_log_line, format_nucleate_event

        basename = os.path.basename(source_file)
        new_count = sum(
            1 for e in fsm.manifest.entries
            if e.source_basename == basename and e.op == "write"
        )
        patch_count = sum(
            1 for e in fsm.manifest.entries
            if e.source_basename == basename and e.op == "patch"
        )

        deferred_count = 0
        content_hashes = getattr(fsm, "_file_content_hashes", [])
        if fi < len(content_hashes):
            try:
                from silica.kernel.deferred import get_deferred_store
                bundle = get_deferred_store().get(content_hashes[fi])
                if bundle:
                    deferred_count = len(bundle.get("rejected_ops", []))
            except Exception as _de:
                logger.debug("CLEANUP: deferred count lookup failed (non-fatal): %s", _de)

        event = format_nucleate_event(basename, new_count, patch_count, deferred_count)
        append_log_line(event, fsm.progress.run_id, dedup_key=f"`{basename}`")
    except Exception as exc:
        logger.debug("CLEANUP: log.md append skipped (non-fatal): %s", exc)


def _record_provenance(fsm: "InjectorFSM", fi: int, source_file: str) -> None:
    """Append one `<vault>/provenance.json` record (spec-hermes-coherence §3).

    Sibling projection to _log_nucleate_completion, at the same CLEANUP point:
    reuses fsm._file_content_hashes[fi] — the sha256 already computed once
    per file at RUN start (silica.router.orchestrator.InjectorFSM.run), the
    same value the /nucleate pre-check will later compare against. Recomputing
    it here would fail anyway: by CLEANUP the source file has already been
    archived (moved) out of its original inbox path. `notes` is the
    projection of this run's validated write/patch ops for this source,
    already recorded in fsm.manifest.entries — no new computation. Records
    even when notes is empty: a version change with zero touched notes still
    means every note derived from the prior version is now stale. Best-
    effort and must never block CLEANUP.
    """
    try:
        from silica.kernel.provenance import append_record

        basename = os.path.basename(source_file)
        content_hashes = getattr(fsm, "_file_content_hashes", [])
        sha256 = content_hashes[fi] if fi < len(content_hashes) else ""
        if not sha256:
            return

        notes = sorted({
            e.path for e in fsm.manifest.entries
            if e.source_basename == basename and e.op in ("write", "patch")
        })

        append_record(basename, sha256, fsm.progress.run_id, notes)
    except Exception as exc:
        logger.debug("CLEANUP: provenance append skipped (non-fatal): %s", exc)


def handle_cleanup(fsm: "InjectorFSM") -> None:
    from silica.tools.wrapped import silica_cleanup

    fsm._get_chunks_from_context_if_empty()
    fi, ci = fsm._chunk_flat_to_fi_ci.get(fsm._current_chunk_idx, (0, fsm._current_chunk_idx))
    fsm._progress_note(fsm._chunk_task_id("cleanup"), "cleanup", "running")

    # Always write ledger for this chunk's ops (per chunk)
    fsm._write_ledger_for_file(fi, "committed")

    # Archive the physical file only on the last chunk of its file group
    file_group = fsm._file_chunks.get(fi, {})
    n_chunks_in_file = len(file_group.get("chunks", []))
    is_last_chunk_of_file = (ci + 1 >= n_chunks_in_file)

    if is_last_chunk_of_file:
        # Only archive if no chunk of this file failed
        fi_prefix = f"f{fi}_"
        file_has_failure = any(
            t.status == "failed" for t in fsm.progress.tasks
            if t.id.startswith(fi_prefix)
        )
        if not file_has_failure:
            inbox_file_for_fi = file_group.get("source_file", fsm.inbox_file)
            res = silica_cleanup(inbox_file_for_fi, "done")
            if "error" in res:
                fsm.context["cleanup_warning"] = res["error"]
            _log_nucleate_completion(fsm, fi, inbox_file_for_fi)
            _record_provenance(fsm, fi, inbox_file_for_fi)
            # Title-index run cache: the archived source moved out of its
            # indexed path — drop its ref so AUTOLINK can't link to a stale
            # inbox title.
            _cached_refs = getattr(fsm, "_run_title_refs", None)
            if _cached_refs is not None:
                _src_abs = os.path.abspath(inbox_file_for_fi)
                _cached_refs[:] = [
                    r for r in _cached_refs
                    if not getattr(r, "path", None) or os.path.abspath(r.path) != _src_abs
                ]
        else:
            logger.info(
                "File %d (%s) had chunk failures — not archiving.",
                fi, file_group.get("source_file", "?"),
            )
    else:
        logger.info(
            "Chunk f%d_c%d done. Archiving deferred until last chunk of file %d.",
            fi, ci, fi,
        )

    if fsm.context.get("final_status") != "no_ops":
        if fsm.context.get("has_partial_failure"):
            fsm.context["final_status"] = "partial"
        else:
            fsm.context["final_status"] = "Success"

    # Persist this run's inverses for /revert, with final content hash.
    if fsm._undo_run_id and fsm._run_inverses:
        import hashlib
        from silica.kernel.ops import InverseOpKind
        from silica.kernel.undo_journal import get_undo_journal
        journal = get_undo_journal()
        for path, inv, _ in fsm._run_inverses:
            try:
                post = orch.DRIVER.read_note(path).content
                post_hash = hashlib.sha256((post or "").encode("utf-8")).hexdigest()
            except Exception:
                post_hash = None
                if inv.kind != InverseOpKind.recreate_deleted:
                    # Note should exist after this write; without its hash the
                    # /revert "modified since inject" guard can't protect it.
                    logger.warning(
                        "finalize: could not hash %s post-write; /revert guard "
                        "disabled for it", path)
            journal.record(fsm._undo_run_id, inv, post_hash)
        fsm._run_inverses.clear()

    fsm._progress_note(fsm._chunk_task_id("cleanup"), "cleanup", "done")
    fsm._transition_success()


def handle_rollback(fsm: "InjectorFSM") -> None:
    fsm._progress_note("rollback", "rollback", "running")
    snapshot_res = fsm._chunk_ctx.get("snapshot", {})
    # fsm._txn.inverses is the single source of truth for rollback (C3 /
    # ADR-009): SNAPSHOT seeds it and every phase that mutates a pre-existing
    # note appends to it. Fall back to the persisted snapshot dict only when
    # no live transaction exists (defensive — both share the per-chunk lifetime).
    if fsm._txn is not None:
        txn_id = fsm._txn.id
        inverses = fsm._txn.inverses_serialized
    else:
        txn_id = snapshot_res.get("txn_id")
        inverses = snapshot_res.get("inverses", [])

    if txn_id and inverses:
        from silica.tools.wrapped import silica_restore
        try:
            res = silica_restore(txn_id=txn_id, inverses=inverses)
            if not res.get("success", False):
                err_msg = "; ".join(res.get("errors", []))
                logger.error("Rollback partially failed: %s", err_msg)
                fsm.context["rollback_error"] = err_msg
            else:
                logger.info("Rollback complete for txn %s", txn_id)
        except Exception as e:
            logger.error("Rollback failed: %s", e)
            fsm.context["rollback_error"] = str(e)
        fsm._write_ledger_rollback(txn_id)

    # Clean up the embedding index for notes that were created and then rolled back
    # to prevent stale phantom entries that would bias future candidate searches.
    created_paths: list[str] = []
    if fsm._txn is not None and fsm._txn.created_paths:
        created_paths = list(fsm._txn.created_paths)
    elif snapshot_res.get("created_paths"):
        created_paths = list(snapshot_res["created_paths"])
    if created_paths:
        try:
            from silica.kernel.embed import get_store
            store = get_store()
            for cp in created_paths:
                store.delete(cp.removesuffix(".md"))
            store.save()
        except Exception as _ee:
            logger.debug("ROLLBACK: embed index cleanup failed (non-fatal): %s", _ee)

    fsm._progress_note("rollback", "rollback", "done")
    # Contain the failure at chunk level instead of aborting the whole run
    fsm._contain_chunk_failure()
