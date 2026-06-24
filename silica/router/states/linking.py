"""Injector linking states: AUTOLINK, BACKLINK.

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


def handle_autolink(fsm: "InjectorFSM") -> None:
    """Best-effort wikilink injection into touched notes (Phase 4).

    Runs autolink on every note written by this chunk.  Failures are
    non-fatal: they are logged and the FSM continues to LINT.  This is
    intentional — autolink only ADDs links; it can never break a valid note.
    """
    idx = fsm._current_chunk_idx
    fsm._progress_note(fsm._chunk_task_id("autolink"), "autolink", "running")

    try:
        from silica.kernel.autolink import build_title_index

        ops = orch.load_ops(fsm._chunk_ctx["ops_path"])
        touched_paths = [
            ref
            for op in ops
            if (ref := op.touched_ref()) and op.op not in (OpType.delete, OpType.skip)
        ]

        if not touched_paths:
            fsm._progress_note(fsm._chunk_task_id("autolink"), "autolink", "done")
            fsm._transition_success()
            return

        all_refs = orch.DRIVER.list_files()
        title_index = build_title_index(all_refs)

        # Build a reverse map: title (basename, no .md) → cluster_id for fast lookup
        vault_ctx = fsm.context.get("vault_graph_ctx", {})
        _title_to_cluster: dict[str, int] = {
            k.rsplit("/", 1)[-1]: v["cluster_id"]
            for k, v in vault_ctx.items()
            if v.get("cluster_id", -1) >= 0
        }

        total_added = 0
        for path in touched_paths:
            try:
                note_title = os.path.splitext(os.path.basename(path))[0]
                note_cluster = _title_to_cluster.get(note_title, -1)

                # Narrow candidates to the same cluster when cluster data is available.
                # This prevents cross-cluster noise links (e.g. Economics ↔ Physics).
                if vault_ctx and note_cluster >= 0:
                    candidates = [
                        t for t in title_index
                        if _title_to_cluster.get(t, -1) == note_cluster and t != note_title
                    ]
                else:
                    candidates = None

                added = orch.DRIVER.autolink_note(
                    path, candidates=candidates if candidates is not None else title_index
                )
                if added:
                    total_added += len(added)
                    logger.info("AUTOLINK: %s — added %d link(s): %s", path, len(added), added)
            except Exception as _ae:
                logger.debug("AUTOLINK: skipped '%s' (non-fatal): %s", path, _ae)

        logger.info("AUTOLINK: finished — %d link(s) added across %d note(s)", total_added, len(touched_paths))
        fsm.context["yield_links"] = fsm.context.get("yield_links", 0) + total_added
    except Exception as e:
        # AUTOLINK is best-effort: log and continue to LINT
        logger.warning("AUTOLINK: phase failed (non-fatal): %s", e)

    fsm._progress_note(fsm._chunk_task_id("autolink"), "autolink", "done")
    fsm._transition_success()


def handle_backlink(fsm: "InjectorFSM") -> None:
    """Best-effort reverse link injection into pre-existing neighbouring notes (Phase 4.5).

    For each newly-written note (write ops, excluding the hub auto-creation),
    scans pre-existing notes that textually mention the new title and wraps
    those mentions as wikilinks.  Extends snapshot_domain and registers
    rollback inverses for any modified note so ROLLBACK and LINT graph-diff
    both cover the backlinks.
    """
    idx = fsm._current_chunk_idx
    fsm._progress_note(fsm._chunk_task_id("backlink"), "backlink", "running")

    try:
        from silica.kernel.autolink import backlink_pass, build_title_index

        ops = orch.load_ops(fsm._chunk_ctx["ops_path"])

        hub_name_lower = (fsm.hub or "").strip("[]").lower()
        new_titles: list[str] = []
        for op in ops:
            if op.op != OpType.write:
                continue
            path = op.touched_ref()
            if not path:
                continue
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem.lower() != hub_name_lower:
                new_titles.append(stem)

        if not new_titles:
            fsm._progress_note(fsm._chunk_task_id("backlink"), "backlink", "done")
            fsm._transition_success()
            return

        touched_paths_abs: set[str] = {
            os.path.abspath(p)
            for op in ops
            for p in (op.touched_ref(),)
            if p is not None
        }

        neighbourhood: list[str] = []
        seen_norm: set[str] = set()

        # Use the O(1) inverted text index if available; fall back to search_context.
        if hasattr(orch.DRIVER, "mentions_of"):
            for title in new_titles:
                try:
                    for path in orch.DRIVER.mentions_of(title):
                        norm = os.path.abspath(path)
                        if norm not in seen_norm and norm not in touched_paths_abs:
                            seen_norm.add(norm)
                            neighbourhood.append(path)
                except Exception as _me:
                    logger.debug("BACKLINK: mentions_of for '%s' failed: %s", title, _me)
        else:
            for title in new_titles:
                try:
                    for hit in orch.DRIVER.search_context(title):
                        p = hit.ref.path or hit.ref.name
                        norm = os.path.abspath(p)
                        if norm not in seen_norm and norm not in touched_paths_abs:
                            seen_norm.add(norm)
                            neighbourhood.append(p)
                except Exception as _se:
                    logger.debug("BACKLINK: search_context for '%s': %s", title, _se)


        if not neighbourhood:
            fsm._progress_note(fsm._chunk_task_id("backlink"), "backlink", "done")
            fsm._transition_success()
            return

        # Pre-read prior content before backlink_pass writes, for rollback inverses
        prior_contents: dict[str, str] = {}
        for path in neighbourhood:
            try:
                prior_contents[path] = orch.DRIVER.read_note(path).content or ""
            except Exception:
                pass

        all_refs = orch.DRIVER.list_files()
        title_index = build_title_index(all_refs)
        added_map = backlink_pass(new_titles, title_index=title_index, neighbourhood=neighbourhood)

        if added_map and fsm._txn is not None:
            from silica.kernel.ops import InverseOp, InverseOpKind
            existing_snapshot_paths = {d["path"] for d in fsm._chunk_ctx.get("snapshot_domain", [])}
            for path_modified in added_map:
                if path_modified not in existing_snapshot_paths:
                    stem = os.path.splitext(os.path.basename(path_modified))[0]
                    fsm._chunk_ctx.setdefault("snapshot_domain", []).append(
                        {"name": stem, "path": path_modified}
                    )
                    existing_snapshot_paths.add(path_modified)
                if path_modified in prior_contents:
                    inverse = InverseOp(
                        kind=InverseOpKind.restore_version,
                        path=path_modified,
                        prior_content=prior_contents[path_modified],
                    )
                    fsm._txn.inverses.append(inverse)

        total_links = sum(len(v) for v in added_map.values())
        logger.info(
            "BACKLINK: %d link(s) added to %d pre-existing note(s)", total_links, len(added_map)
        )
        fsm.context["yield_links"] = fsm.context.get("yield_links", 0) + total_links
    except Exception as e:
        logger.warning("BACKLINK: phase failed (non-fatal): %s", e)

    fsm._progress_note(fsm._chunk_task_id("backlink"), "backlink", "done")
    fsm._transition_success()
