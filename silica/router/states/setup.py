"""Injector run-setup states: RECON, CROSSDEDUP, PAYLOAD, SALIENCE.

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


def handle_recon(fsm: "InjectorFSM") -> None:
    fsm._progress_note("recon", "recon", "running")

    # Iterate all inbox files and aggregate recon reports into a list
    recon_list: list[dict] = []
    deferred_notices: list[dict] = []
    for fi, inbox_file in enumerate(fsm.inbox_files):
        res = orch.silica_recon(inbox_file)
        if "error" in res:
            fsm._progress_note("recon", "recon", "failed", error=res["error"])
            raise RuntimeError(f"Recon failed for {inbox_file}: {res['error']}")
        recon_list.append(res)

        # Degraded extraction held back uncorroborated concepts (see silica_recon /
        # CONFIG.defer_uncorroborated_concepts) — make it visible, never silent.
        deferred_concepts = res.get("deferred_concepts") or []
        if deferred_concepts:
            logger.info(
                "RECON: %d uncorroborated concept(s) deferred for '%s' (embedder down) "
                "— re-inject with the embedder available to admit them.",
                len(deferred_concepts), inbox_file,
            )

        # Surface any deferred ops from a previous run of this file
        content_hash = fsm._file_content_hashes[fi] if fi < len(fsm._file_content_hashes) else ""
        if content_hash:
            from silica.kernel.deferred import get_deferred_store
            bundle = get_deferred_store().get(content_hash)
            if bundle:
                rejected_count = len(bundle.get("rejected_ops", []))
                logger.info(
                    "RECON: %d deferred op(s) from a previous run of '%s' are waiting. "
                    "Call silica_deferred_retry('%s') to attempt them.",
                    rejected_count, inbox_file, content_hash[:8],
                )
                deferred_notices.append({
                    "inbox_file": inbox_file,
                    "content_hash": content_hash,
                    "rejected_count": rejected_count,
                })

    # Always a list — even for single-file runs — so _handle_payload is uniform
    fsm.context["recon"] = recon_list
    if deferred_notices:
        fsm.context["deferred"] = deferred_notices[0] if len(deferred_notices) == 1 else deferred_notices

    fsm._progress_note("recon", "recon", "done")
    fsm._transition_success()


def handle_crossdedup(fsm: "InjectorFSM") -> None:
    """Cross-file concept deduplication — Phase 1.5.

    Embeds concept names extracted by RECON across all inbox files.
    Near-duplicate concepts from different files (cosine ≥ τ_high) are
    merged: the first-file occurrence is kept, the duplicate is removed.
    Best-effort: silently skips when the embedder is unavailable or
    fewer than two inbox files are present.
    """
    recon_list: list[dict] = fsm.context.get("recon", [])

    if len(recon_list) < 2:
        fsm._transition_success()
        return

    # Collect (file_index, concept_name) for all new_concepts across files
    all_concepts: list[tuple[int, str]] = [
        (fi, name)
        for fi, rec in enumerate(recon_list)
        for name in rec.get("new_concepts", [])
    ]

    if len(all_concepts) < 2:
        fsm._transition_success()
        return

    try:
        from silica.agent.providers import get_embedder
        from silica.kernel.embed import _cosine
        embedder = get_embedder(orch.CONFIG)
    except Exception as _e:
        logger.warning("CROSSDEDUP: embedder unavailable (%s) — skipping", _e)
        fsm._transition_success()
        return

    texts = [name for _, name in all_concepts]
    try:
        vecs = embedder.embed(texts)
    except Exception as _e:
        logger.warning("CROSSDEDUP: embed call failed (%s) — skipping", _e)
        fsm._transition_success()
        return

    τ_high = getattr(orch.CONFIG, "sim_threshold_high", 0.85)

    # Greedy O(n²) clustering: mark cross-file near-duplicates for removal.
    # The first occurrence (lowest file index) is always the winner.
    losers: set[int] = set()
    for i in range(len(all_concepts)):
        if i in losers:
            continue
        fi, name_i = all_concepts[i]
        for j in range(i + 1, len(all_concepts)):
            if j in losers:
                continue
            fj, name_j = all_concepts[j]
            if fi == fj:
                continue
            if _cosine(vecs[i], vecs[j]) >= τ_high:
                losers.add(j)
                logger.info(
                    "CROSSDEDUP: '%s' (file %d) merged into '%s' (file %d, score=%.3f)",
                    name_j, fj, name_i, fi, _cosine(vecs[i], vecs[j]),
                )

    if not losers:
        fsm._transition_success()
        return

    for idx in losers:
        fi, name = all_concepts[idx]
        nc = recon_list[fi].get("new_concepts", [])
        if name in nc:
            nc.remove(name)

    fsm.context["recon"] = recon_list
    fsm.context["crossdedup_merged"] = len(losers)
    logger.info(
        "CROSSDEDUP: %d duplicate concept(s) removed across %d files",
        len(losers), len(recon_list),
    )
    fsm._transition_success()


def build_vault_graph_ctx(fsm: "InjectorFSM") -> dict[str, dict]:
    """Compute per-note graph context (cluster/hub/pagerank) from the current vault state.

    Returns a dict keyed by vault-relative path without .md extension:
        {"cluster_id": int, "hub": str|None, "is_hub": bool, "pagerank": float}
    Empty dict on any failure — all consumers treat missing context as a no-op.
    """
    try:
        from silica.kernel.graph_report import compute_report
        _t = orch.time.monotonic()
        report = compute_report()
        ctx: dict[str, dict] = {}
        for cs in report.clusters:
            for member in cs.members:
                ctx[member] = {
                    "cluster_id": cs.cluster_id,
                    "hub": cs.hub,
                    "is_hub": member == cs.hub,
                    "pagerank": report.pagerank_map.get(member, 0.0),
                }
        # Include isolated nodes (not in any cluster) so pagerank is available
        for node_id, pr_val in report.pagerank_map.items():
            if node_id not in ctx:
                ctx[node_id] = {"cluster_id": -1, "hub": None, "is_hub": False, "pagerank": pr_val}
        logger.info(
            "PAYLOAD: vault graph context built — %d nodes, %d clusters (%.2fs)",
            len(ctx), len(report.clusters), orch.time.monotonic() - _t,
        )
        return ctx
    except Exception as _e:
        logger.info("PAYLOAD: vault graph context unavailable (%s) — graph features disabled", _e)
        return {}


def handle_payload(fsm: "InjectorFSM") -> None:
    fsm._progress_note("payload", "payload", "running")
    # fsm.context["recon"] is now always a list of per-file recon dicts
    recon_path = fsm._make_tmp(fsm.context["recon"])
    phase_conf = fsm._get_recipe_phase("payload")
    max_concepts = phase_conf.get("partition_if_over", 200)
    max_bytes = int(os.getenv("DISTILLER_CHUNK_MAX_BYTES", str(30 * 1024)))
    res = orch.silica_payload(recon_path, max_concepts=max_concepts, max_bytes=max_bytes)
    if "error" in res:
        fsm._progress_note("payload", "payload", "failed", error=res["error"])
        raise RuntimeError(f"Payload failed: {res['error']}")
    fsm.context["payload"] = res

    # Build per-file chunk hierarchy (§3.6).
    # Try to use partition_by_file when the payload has proper batch structure;
    # fall back to the legacy flat-chunk path when batches are absent (e.g. tests).
    from silica.kernel.partition import partition_by_file

    raw_payload: dict | None = None
    if "chunks" in res and res["chunks"]:
        all_batches: list[dict] = []
        for chunk in res["chunks"]:
            all_batches.extend(chunk.get("batches", []))
        if all_batches:
            raw_payload = {
                "schema_version": res["chunks"][0].get("schema_version", 1),
                "batches": all_batches,
            }
    elif "payload" in res:
        raw_payload = res["payload"]

    if raw_payload and max_concepts > 0:
        attempt = partition_by_file(raw_payload, max_concepts)
        if attempt:
            fsm._file_chunks = attempt

    if not fsm._file_chunks:
        # Fallback: all chunks belong to the first (or only) inbox file.
        # Do NOT split by chunk — one physical file = one file group.
        raw_chunks = res.get("chunks", [])
        if not raw_chunks and "payload" in res:
            raw_chunks = [res["payload"]]
        if not raw_chunks:
            raw_chunks = [res]
        fsm._file_chunks.append({"source_file": fsm.inbox_file, "chunks": raw_chunks})

    # Build flat chunk list preserving file order (for existing handler logic)
    fsm._chunks = []
    fsm._chunk_flat_to_fi_ci = {}
    flat_idx = 0
    for fi, fg in enumerate(fsm._file_chunks):
        for ci, chunk in enumerate(fg.get("chunks", [])):
            fsm._chunks.append(chunk)
            fsm._chunk_flat_to_fi_ci[flat_idx] = (fi, ci)
            flat_idx += 1

    if not fsm._chunks:
        fsm._chunks = [res]
        fsm._chunk_flat_to_fi_ci = {0: (0, 0)}

    fsm._current_chunk_idx = 0

    # Build facts["sources"] with per-file concept + chunk counts
    sources_facts: list[dict] = []
    for fi, fg in enumerate(fsm._file_chunks):
        n_chunks = len(fg.get("chunks", []))
        n_concepts = sum(
            len(b.get("concepts", []))
            for chunk in fg.get("chunks", [])
            for b in chunk.get("batches", [])
        )
        sources_facts.append({
            "inbox_file": fg["source_file"],
            "concepts": n_concepts,
            "chunks": n_chunks,
        })

    # Stash per-file stats in progress inputs for the digest
    fsm.progress.inputs["sources"] = sources_facts

    # Register per-chunk tasks with f{fi}_c{ci}_{cap} IDs and intra-file deps
    caps = ("collision", "distill", "sanitize", "validate", "snapshot", "write", "hub_update", "autolink", "backlink", "lint", "cleanup")
    for fi, fg in enumerate(fsm._file_chunks):
        prev_in_file = "payload"
        for ci in range(len(fg.get("chunks", []))):
            for cap in caps:
                tid = f"f{fi}_c{ci}_{cap}"
                fsm.progress.add_task(cap, task_id=tid, depends_on=[prev_in_file])
                prev_in_file = tid
    try:
        fsm.progress.save()
    except Exception as _e:
        logger.debug("progress save error (suppressed): %s", _e)

    fsm._progress_note("payload", "payload", "done")
    logger.info(
        "Pipeline initialized: %d file(s), %d total chunk(s). Files: %s",
        len(fsm._file_chunks),
        len(fsm._chunks),
        [fg["source_file"] for fg in fsm._file_chunks],
    )

    # Build vault graph context (cluster/hub/pagerank) once per run.
    # Stored in context["vault_graph_ctx"] and consumed by COLLISION,
    # DELEGATE (distiller enrichment), AUTOLINK, and HUB_UPDATE.
    fsm.context["vault_graph_ctx"] = build_vault_graph_ctx(fsm)

    fsm._transition_success()


def handle_salience(fsm: "InjectorFSM") -> None:
    """Thematic salience gate — Phase 2.05.

    Single-pass over ALL chunks: drops concepts whose embedding is too far
    from the document's thematic centroid.  Best-effort: any failure
    (embedder down, empty index) is logged and chunks pass unchanged.
    Does NOT re-run on subsequent chunk iterations — _eval_loop_or_done
    restarts from COLLISION, which is correct.
    """
    if not getattr(orch.CONFIG, "salience_gate_enabled", True):
        fsm._transition_success()
        return

    τ_theme = getattr(orch.CONFIG, "sim_threshold_theme", 0.35)
    try:
        from silica.agent.providers import get_embedder
        from silica.kernel.embed import document_theme_vector, _cosine
        from silica.kernel.recon import _strip_frontmatter
        embedder = get_embedder(orch.CONFIG)
    except Exception as _e:
        logger.warning("SALIENCE: embedder unavailable (%s) — skipping", _e)
        fsm._transition_success()
        return

    fsm._get_chunks_from_context_if_empty()
    theme_cache: dict[str, list[float]] = {}
    dropped = 0

    for chunk in fsm._chunks:
        for batch in chunk.get("batches", []):
            inbox_file = batch.get("inbox_file", fsm.inbox_file)
            if inbox_file not in theme_cache:
                try:
                    body = _strip_frontmatter(orch.DRIVER.read_note(inbox_file).content)
                except Exception:
                    body = ""
                theme_cache[inbox_file] = document_theme_vector(embedder, body)
            theme = theme_cache[inbox_file]
            if not theme:
                continue

            concepts = batch.get("concepts", [])
            texts = [
                (c.get("name", "") + "\n" + c.get("inbox_excerpt", "")) if isinstance(c, dict) else str(c)
                for c in concepts
            ]
            if not texts:
                continue
            try:
                vecs = embedder.embed(texts)
            except Exception as _e:
                logger.debug("SALIENCE: embed failed (%s) — keeping batch", _e)
                continue

            kept = []
            for c, v in zip(concepts, vecs):
                score = _cosine(v, theme)
                name = c.get("name", "") if isinstance(c, dict) else str(c)
                if score < τ_theme:
                    logger.info(
                        "SALIENCE: drop '%s' (score=%.3f < τ_theme=%.2f)", name, score, τ_theme
                    )
                    dropped += 1
                else:
                    kept.append(c)
            batch["concepts"] = kept

    fsm.context["salience_dropped"] = dropped
    if dropped:
        logger.info("SALIENCE: %d concept(s) below thematic threshold removed", dropped)
    fsm._transition_success()
