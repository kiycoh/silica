# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

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
    """Concept recon for the CURRENT file only (per-file pipeline).

    The FSM loops RECON→…→WRITE per file: file 0 reaches its first write
    after one file's worth of embedding, not the whole inbox's. Cross-file
    coherence is carried by the substrate refreshed after each write, not by
    an up-front all-files pass.
    """
    fi = fsm._current_file_idx
    inbox_file = fsm.inbox_files[fi]
    fsm._progress_note("recon", "recon", "running")

    res = orch.silica_recon(inbox_file)
    if "error" in res:
        fsm._progress_note("recon", "recon", "failed", error=res["error"])
        raise RuntimeError(f"Recon failed for {inbox_file}: {res['error']}")
    # Accumulated across files — context["recon"] stays a list for uniformity
    fsm.context.setdefault("recon", []).append(res)

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
            notice = {
                "inbox_file": inbox_file,
                "content_hash": content_hash,
                "rejected_count": rejected_count,
            }
            existing = fsm.context.get("deferred")
            if existing is None:
                fsm.context["deferred"] = notice
            elif isinstance(existing, list):
                existing.append(notice)
            else:
                fsm.context["deferred"] = [existing, notice]

    fsm._progress_note("recon", "recon", "done")
    fsm._transition_success()


def handle_crossdedup(fsm: "InjectorFSM") -> None:
    """Cross-file concept deduplication — Phase 1.5, incremental variant.

    Embeds the CURRENT file's new_concepts (one small call) and compares them
    against the cached vectors of prior files' survivors: a near-duplicate
    (cosine ≥ τ_high) is removed, first-file occurrence wins — same semantics
    as the old all-files pass, paid per file instead of up-front. Best-effort:
    silently skips when the embedder is unavailable or the run is single-file.
    """
    recon_list: list[dict] = fsm.context.get("recon", [])
    if not recon_list or len(fsm.inbox_files) < 2:
        fsm._transition_success()
        return

    cur = recon_list[-1]  # appended by RECON for the current file
    names = list(cur.get("new_concepts", []))
    if not names:
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

    try:
        vecs = embedder.embed(names)
    except Exception as _e:
        logger.warning("CROSSDEDUP: embed call failed (%s) — skipping", _e)
        fsm._transition_success()
        return

    τ_high = getattr(orch.CONFIG, "sim_threshold_high", 0.85)
    fi = fsm._current_file_idx
    removed = 0
    for name, vec in zip(names, vecs):
        dup = next(
            ((pn, _cosine(vec, pv)) for pn, pv in fsm._crossdedup_vecs
             if _cosine(vec, pv) >= τ_high),
            None,
        )
        if dup is not None:
            nc = cur.get("new_concepts", [])
            if name in nc:
                nc.remove(name)
            removed += 1
            logger.info(
                "CROSSDEDUP: '%s' (file %d) merged into '%s' (score=%.3f)",
                name, fi, dup[0], dup[1],
            )
        else:
            fsm._crossdedup_vecs.append((name, vec))

    if removed:
        fsm.context["crossdedup_merged"] = fsm.context.get("crossdedup_merged", 0) + removed
        logger.info("CROSSDEDUP: %d duplicate concept(s) removed from file %d", removed, fi)
    fsm._transition_success()


def _within_cluster_tol(cached_sig, sig: list[int]) -> bool:
    """Reuse cached clusters while the graph drifted < ~2% (or 50 nodes / 100 edges)."""
    if not cached_sig or len(cached_sig) != 2:
        return False
    cn, ce = cached_sig
    n, e = sig
    return abs(n - cn) <= max(50, n // 50) and abs(e - ce) <= max(100, e // 50)


def build_vault_graph_ctx(fsm: "InjectorFSM") -> dict[str, dict]:
    """Compute per-note graph context (cluster/hub) from the current vault state.

    Returns a dict keyed by vault-relative path without .md extension:
        {"cluster_id": int, "hub": str|None, "is_hub": bool}
    Empty dict on any failure — all consumers treat missing context as a no-op.
    Uses the cheap structural report (no analytics): consumers read only
    cluster/hub, never PageRank.

    Scaling E: Louvain (~3.1s at 10k) is the per-run cost here. Clusters drift
    slowly, so the resulting ctx is cached keyed by a graph signature (node/edge
    counts) and reused while the graph drifted < ~2% — recomputed only when it
    has grown enough to matter. Accepts bounded staleness: a few recently-added
    notes read as cluster -1 (which consumers treat as "no cluster") until the
    next recompute — fine for routing context.
    """
    try:
        from silica.kernel.graph_export import (
            build_graph_data,
            ctx_from_report,
            load_cluster_ctx,
            save_cluster_ctx,
        )
        from silica.kernel.graph_report import compute_report
        _t = orch.time.monotonic()

        nodes, edges = build_graph_data(folder="")  # cheap snapshot (no Louvain)
        sig = [
            sum(1 for n in nodes if n.get("type") != "ghost"),
            sum(1 for e in edges if e.get("type") == "EXTRACTED"),
        ]
        cached = load_cluster_ctx()
        if cached and _within_cluster_tol(cached.get("sig"), sig):
            ctx = cached.get("ctx") or {}
            logger.info(
                "PAYLOAD: vault graph context reused from cache — %d nodes (%.2fs, Louvain skipped)",
                len(ctx), orch.time.monotonic() - _t,
            )
            return ctx

        report = compute_report(_nodes_edges_override=(nodes, edges))  # Louvain on miss
        ctx = ctx_from_report(report)
        save_cluster_ctx(sig, ctx)
        logger.info(
            "PAYLOAD: vault graph context built — %d nodes, %d clusters (%.2fs)",
            len(ctx), len(report.clusters), orch.time.monotonic() - _t,
        )
        return ctx
    except Exception as _e:
        logger.info("PAYLOAD: vault graph context unavailable (%s) — graph features disabled", _e)
        return {}


def novelty_gate(fsm: "InjectorFSM", raw_payload: dict) -> tuple[dict, int]:
    """SAGE-style capture-side novelty gate (Tier 2 cost).

    A concept whose TITLE cosine to an existing note's title is >=
    CONFIG.novelty_tau leaves the payload BEFORE chunking, so chunk count
    (= distiller calls) falls with
    them. They are never dropped: each goes to the deferred store and, when a
    work queue is running, to the concurrent ternary dedup judge (duplicate /
    distinct / contradicts), which authors the patch when warranted.

    tau unset/0 = off (payload returned untouched). Best-effort: embedder or
    store trouble keeps concepts in the payload, same contract as COLLISION.
    Returns (filtered_payload, diverted_count).
    """
    tau = float(getattr(orch.CONFIG, "novelty_tau", 0.0) or 0.0)
    if tau <= 0.0:
        return raw_payload, 0

    try:
        from silica.agent.providers import get_embedder
        from silica.kernel.embed import get_store
        store = get_store()
        if len(store) == 0:
            return raw_payload, 0
        embedder = get_embedder(orch.CONFIG)
    except Exception as _e:
        logger.debug("NOVELTY: embedder unavailable (%s); gate skipped", _e)
        return raw_payload, 0

    from silica.kernel.embed import _note_title_text
    from silica.kernel.paths import is_inbox_path
    from silica.router.states.collision import _names_agree

    def _name_of(c) -> str:
        return c.get("name", "") if isinstance(c, dict) else str(c)

    # Order parameter: TITLE-vs-title cosine (like-vs-like). A short concept
    # name is embedded as a title and scored against stored title vectors,
    # never against full note bodies — the body signal was measured not to
    # separate captured from novel concepts (their cosine distributions
    # overlap; docs/Silica_x_chemistry.md IV.3).
    names: list[str] = []
    for batch in raw_payload.get("batches", []):
        for c in batch.get("concepts", []):
            n = _name_of(c)
            if n.strip():
                names.append(n)
    uniq = list(dict.fromkeys(names))
    if not uniq:
        return raw_payload, 0
    try:
        vecs = embedder.embed([_note_title_text(n) for n in uniq])
        if len(vecs) != len(uniq):
            return raw_payload, 0
    except Exception as _e:
        logger.debug("NOVELTY: batch embed failed (%s); gate skipped", _e)
        return raw_payload, 0
    vec_by_name = dict(zip(uniq, vecs))

    kept_batches: list[dict] = []
    diverted: list[dict] = []
    for batch in raw_payload.get("batches", []):
        inbox_file = batch.get("inbox_file", fsm.inbox_file)
        kept: list = []
        for c in batch.get("concepts", []):
            name = _name_of(c)
            vec = vec_by_name.get(name)
            if not name or vec is None:
                kept.append(c)
                continue
            try:
                hits = store.title_cosine_top_k(vec, k=5)
            except Exception as _se:
                logger.debug("NOVELTY: title lookup failed for '%s': %s", name, _se)
                kept.append(c)
                continue
            hits = [h for h in hits if not is_inbox_path(h["path"])]
            best = hits[0] if hits else None
            # Cosine alone is a soup on dense taxonomic vaults: near-synonym and
            # negation-differing titles score ~0.97 (probe: nearest-distinct
            # p99=0.978). Require the same lexical name agreement COLLISION uses,
            # which rejects negation pairs ("context-free" vs "non context-free")
            # the embedding cannot tell apart.
            if (best is None or best["score"] < tau
                    or not _names_agree(name, best["name"])):
                kept.append(c)
                continue
            logger.info(
                "NOVELTY: '%s' ~ '%s' (title score=%.3f >= tau=%.2f); diverted",
                name, best["path"], best["score"], tau,
            )
            diverted.append({
                "concept": c,
                "inbox_file": inbox_file,
                "top_match": {"path": best["path"], "name": best["name"],
                              "score": best["score"]},
                "score": best["score"],
            })
        if kept:
            kept_batches.append({**batch, "concepts": kept})

    if diverted:
        from silica.router.states.collision import _deferred_op_dict
        fsm._defer_ops(
            [_deferred_op_dict(fsm, d, "novelty_gate") for d in diverted],
            {
                (d["concept"].get("name", str(i)) if isinstance(d["concept"], dict) else str(i)):
                f"novelty_gate score={d['score']:.3f}"
                for i, d in enumerate(diverted)
            },
            phase="NOVELTY",
        )
        if fsm.work_queue is not None:
            from silica.kernel.workqueue import WorkItem
            for d in diverted:
                c = d["concept"]
                match = d["top_match"]
                if not match.get("path"):
                    continue
                try:
                    fsm.work_queue.enqueue(WorkItem(
                        kind="dedup",
                        target_path=match["path"],
                        context={
                            "concept": c.get("name", "") if isinstance(c, dict) else str(c),
                            "excerpt": c.get("excerpt", "") if isinstance(c, dict) else "",
                            "candidate": match.get("name", match["path"]),
                            "score": d["score"],
                            "inbox_file": d["inbox_file"],
                            "hub": fsm.hub,
                            "content_hash": fsm._current_content_hash,
                            "target_dir": fsm.target_dir,
                        },
                        reason=f"novelty_gate score={d['score']:.3f}",
                    ))
                except Exception as _qe:
                    logger.debug("NOVELTY: failed to enqueue dedup item: %s", _qe)
        logger.info("NOVELTY: %d concept(s) diverted to the dedup lane pre-chunk",
                    len(diverted))

    return {**raw_payload, "batches": kept_batches}, len(diverted)


def handle_payload(fsm: "InjectorFSM") -> None:
    """Payload assembly for the CURRENT file only (per-file pipeline).

    Appends this file's chunks to the flat chunk list and registers its
    progress tasks; earlier files' chunks are already written by the time
    this runs again for the next file.
    """
    fi = fsm._current_file_idx
    inbox_file = fsm.inbox_files[fi] if fi < len(fsm.inbox_files) else fsm.inbox_file
    fsm._progress_note("payload", "payload", "running")
    # Current file's recon only — appended last by RECON
    recon_cur = fsm.context["recon"][-1]
    recon_path = fsm._make_tmp([recon_cur])
    phase_conf = fsm._get_recipe_phase("payload")
    max_concepts = phase_conf.get("partition_if_over", 200)
    max_bytes = int(os.getenv("DISTILLER_CHUNK_MAX_BYTES", str(30 * 1024)))
    res = orch.silica_payload(recon_path, max_concepts=max_concepts, max_bytes=max_bytes)
    if "error" in res:
        fsm._progress_note("payload", "payload", "failed", error=res["error"])
        raise RuntimeError(f"Payload failed: {res['error']}")
    fsm.context["payload"] = res

    # Re-partition this file's payload (§3.6); fall back to the legacy
    # flat-chunk path when batch structure is absent (e.g. tests).
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

    # Tier 2 novelty gate: divert already-captured concepts to the dedup lane
    # BEFORE chunking so chunk count (= distiller calls) falls with them.
    diverted_all = False
    if raw_payload is not None:
        raw_payload, _n_diverted = novelty_gate(fsm, raw_payload)
        diverted_all = _n_diverted > 0 and not any(
            b.get("concepts") for b in raw_payload.get("batches", [])
        )

    new_chunks: list[dict] = []
    if raw_payload and max_concepts > 0:
        # Single-file recon → normally a single group; collect all defensively.
        for fg in partition_by_file(raw_payload, max_concepts) or []:
            new_chunks.extend(fg.get("chunks", []))

    if not new_chunks and diverted_all:
        # Every concept diverted: one empty chunk carries the file through the
        # normal pipeline (DELEGATE skips the LLM, VALIDATE short-circuits).
        # Falling through would resurrect the unfiltered fallback chunks.
        new_chunks = [{"schema_version": raw_payload.get("schema_version", 1),
                       "batches": []}]

    if not new_chunks:
        # Fallback: all chunks of this payload belong to the current file.
        new_chunks = res.get("chunks", [])
        if not new_chunks and "payload" in res:
            new_chunks = [res["payload"]]
        if not new_chunks:
            new_chunks = [res]

    # Append this file's chunk group; flat indices continue after prior files'
    start_flat = len(fsm._chunks)
    fsm._file_chunks[fi] = {"source_file": inbox_file, "chunks": new_chunks}
    for ci, chunk in enumerate(new_chunks):
        fsm._chunks.append(chunk)
        fsm._chunk_flat_to_fi_ci[start_flat + ci] = (fi, ci)
    fsm._current_chunk_idx = start_flat

    # Cache-stable prompt: pin the distiller LANGUAGE once per file so the
    # rendered template prefix is byte-identical across this file's chunks
    # (per-chunk detection can flap between chunks and bust the prefix cache).
    try:
        from silica.kernel import language as lang_mod
        from silica.kernel.prep_delegation import _payload_sample_text
        from silica.kernel.vault_manifest import get_active_manifest
        if not get_active_manifest().conventions.language:
            sample = ""
            for _chunk in new_chunks:
                sample = _payload_sample_text(_chunk)
                if sample:
                    break
            fsm.context[f"file_{fi}_language"] = lang_mod.display_name(
                lang_mod.detect(sample[:4000])
            )
    except Exception as _lang_e:
        logger.debug("PAYLOAD: language pin skipped (non-fatal): %s", _lang_e, exc_info=True)

    # Accumulate facts["sources"] with per-file concept + chunk counts
    n_concepts = sum(
        len(b.get("concepts", []))
        for chunk in new_chunks
        for b in chunk.get("batches", [])
    )
    fsm.progress.inputs.setdefault("sources", []).append({
        "inbox_file": inbox_file,
        "concepts": n_concepts,
        "chunks": len(new_chunks),
    })

    # Register per-chunk tasks with f{fi}_c{ci}_{cap} IDs and intra-file deps
    caps = ("collision", "distill", "sanitize", "validate", "snapshot", "write", "hub_update", "autolink", "backlink", "lint", "cleanup")
    prev_in_file = "payload"
    for ci in range(len(new_chunks)):
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
        "File %d/%d '%s': %d chunk(s) queued (flat %d–%d).",
        fi + 1, len(fsm.inbox_files), inbox_file,
        len(new_chunks), start_flat, len(fsm._chunks) - 1,
    )

    # Build vault graph context (cluster/hub/pagerank) once per run — reused
    # across files (consumers accept bounded staleness). Consumed by COLLISION,
    # DELEGATE (distiller enrichment), AUTOLINK, and HUB_UPDATE.
    if "vault_graph_ctx" not in fsm.context:
        fsm.context["vault_graph_ctx"] = build_vault_graph_ctx(fsm)

    fsm._transition_success()


def handle_salience(fsm: "InjectorFSM") -> None:
    """Thematic salience gate — Phase 2.05, current file's chunks only.

    Drops concepts whose embedding is too far from the document's thematic
    centroid.  Best-effort: any failure (embedder down, empty index) is
    logged and chunks pass unchanged.  Runs once per file (per-file
    pipeline); _eval_loop_or_done restarts chunks from COLLISION, which is
    correct.
    """
    τ_theme = getattr(orch.CONFIG, "sim_threshold_theme", 0.35)
    try:
        from silica.agent.providers import get_embedder
        from silica.kernel.embed import document_theme_vector, _cosine
        from silica.kernel.text import clean_body
        embedder = get_embedder(orch.CONFIG)
    except Exception as _e:
        logger.warning("SALIENCE: embedder unavailable (%s) — skipping", _e)
        fsm._transition_success()
        return

    fsm._get_chunks_from_context_if_empty()
    theme_cache: dict[str, list[float]] = {}
    dropped = 0

    cur_fi = fsm._current_file_idx
    current_chunks = [
        chunk for flat_idx, chunk in enumerate(fsm._chunks)
        if fsm._chunk_flat_to_fi_ci.get(flat_idx, (0, 0))[0] == cur_fi
    ] or fsm._chunks  # fallback: no fi map (legacy/test paths) → all chunks

    for chunk in current_chunks:
        for batch in chunk.get("batches", []):
            inbox_file = batch.get("inbox_file", fsm.inbox_file)
            if inbox_file not in theme_cache:
                try:
                    # Same cleaned body as RECON's keyphrase pass → the theme
                    # vector is a cache hit in embed._theme_cache, no re-embed.
                    body = clean_body(orch.DRIVER.read_note(inbox_file).content, fences=True)
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
