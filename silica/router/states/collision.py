"""Injector COLLISION state: embedding-based dedup routing.

Handler bodies for InjectorFSM, extracted from orchestrator.py: each function
takes the FSM instance and mutates its context/state exactly as the former
method did. Patchable collaborators (DRIVER, CONFIG, tools, load_ops, time)
are resolved through the orchestrator module namespace (orch.X) so tests that
patch silica.router.orchestrator.* keep working.
"""
from __future__ import annotations

import logging
import os
import hashlib
from typing import Any, TYPE_CHECKING

from silica.router import orchestrator as orch

if TYPE_CHECKING:
    from silica.router.orchestrator import InjectorFSM

logger = logging.getLogger(__name__)


def _names_agree(concept: str, note_name: str) -> bool:
    """Conservative lexical gate for the mechanical high-score auto-patch.

    A high cosine can be driven by a single shared word — e.g. the concept
    "MEMORY" against the note "RAM (Random Access Memory)" — which is a domain
    collision, not the same concept. COLLISION may bypass the distiller and patch
    directly ONLY when the names genuinely agree; otherwise the concept is demoted
    to normal distillation so the distiller can judge from the excerpts. A wrong
    demotion only costs an extra distillation pass; a wrong auto-patch pollutes the
    vault, so the gate is deliberately strict.

    Agreement holds when either name normalizes to the other (same slug) or the
    concept equals the note's acronym — its parenthetical token (e.g. "(GPT)") or
    the head token before the first parenthesis.
    """
    import re
    from silica.kernel.templates import slugify

    if not concept.strip() or not note_name.strip():
        return False
    if slugify(concept) == slugify(note_name):
        return True
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    acronyms = set(re.findall(r"\(([^)]*)\)", note_name))   # parenthetical contents
    acronyms.add(note_name.split("(", 1)[0])                # head before any paren
    nc = norm(concept)
    return bool(nc) and nc in {norm(a) for a in acronyms if a.strip()}


def _embedder_free_near_dups(
    chunk: dict, corpus: dict[str, str], *, threshold: float = 0.6
) -> list[dict]:
    """Concepts in `chunk` that have a MinHash near-duplicate in `corpus`.

    Pure: no FSM, no DRIVER. `corpus` maps note path → body text. Returns dicts
    shaped like the borderline `deferred_concepts` entries so the existing defer
    plumbing handles them unchanged. The returned `concept` is the same object
    held in the chunk, so callers can drop it by identity.
    """
    from silica.kernel.minhash_dedup import near_duplicates

    out: list[dict] = []
    for batch in chunk.get("batches", []):
        inbox_file = batch.get("inbox_file", "")
        for concept in batch.get("concepts", []):
            if isinstance(concept, dict):
                name, excerpt = concept.get("name", ""), concept.get("excerpt", "")
            else:
                name, excerpt = str(concept), ""
            query = f"{name}\n{excerpt}".strip()
            if not query:
                continue
            hits = near_duplicates(query, corpus, threshold=threshold)
            if not hits:
                continue
            path, score = hits[0]
            note_name = os.path.splitext(os.path.basename(path))[0]
            out.append({
                "concept": concept,
                "inbox_file": inbox_file,
                "top_match": {"path": path, "name": note_name, "score": score},
                "score": score,
            })
    return out


def _run_embedder_free_dedup_leg(fsm: "InjectorFSM", idx: int, chunk: dict) -> None:
    """STABLE dedup leg: MinHash near-dup pass for when the embedder is unavailable.

    Near-dups are DEFERRED for review and dropped from the chunk so DELEGATE does
    not write a duplicate note. They are never mechanically patched — the
    embedder-free signal is weaker than a cosine, so it routes to escalate-tier,
    not an auto-write. Best-effort: any failure leaves the chunk untouched.
    """
    threshold = getattr(orch.CONFIG, "minhash_dup_threshold", 0.6)
    try:
        corpus: dict[str, str] = {}
        # "" matches every note name → full-vault enumeration. Acceptable on this
        # cold path (only reached when the embedder/index is down).
        # ponytail: rebuilds the corpus per chunk; share a MinHash index with the
        # embed-refresh hook if this ever runs hot.
        for ref in orch.DRIVER.search_names(""):
            try:
                corpus[ref.path] = orch.DRIVER.read_note(ref).content or ""
            except Exception:
                continue
    except Exception as _e:
        logger.debug("COLLISION minhash leg: corpus build failed (%s) — skipping", _e)
        return
    if not corpus:
        return

    deferred = _embedder_free_near_dups(chunk, corpus, threshold=threshold)
    if not deferred:
        return

    dup_ids = {id(d["concept"]) for d in deferred}
    new_batches: list[dict] = []
    for batch in chunk.get("batches", []):
        kept = [c for c in batch.get("concepts", []) if id(c) not in dup_ids]
        if kept:
            new_batches.append(
                {"inbox_file": batch.get("inbox_file", fsm.inbox_file), "concepts": kept}
            )
    fsm._chunks[idx] = {"schema_version": chunk.get("schema_version", 1), "batches": new_batches}

    deferred_op_dicts = [
        {
            "op": "skip",
            "heading": (d["concept"].get("name", "") if isinstance(d["concept"], dict) else str(d["concept"])),
            "source_basename": os.path.basename(d["inbox_file"]),
            "reason": f"minhash_near_dup score={d['score']:.3f} candidate={d['top_match'].get('name','?')}",
            "path": None,
        }
        for d in deferred
    ]
    fsm._defer_ops(
        deferred_op_dicts,
        {
            (d["concept"].get("name", str(i)) if isinstance(d["concept"], dict) else str(i)):
            f"minhash_near_dup score={d['score']:.3f}"
            for i, d in enumerate(deferred)
        },
        phase="COLLISION",
    )
    logger.info(
        "COLLISION minhash leg: deferred %d near-duplicate concept(s) (embedder-free)",
        len(deferred),
    )


def handle_collision(fsm: "InjectorFSM") -> None:
    """Dedup/collision routing — Phase 5.

    Candidates come from the relatedness facade (RRF fusion of embeddings +
    co-occurrence), so an existing note the author co-mentions heavily can
    outrank a merely cosine-close one. Routing stays embedding-anchored: the
    thresholds below apply to the candidate's cosine (embed_score), and a
    candidate the embed leg did not propose is never auto-routed.

    For each concept in the current chunk:
    - score ≥ τ_high  → pre-route as a 'patch' op on the existing note
                        (graph check: note must exist in vault)
    - τ_low < score < τ_high → defer (borderline, ambiguous)
    - score ≤ τ_low   → keep for normal distillation (new write)

    Best-effort: any failure (missing index, embedder down) silently skips
    the check and lets the chunk flow to DELEGATE unchanged.
    """
    idx = fsm._current_chunk_idx
    fsm._progress_note(fsm._chunk_task_id("collision"), "collision", "running")

    τ_high = getattr(orch.CONFIG, "sim_threshold_high", 0.85)
    τ_low = getattr(orch.CONFIG, "sim_threshold_low", 0.65)

    try:
        from silica.agent.providers import get_embedder
        from silica.kernel.embed import EmbedStore

        store = EmbedStore()
        if len(store) == 0:
            logger.info("COLLISION: embedding index empty — falling back to MinHash dedup leg")
            fsm._get_chunks_from_context_if_empty()
            _run_embedder_free_dedup_leg(fsm, idx, fsm._chunks[idx])
            fsm._progress_note(fsm._chunk_task_id("collision"), "collision", "done")
            fsm._transition_success()
            return
        embedder = get_embedder(orch.CONFIG)
    except Exception as _e:
        logger.warning("COLLISION: embedder unavailable (%s) — falling back to MinHash dedup leg", _e)
        fsm._get_chunks_from_context_if_empty()
        _run_embedder_free_dedup_leg(fsm, idx, fsm._chunks[idx])
        fsm._progress_note(fsm._chunk_task_id("collision"), "collision", "done")
        fsm._transition_success()
        return

    # Co-occurrence leg for the relatedness facade — embedder-free, best-effort:
    # an unavailable or empty index means the leg abstains and fusion degrades
    # to the embedding ranking alone.
    cooccur_store = None
    try:
        from silica.kernel.cooccurrence import CooccurStore
        cooccur_store = CooccurStore(lang=orch.CONFIG.cooccurrence_lang)
        if len(cooccur_store) == 0:
            cooccur_store = None
    except Exception:
        cooccur_store = None

    fsm._get_chunks_from_context_if_empty()
    chunk = fsm._chunks[idx]

    pre_routed_ops: list[dict] = []
    deferred_concepts: list[dict] = []
    modified_batches: list[dict] = []

    # Embed every concept in the chunk in a SINGLE call (one network
    # round-trip per chunk instead of one per concept).  Falls back to
    # per-concept embedding only if the embedder returns a ragged response,
    # so a short/odd reply can never silently drop concepts.
    #
    # The query text is name+excerpt, built with the SAME _note_text used to
    # index notes, so the query vector is comparable to the stored title+body
    # vectors. Embedding the bare name lets short acronyms ("MEM", "ACE") score
    # spuriously high against unrelated short-acronym notes ("RAM (Random Access
    # Memory)", "MACE"); the excerpt anchors the concept in its real neighbourhood.
    from silica.kernel.embed import _note_text

    def _concept_embed_text(concept: Any) -> str:
        if isinstance(concept, dict):
            return _note_text(concept.get("name", ""), concept.get("excerpt", ""))
        return _note_text(str(concept), "")

    all_texts: list[str] = []
    for batch in chunk.get("batches", []):
        for concept in batch.get("concepts", []):
            et = _concept_embed_text(concept)
            if et.strip():
                all_texts.append(et)

    vec_by_text: dict[str, Any] = {}
    uniq_texts = list(dict.fromkeys(all_texts))
    if uniq_texts:
        try:
            embedded = embedder.embed(uniq_texts)
            batched_ok = len(embedded) == len(uniq_texts)
        except Exception as _embed_err:
            logger.debug("COLLISION: batch embed failed (%s) — keeping concepts unrouted", _embed_err)
            embedded, batched_ok = [], False
        if batched_ok:
            vec_by_text = dict(zip(uniq_texts, embedded))
        else:
            for _t in uniq_texts:
                try:
                    _ev = embedder.embed([_t])
                    vec_by_text[_t] = _ev[0]
                except Exception as _embed_err:
                    logger.debug("COLLISION: embed failed for '%s': %s", _t, _embed_err)

    for batch in chunk.get("batches", []):
        inbox_file = batch.get("inbox_file", fsm.inbox_file)
        kept: list = []

        for concept in batch.get("concepts", []):
            concept_text = concept.get("name", "") if isinstance(concept, dict) else str(concept)
            if not concept_text:
                kept.append(concept)
                continue

            vec = vec_by_text.get(_concept_embed_text(concept))
            if vec is None:
                # Embedding unavailable for this concept (batch failed or
                # missing) — keep it for normal distillation.
                kept.append(concept)
                continue
            try:
                from silica.kernel.relatedness import related_notes_for_query
                excerpt_text = concept.get("excerpt", "") if isinstance(concept, dict) else ""
                related = related_notes_for_query(
                    query_vec=vec,
                    query_text=f"{concept_text}\n{excerpt_text}".strip(),
                    embed_store=store,
                    cooccur_store=cooccur_store,
                    k=1,
                )
            except Exception as _search_err:
                logger.debug("COLLISION: relatedness lookup failed for '%s': %s", concept_text, _search_err)
                kept.append(concept)
                continue

            if not related:
                kept.append(concept)
                continue

            best = related[0]
            if best.embed_score is None:
                # Co-occurrence-only candidate: there is no cosine to hold the
                # thresholds against, so it is never auto-routed — the concept
                # flows to normal distillation.
                logger.debug(
                    "COLLISION: '%s' top candidate '%s' lacks an embed score (%s) — keeping",
                    concept_text, best.path, ", ".join(best.evidence),
                )
                kept.append(concept)
                continue

            top = {"path": best.path, "name": best.name, "score": best.embed_score}
            score: float = best.embed_score
            existing_path = best.path

            # Lower effective threshold for cluster hubs: merging into an
            # anchor note is safer than creating a competing shadow note.
            _vault_ctx = fsm.context.get("vault_graph_ctx", {})
            _match_key = existing_path.removesuffix(".md")
            _is_hub = _vault_ctx.get(_match_key, {}).get("is_hub", False)
            τ_eff = τ_high - (0.08 if _is_hub else 0.0)

            if score >= τ_eff and not _names_agree(concept_text, top["name"]):
                # High cosine but the names disagree — a domain collision (shared
                # word, different concept). Don't bypass the distiller mechanically;
                # demote to normal distillation so it can judge from the excerpts.
                logger.info(
                    "COLLISION: '%s' ~ '%s' (score=%.3f) but names disagree — "
                    "demoting to distiller (no mechanical patch)",
                    concept_text, existing_path, score,
                )
                kept.append(concept)

            elif score >= τ_eff:
                try:
                    orch.DRIVER.read_note(existing_path)
                    # Graph confirms node exists — safe to patch
                    logger.info(
                        "COLLISION: '%s' → patch '%s' (score=%.3f ≥ τ_eff=%.2f%s)",
                        concept_text, existing_path, score, τ_eff,
                        " [hub]" if _is_hub else "",
                    )
                    pre_routed_ops.append({
                        "op": "patch",
                        "path": existing_path,
                        "heading": concept_text,
                        "source_basename": os.path.basename(inbox_file),
                        "snippet": concept.get("excerpt", "") if isinstance(concept, dict) else "",
                        "hub": fsm.hub,
                        "reason": f"collision_routed score={score:.3f}{' [hub]' if _is_hub else ''}",
                    })
                except Exception:
                    # Node not in graph — treat as new write
                    logger.debug(
                        "COLLISION: '%s' high score but '%s' not in graph → keep as write",
                        concept_text, existing_path,
                    )
                    kept.append(concept)

            elif score > τ_low:
                logger.info(
                    "COLLISION: '%s' → deferred (score=%.3f in borderline zone)",
                    concept_text, score,
                )
                deferred_concepts.append({
                    "concept": concept,
                    "inbox_file": inbox_file,
                    "top_match": top,
                    "score": score,
                })

            else:
                kept.append(concept)

        if kept:
            modified_batches.append({"inbox_file": inbox_file, "concepts": kept})

    # Persist borderline concepts in the deferred store
    if deferred_concepts:
        deferred_op_dicts = [
            {
                "op": "skip",
                "heading": (d["concept"].get("name", "") if isinstance(d["concept"], dict) else str(d["concept"])),
                "source_basename": os.path.basename(d["inbox_file"]),
                "reason": f"collision_deferred score={d['score']:.3f} candidate={d['top_match'].get('name','?')}",
                "path": None,
            }
            for d in deferred_concepts
        ]
        fsm._defer_ops(
            deferred_op_dicts,
            {
                (d["concept"].get("name", str(i)) if isinstance(d["concept"], dict) else str(i)):
                f"borderline_similarity score={d['score']:.3f}"
                for i, d in enumerate(deferred_concepts)
            },
            phase="COLLISION",
        )

    # Producer: hand each borderline pair to the leashed dedup sub-agent so it
    # can run concurrently while the Injector keeps writing its other batches.
    # The candidate match is a pre-existing (committed) vault note, so the
    # sub-agent's append-only patch never races the Injector's new-note writes;
    # the per-path lease covers the rare same-note overlap.
    if deferred_concepts and fsm.work_queue is not None:
        from silica.kernel.workqueue import WorkItem
        for d in deferred_concepts:
            concept = d["concept"]
            match = d.get("top_match", {})
            candidate_path = match.get("path", "")
            if not candidate_path:
                continue
            name = concept.get("name", "") if isinstance(concept, dict) else str(concept)
            excerpt = concept.get("excerpt", "") if isinstance(concept, dict) else ""
            try:
                fsm.work_queue.enqueue(WorkItem(
                    kind="dedup",
                    target_path=candidate_path,
                    context={
                        "concept": name,
                        "excerpt": excerpt,
                        "candidate": match.get("name", candidate_path),
                        "score": d.get("score"),
                        "inbox_file": d.get("inbox_file", fsm.inbox_file),
                        "hub": fsm.hub,
                    },
                    reason=f"borderline_similarity score={d.get('score', 0):.3f}",
                ))
            except Exception as _qe:
                logger.debug("COLLISION: failed to enqueue dedup item: %s", _qe)

    # Store pre-routed ops for merging in VALIDATE (Phase 5)
    fsm.context[f"chunk_{idx}_collision_ops"] = pre_routed_ops

    # Capture the idempotency hash BEFORE mutating the chunk.
    # COLLISION re-routes concepts based on what is currently in the vault,
    # which changes between a partial run and its resume (done chunks have
    # already written their notes).  Hashing the pre-COLLISION chunk means
    # the key is stable across runs with the same source input.
    import json as _json
    fsm.context[f"chunk_{idx}_input_hash"] = hashlib.sha256(
        _json.dumps(chunk, sort_keys=True).encode()
    ).hexdigest()

    # Replace chunk with filtered version (remove patched/deferred concepts)
    fsm._chunks[idx] = {
        "schema_version": chunk.get("schema_version", 1),
        "batches": modified_batches,
    }

    fsm._progress_note(
        fsm._chunk_task_id("collision"), "collision", "done",
        output_ref=f"{len(pre_routed_ops)} patch-routed, {len(deferred_concepts)} deferred",
    )
    fsm._transition_success()
