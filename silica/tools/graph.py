# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Graph & relatedness tools — indexes, search, linking, and the vault audit.

Embedding and co-occurrence index refresh, semantic search, autolink/backlink
passes, the vis.js graph export, and the structural vault report.
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from silica.driver import DRIVER
from silica.tools import tool
from silica.tools.atomic import EmptyArgs

logger = logging.getLogger(__name__)


def _in_folder(path: str, folder: str) -> bool:
    """True if vault-rel `path` is inside `folder` (empty folder ⇒ whole vault)."""
    if not folder:
        return True
    f = folder.replace("\\", "/").strip("/").lower()
    p = path.replace("\\", "/").removesuffix(".md").lower()
    return p == f or p.startswith(f + "/")


class GraphExportArgs(BaseModel):
    output_path: str = Field(
        default="graph.html",
        description="Filesystem path for the output HTML file (e.g. 'graph.html' or '/tmp/vault_graph.html')",
    )
    folder: str = Field(
        default="",
        description="Vault-relative folder to restrict scope (empty = entire vault)",
    )
    title: str = Field(
        default="Vault Graph",
        description="Title shown in the visualization header",
    )

@tool(GraphExportArgs, cls="composed")
def silica_graph_export(output_path: str = "graph.html", folder: str = "", title: str = "Vault Graph") -> dict[str, Any]:
    """Generates a self-contained interactive HTML graph of the vault's wikilink structure.

    Runs Louvain community detection to cluster notes by topic; ghost nodes mark
    unresolved wikilinks. The output opens directly in any browser.
    Visualization only — for an actionable structural audit use silica_vault_report.
    """
    from silica.ui.web.graph_view import export_graph

    # Best-effort: refresh the co-occurrence index so clusters get named labels
    # (incremental — skips already-indexed notes). Naming degrades to "Cluster N"
    # if this fails; the graph still renders. ponytail: full-vault refresh, scope
    # to changed notes only if it gets slow on big vaults.
    try:
        silica_cooccurrence_refresh(folder=folder)
    except Exception as exc:
        logger.warning("silica_graph_export: cooccurrence refresh skipped (%s)", exc)

    return export_graph(output_path=output_path, folder=folder, title=title)


class MindmapArgs(BaseModel):
    note_path: str = Field(description="Vault-relative path of the note to root the map on")
    force: bool = Field(default=False, description="Overwrite an existing maps/<stem>.canvas (defaults to no-clobber)")

@tool(MindmapArgs, cls="composed")
def silica_mindmap(note_path: str, force: bool = False) -> dict[str, Any]:
    """Builds a radial mind-map rooted on one note and writes it as an Obsidian .canvas.

    Deterministic, no LLM: BFS over the wikilink graph plus the latent (embeddings
    + co-occurrence) relatedness leg, laid out as radial wedges by community. The
    .canvas lands in maps/<stem>.canvas and is manipulable in Obsidian. No-clobber:
    an existing map is not overwritten unless force=True (so your rearrangements
    survive). For the flat whole-vault network instead, use silica_graph_export.
    """
    from pathlib import Path

    from silica.config import CONFIG
    from silica.kernel.mindmap import (
        build_mapview,
        gather_materials,
        mapview_to_canvas,
        resolve_note_path,
    )

    # Accept a path OR a title (the GUI input and casual CLI use give titles).
    root = resolve_note_path(note_path)
    if root is None:
        return {"error": f"'{note_path}' not found in the vault graph."}

    stem = Path(root).stem
    vault = CONFIG.vault_path or "."
    out = Path(vault) / "maps" / f"{stem}.canvas"

    # ponytail: no-clobber v1 = exists + not force → refuse. Diffing the generated
    # map against a user-rearranged one is v2; here we simply never clobber.
    if out.exists() and not force:
        return {"skipped": str(out), "reason": "exists", "hint": "re-run with force=True to regenerate"}

    materials = gather_materials(root, latent_k=CONFIG.mindmap_latent_k)
    mv = build_mapview(
        root, materials, max_nodes=CONFIG.mindmap_max_nodes, hops=CONFIG.mindmap_hops
    )
    if len(mv.nodes) <= 1:
        return {"error": f"'{root}' has no neighbours to map (isolated in the graph)."}

    import orjson
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(orjson.dumps(mapview_to_canvas(mv), option=orjson.OPT_INDENT_2))
    logger.info("silica_mindmap: wrote %s — %d nodes, %d edges", out, len(mv.nodes), len(mv.edges))
    return {"path": str(out), "nodes": len(mv.nodes), "edges": len(mv.edges)}


class AutolinkArgs(BaseModel):
    note_paths: list[str] | None = Field(default=None, description="List of vault-relative paths to autolink")
    note_path: str = Field(default="", description="Vault-relative path of the note to autolink (legacy single-file)")
    use_candidates: bool = Field(default=True, description="Use embedding candidates to focus autolinking (requires index)")

@tool(AutolinkArgs, cls="composed", collapse="eager")
def silica_autolink(note_paths: list[str] | None = None, note_path: str = "", use_candidates: bool = True) -> dict[str, Any]:
    """Scan the given notes for mentions of existing vault titles and wrap them as wikilinks.

    Skips frontmatter, code blocks, math, headings, and already-linked text.
    Only links titles that exist in the vault graph (graph-safe by construction).
    Returns the total number of links added.

    For the reverse direction (inject links TO newly created notes into older
    neighbours) use silica_backlink; for a vault-wide maintenance pass that also
    finds the candidates itself, use silica_curate.
    """
    from silica.kernel.autolink import build_title_index

    paths = note_paths or []
    if note_path and note_path not in paths:
        paths.append(note_path)

    if not paths:
        return {"error": "No note paths provided."}

    try:
        all_refs = DRIVER.list_files()
    except Exception as e:
        return {"error": f"Failed to list vault files: {e}"}

    title_index = build_title_index(all_refs)
    
    store = None
    embedder = None
    cooccur_store = None
    if use_candidates:
        try:
            from silica.agent.providers import get_embedder
            from silica.config import CONFIG
            from silica.kernel.embed import get_store
            store = get_store()
            if len(store) > 0:
                embedder = get_embedder(CONFIG)
        except Exception:
            pass
        # The co-occurrence leg is embedder-free: load it independently so
        # candidates survive (focused) even when the embedder is down.
        try:
            from silica.config import CONFIG
            from silica.kernel.cooccurrence import get_cooccur_store
            cooccur_store = get_cooccur_store(lang=CONFIG.cooccurrence_lang)
            if len(cooccur_store) == 0:
                cooccur_store = None
        except Exception:
            cooccur_store = None

    total_added = 0
    processed = 0
    write_errors: list[str] = []

    for path in paths:
        try:
            nc = DRIVER.read_note(path)
        except Exception:
            continue

        body = nc.content or ""
        if not body.strip():
            continue

        candidates: list[str] | None = None
        if use_candidates and (cooccur_store is not None or (store is not None and embedder is not None)):
            query_vec = None
            if store is not None and embedder is not None:
                try:
                    query_vec = embedder.embed([body[:800]])[0]
                except Exception:
                    query_vec = None
            try:
                from silica.kernel.relatedness import related_notes_for_query
                related = related_notes_for_query(
                    query_vec=query_vec,
                    query_text=body,
                    embed_store=store,
                    cooccur_store=cooccur_store,
                    k=20,
                )
                # Only narrow to candidates when the facade actually proposed
                # some; an empty list would suppress linking, so leave it None
                # to fall back to the full title_index scan.
                if related:
                    candidates = [r.name for r in related]
            except Exception:
                pass  # fall back to full title_index scan

        try:
            added = DRIVER.autolink_note(
                path, candidates=candidates if candidates is not None else title_index
            )
            if added:
                total_added += len(added)
                processed += 1
        except Exception as e:
            write_errors.append(f"{path}: {e}")

    result = {"notes_processed": processed, "total_links_added": total_added}
    if write_errors:
        result["write_errors"] = write_errors
    return result


class BacklinkArgs(BaseModel):
    new_titles: list[str] = Field(description="Titles of notes just created in this run")
    neighbourhood: list[str] = Field(description="Vault-relative paths of candidate notes to scan")

@tool(BacklinkArgs, cls="composed", collapse="eager")
def silica_backlink(new_titles: list[str], neighbourhood: list[str]) -> dict[str, Any]:
    """Inject wikilinks to newly-created notes into pre-existing neighbouring notes.

    For each note in `neighbourhood`, wraps mentions of any title in `new_titles`
    with a wikilink — the reverse of silica_autolink. Skips frontmatter, code,
    math, and already-linked spans. Returns {path: [titles_added]}.
    """
    from silica.kernel.autolink import backlink_pass, build_title_index

    try:
        all_refs = DRIVER.list_files()
    except Exception as e:
        return {"error": f"Failed to list vault files: {e}"}

    title_index = build_title_index(all_refs)
    added_map = backlink_pass(new_titles, title_index=title_index, neighbourhood=neighbourhood)
    total = sum(len(v) for v in added_map.values())
    return {"added": total, "notes_modified": len(added_map), "details": added_map}


def _facade_search(text: str, k: int) -> dict[str, Any]:
    """Fused embeddings + co-occurrence search for a fresh text, then reranked.

    Shared core of silica_semantic_search and silica_similar. Returns
    ``{"results": [{path, name, score}, ...]}`` or ``{"error": ...}`` when no
    index is available at all. The two legs abstain independently: an empty
    embedding index (or an offline embedder) still serves co-occurrence results,
    and vice versa — mirroring how autolink/collision consume the facade.
    """
    from silica.agent.providers import get_embedder, get_reranker
    from silica.config import CONFIG
    from silica.kernel.cooccurrence import get_cooccur_store
    from silica.kernel.embed import get_store
    from silica.kernel.relatedness import related_notes_for_query
    from silica.kernel.rerank import rerank_related

    embed_store = get_store()
    try:
        cooccur_store = get_cooccur_store(lang=CONFIG.cooccurrence_lang)
        if len(cooccur_store) == 0:
            cooccur_store = None
    except Exception:
        cooccur_store = None

    query_vec = None
    if len(embed_store) > 0:
        try:
            query_vec = get_embedder(CONFIG).embed([text])[0]
        except Exception:
            query_vec = None  # embed leg abstains; co-occurrence may still carry

    if query_vec is None and cooccur_store is None:
        return {"error": "No index available. Run silica_embed_refresh or silica_cooccurrence_refresh first."}

    reranker = get_reranker(CONFIG)
    pool = max(k, 20) if reranker else k
    results = related_notes_for_query(
        query_vec=query_vec,
        query_text=text,
        embed_store=embed_store,
        cooccur_store=cooccur_store,
        k=pool,
    )
    results = rerank_related(reranker, text, results, k=k) if reranker else results[:k]
    return {"results": [{"path": r.path, "name": r.name, "score": round(r.score, 4)} for r in results]}


class SemanticSearchArgs(BaseModel):
    query: str = Field(description="Free-form query text to embed and search against the vault index")
    k: int = Field(default=5, description="Number of results to return")

@tool(SemanticSearchArgs, cls="composed")
def silica_semantic_search(query: str, k: int = 5) -> dict[str, Any]:
    """Find vault notes by MEANING: fuses embeddings + co-occurrence, then reranks.

    Use for "what do I have about X" when the exact wording is unknown. Routes
    through the same relatedness facade as autolink/collision — RRF fusion of the
    embedding and co-occurrence legs, cross-encoder reranked when configured — so
    a leg that is down (empty embedding index, embedder offline) degrades to the
    survivor instead of failing. For literal text matches use
    silica_search_context; to rank against a longer text you already have use
    silica_similar. Returns at most k results, best first; verify with
    silica_read_note before acting on them.
    """
    return {"query": query, **_facade_search(query, k=k)}


class SimilarArgs(BaseModel):
    text: str = Field(description="Text to find similar notes for (title, snippet, or concept description)")
    k: int = Field(default=5, description="Number of results to return")

@tool(SimilarArgs, cls="composed")
def silica_similar(text: str, k: int = 5) -> dict[str, Any]:
    """Find vault notes semantically similar to a given text snippet.

    Same relatedness facade as silica_semantic_search (fused embeddings +
    co-occurrence, reranked), but framed for a longer text (a note body, a
    paragraph) rather than a short query — use it for "which notes resemble this
    content". A down leg degrades to the survivor instead of failing. When the text
    IS an existing note, prefer silica_related (it takes the note directly and adds
    the note-edges leg).
    """
    return {"text": text[:120], **_facade_search(text, k=k)}


class RelatedArgs(BaseModel):
    note: str = Field(description="Note name (wikilink-style) or vault-relative path to find related notes for")
    k: int = Field(default=5, description="Number of results to return")

@tool(RelatedArgs, cls="composed")
def silica_related(note: str, k: int = 5) -> dict[str, Any]:
    """Given an EXISTING note (by name or path), the notes most related to it.

    Fuses three graph metrics over the whole vault — embeddings + co-occurrence +
    direct note-edges (CORRELATE) — into one ranked shortlist with provenance, so
    you get a bounded set of candidates instead of guessing. Use this when asked
    "what's related/relevant to note X" INSTEAD of reading X and keyword-searching
    from its words. For free-form text that is not a note, use silica_similar. Each
    result carries `evidence` (embed:0.83, cooccur:w9, edge:0.57) naming which metric
    proposed it; verify with silica_read_note before acting.
    """
    from silica.config import CONFIG
    from silica.driver import DRIVER
    from silica.kernel.cooccurrence import cooccur_key, get_cooccur_store
    from silica.kernel.embed import get_store
    from silica.kernel.relatedness import related_notes

    # Resolve name-or-path to the canonical vault path (any backend), then reduce to
    # the store keyspace via cooccur_key (strip .md, posix, CASE-PRESERVED). This is
    # the single source of truth for both index keyspaces: it makes the query hit the
    # stored vectors/nodes AND lets related_notes exclude the query itself (blocking
    # a raw ".md" path would let the note resurface among its own results). Never
    # _norm_path here — its lowercasing misses the case-preserving stored keys.
    try:
        query_path = DRIVER.read_note(note).ref.path
        resolved = bool(query_path)  # a backend returning '' must not pass as resolved
    except Exception:
        resolved = False
    if not resolved:
        query_path = note  # unresolved: treat the input itself as a path
    query_path = cooccur_key(query_path)

    embed_store = get_store()
    try:
        cooccur_store = get_cooccur_store(lang=CONFIG.cooccurrence_lang)
        if len(cooccur_store) == 0:
            cooccur_store = None
    except Exception:
        cooccur_store = None
    if len(embed_store) == 0 and cooccur_store is None:
        return {"note": note, "error": "No index available. Run silica_embed_refresh or silica_cooccurrence_refresh first."}

    results = related_notes(query_path, embed_store=embed_store, cooccur_store=cooccur_store, k=k)
    out: dict[str, Any] = {
        "note": note,
        "results": [
            {"path": r.path, "name": r.name, "score": round(r.score, 4), "evidence": r.evidence}
            for r in results
        ],
    }
    if not results:
        # Empty is ambiguous — say why so the caller acts instead of guessing.
        if not resolved:
            out["hint"] = f"note '{note}' did not resolve to a vault note — check the name/path."
        elif len(embed_store) == 0:
            out["hint"] = "embedding index empty — co-occurrence only. Run silica_embed_refresh for semantic neighbors."
    return out


class EmbedRefreshArgs(BaseModel):
    folder: str = Field(default="", description="Vault-relative folder to restrict indexing (empty = entire vault)")
    force: bool = Field(default=False, description="Re-embed all notes, even if already indexed")

@tool(EmbedRefreshArgs, cls="composed", collapse="eager")
def silica_embed_refresh(folder: str = "", force: bool = False) -> dict[str, Any]:
    """Build or refresh the vault embedding index.

    Powers silica_semantic_search, silica_similar, and silica_dedup — run it
    first if those report an empty index. Incremental: skips notes already
    indexed (unless force=True). Call after bulk writes to keep it fresh.
    """
    from silica.agent.providers import get_embedder
    from silica.config import CONFIG
    from silica.kernel.embed import build_index

    try:
        all_refs = DRIVER.list_files(folder or None)
    except Exception as e:
        return {"error": f"Failed to list vault files: {e}"}

    from silica.kernel.media import strip_images
    notes: list[tuple[str, str, str]] = []
    errors: list[str] = []
    for ref in all_refs:
        path = ref.path or ref.name
        name = ref.name or path
        try:
            nc = DRIVER.read_note(path)
            body = strip_images(nc.content or "")
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue
        # Strip .md extension for index key
        idx_path = path.removesuffix(".md")
        notes.append((idx_path, name, body))

    if not notes:
        return {"error": "No notes found to index", "read_errors": errors}

    try:
        embedder = get_embedder(CONFIG)
        store = build_index(embedder, notes, force=force)
    except Exception as e:
        return {"error": f"Index build failed: {e}", "read_errors": errors}

    # Garbage collection: remove stale paths from the store
    current_paths = {idx_path for idx_path, _, _ in notes}
    stale_paths = [
        p for p in store.paths()
        if _in_folder(p, folder) and p not in current_paths
    ]
    for p in stale_paths:
        store.delete(p)
    if stale_paths:
        store.save()

    return {
        "indexed": len(store),
        "total_notes": len(notes),
        "read_errors": len(errors),
        "index_path": str(store._path),
    }


class CooccurrenceRefreshArgs(BaseModel):
    folder: str = Field(default="", description="Vault-relative folder to restrict indexing (empty = entire vault)")
    force: bool = Field(default=False, description="Re-process all notes, even if already indexed")

@tool(CooccurrenceRefreshArgs, cls="composed", collapse="eager")
def silica_cooccurrence_refresh(folder: str = "", force: bool = False) -> dict[str, Any]:
    """Build or refresh the vault co-occurrence index.

    The embedder-free twin of silica_embed_refresh: a deterministic concept
    co-occurrence graph derived purely from note text — works even when the
    embedder is unavailable. Powers cluster naming and the co-occurrence
    signals in silica_vault_report. Incremental: skips notes already indexed
    (unless force=True). Run once to seed an existing vault; writes keep it
    fresh automatically afterwards.
    """
    from silica.config import CONFIG
    from silica.kernel import correlate
    from silica.kernel.cooccurrence import build_index, get_cooccur_store

    try:
        all_refs = DRIVER.list_files(folder or None)
    except Exception as e:
        return {"error": f"Failed to list vault files: {e}"}

    notes: list[tuple[str, str, str]] = []
    errors: list[str] = []
    for ref in all_refs:
        path = ref.path or ref.name
        name = ref.name or path
        try:
            # Pass RAW content: build_contribution strips frontmatter + media itself.
            body = DRIVER.read_note(path).content or ""
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue
        idx_path = path.removesuffix(".md")
        notes.append((idx_path, name, body))

    if not notes:
        return {"error": "No notes found to index", "read_errors": errors}

    store = get_cooccur_store(lang=CONFIG.cooccurrence_lang)
    seeded_before = set(store.paths())
    try:
        # refreeze rides on force: `/cooccur --force` is the deliberate rebuild
        # (and the doctor remedy for a wrong-frozen store language) — it
        # re-processes every note, so re-detecting store.lang here is safe.
        # A plain incremental /cooccur skips already-indexed notes and must NOT
        # refreeze: flipping the language without re-stemming existing
        # contributions would mix stemmers across node keys. save=False: one
        # flush at the end after GC + edge refresh.
        build_index(
            notes, store=store, lang=CONFIG.cooccurrence_lang,
            force=force, refreeze=force, save=False,
        )
    except Exception as e:
        return {"error": f"Index build failed: {e}", "read_errors": errors}

    # Garbage collection: remove stale paths from the store (also prunes their edges)
    current_paths = {idx_path for idx_path, _, _ in notes}
    stale_paths = [
        p for p in store.paths()
        if _in_folder(p, folder) and p not in current_paths
    ]
    for p in stale_paths:
        store.delete_note(p)

    # CORRELATE note_edges (ADR-0013): --force rebuilds the whole graph; a plain
    # incremental /cooccur only recomputes rows for notes seeded this run — the
    # rest have unchanged contributions, so their edges are unchanged too.
    if force:
        correlate.recompute_all_edges(store)
    else:
        new_paths = [idx for idx, _, _ in notes if idx not in seeded_before]
        if new_paths:
            correlate.refresh_edges(store, new_paths)
    store.save()

    return {
        "indexed": len(store),
        "total_notes": len(notes),
        "read_errors": len(errors),
        "index_path": str(store._path),
    }


class VaultReportArgs(BaseModel):
    folder: str = Field(default="", description="Vault-relative folder to scope (empty = whole vault)")
    top_k: int = Field(default=10, description="How many god-nodes / bridges to surface")
    with_embeddings: bool = Field(default=False, description="Also propose missing links via the embedding index")
    with_cooccurrence: bool = Field(default=False, description="Also compute the co-occurrence vs wikilink delta (autolink candidates, stale links, missing hubs) — embedder-free")
    seed_ledger: bool = Field(default=True, description="Persist a run (TaskLedger+ProgressLedger) pre-seeded with remediation tasks")

@tool(VaultReportArgs, cls="composed")
def silica_vault_report(
    folder: str = "",
    top_k: int = 10,
    with_embeddings: bool = False,
    with_cooccurrence: bool = False,
    seed_ledger: bool = True,
) -> dict[str, Any]:
    """Deterministic structural audit of the vault — the entry point for /graph and vault health checks.

    Computes god-nodes, surprising cross-cluster connections, orphans, dangling
    links, and clusters. Writes GRAPH_REPORT.md and (if seed_ledger=True) seeds
    a remediation run to advance task-by-task via silica_ledger_next.
    For a visual graph instead, use silica_graph_export; to go straight to
    executable maintenance work, use silica_curate.

    Tier semantics:
      auto     — reversible, graph-safe ops the agent executes without confirmation
      propose  — reversible but borderline; agent asks before executing
      escalate — IssueCards requiring human judgment (create/rename/delete)
    """
    import orjson
    from pathlib import Path

    from silica.config import CONFIG
    from silica.kernel.graph_report import compute_report, to_digest, to_facts, write_report
    from silica.kernel.analyst_plan import build_task_plan
    from silica.kernel.progress import IssueCard, Run

    # 1. Build report (on-demand /graph: full analytics — god_nodes/bridges/cohesion)
    report = compute_report(
        folder=folder, top_k=top_k, analytics=True,
        with_embeddings=with_embeddings, with_cooccurrence=with_cooccurrence,
    )

    # 2. Determine output path
    vault_path = getattr(CONFIG, "vault_path", None) or ""
    if vault_path:
        report_path = str(Path(vault_path) / "GRAPH_REPORT.md")
    else:
        report_path = "GRAPH_REPORT.md"

    paths = write_report(report, report_path)

    result: dict[str, Any] = {
        "digest": to_digest(report),
        "report_md": paths["path_md"],
    }

    if not seed_ledger:
        return result

    # 3. Build plan and seed ledger
    plan = build_task_plan(report)

    run = Run.new(
        mode="analyst",
        user_request=f"audit {folder or 'vault'}",
        checkpoints=plan.checkpoints,
        inputs={"scope": folder or "vault"},
        facts=to_facts(report),
    )
    payloads_dir = run.payloads_dir

    # Seed tasks from auto + propose (propose carries needs_confirmation flag)
    for candidate in plan.auto + plan.propose:
        task = run.progress.add_task(candidate.capability_name)
        # Write payload to disk
        payload = dict(candidate.payload)
        payload["_reason"] = candidate.reason
        if candidate.tier == "propose":
            payload["needs_confirmation"] = True
        payload_path = str(payloads_dir / f"{task.id}.json")
        Path(payload_path).write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        task.input_ref = payload_path

    # Escalate items → IssueCards
    for i, candidate in enumerate(plan.escalate):
        card = IssueCard(
            task_id=f"issue_{i}",
            question=candidate.reason,
            options=[
                {"label": "create_note", "description": "Create a new note with this title"},
                {"label": "rename_existing", "description": "Rename an existing note to match"},
                {"label": "ignore", "description": "Leave the broken link as-is"},
            ],
        )
        run.progress.issues.append(card)

    run.save()

    result["run_id"] = run.run_id
    result["auto"] = len(plan.auto)
    result["propose"] = len(plan.propose)
    result["issues"] = len(plan.escalate)

    return result


@tool(EmptyArgs, cls="composed")
def silica_health() -> dict[str, Any]:
    """Retrieval + write-path health check — the golden harness's two GATED metrics, live.

    Runs both gated probes against the current vault and its on-disk indexes:
      fusion    — masked-wikilink recovery through the full relatedness facade:
                  recall@10, mrr, embed_coverage, and which legs were live.
                  Low recall or embed_coverage < 1.0 means related/semantic
                  search is degraded — refresh with silica_embed_refresh /
                  silica_cooccurrence_refresh and re-run.
      integrity — differential lint across the 4 write-path transforms
                  (frontmatter round-trip, autolink, fs write→read, sanitize);
                  rate must be exactly 1.0 — anything less means the pipeline
                  CORRUPTS note bodies and writes should stop.

    Full-vault sweep (reads every note): a diagnostic to run on demand, not a
    per-write gate. Numbers are regression trends, not absolute quality claims —
    for a structural audit of the vault's content use silica_vault_report.
    """
    from pathlib import Path

    from silica.config import CONFIG
    from silica.kernel.cooccurrence import get_cooccur_store
    from silica.kernel.embed import get_store
    from silica.kernel.health import fusion_probe, integrity_probe

    vault = Path(getattr(CONFIG, "vault_path", "") or "").expanduser()
    if not vault.is_dir():
        return {"error": "No vault configured."}

    try:
        store = get_cooccur_store(lang=CONFIG.cooccurrence_lang)
    except Exception as e:
        store = None
        fusion: dict[str, Any] = {"error": f"co-occurrence store unavailable ({e}) — run silica_cooccurrence_refresh"}
    if store is not None:
        fusion = fusion_probe(vault, store, embed_store=get_store())

    return {"fusion": fusion, "integrity": integrity_probe(vault)}
