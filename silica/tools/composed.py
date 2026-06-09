"""Composed tools — L2/L3 logic promoted to system tools.

From SILICA.md §4.2:
  Composed tools encode mechanical workflows that span multiple atomic operations.
  These are the former Python scripts from Hermes (recon, payload, validate, etc.)
  refactored to use DRIVER instead of os.walk or open().
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from silica.driver import DRIVER
from silica.kernel.ops import Op, OpType
from silica.kernel.ops_io import load_ops, dump_ops
from silica.tools import tool


def _same_note(ref_a, ref_b) -> bool:
    """Path-safe comparison between two NoteRefs — handles slashes, casing, and .md suffix."""
    import os
    def norm(r):
        p = r.path or r.name
        return os.path.normcase(p.replace("\\", "/").removesuffix(".md").strip("/"))
    return norm(ref_a) == norm(ref_b)


class ReconArgs(BaseModel):
    inbox_file: str = Field(description="Path to the inbox file to analyze")
    limit: int = Field(default=0, description="Limit for concept extraction")

@tool(ReconArgs, cls="composed")
def silica_recon(inbox_file: str, limit: int = 0) -> dict[str, Any]:
    """Mechanical extraction of concepts from an inbox file and searching for collisions in the vault."""
    from silica.kernel.recon import extract_concepts, is_title_match, rank_hits, collision_priority
    
    try:
        nc = DRIVER.read_note(inbox_file)
    except RuntimeError:
        return {"error": f"File not found: {inbox_file}"}
        
    concepts = extract_concepts(nc.content)
    if not concepts:
        return {"file": inbox_file, "collisions": [], "new_concepts": []}

    collisions = []
    new_concepts = []
    
    for c in concepts:
        # Search the vault for the concept
        hits = DRIVER.search_context(c)
        if not hits:
            new_concepts.append(c)
            continue
            
        # Group hits by ref
        grouped = {}
        for h in hits:
            if h.ref.path and ('/done/' in h.ref.path or h.ref.path.startswith('done/')):
                continue
            if _same_note(h.ref, nc.ref):
                continue
                
            name = h.ref.name
            if name not in grouped:
                grouped[name] = {"ref": h.ref, "count": 0}
            grouped[name]["count"] += 1
            
        if not grouped:
            new_concepts.append(c)
            continue
            
        raw_hits = []
        for name, data in grouped.items():
            in_t = is_title_match(c, name)
            raw_hits.append({
                "path": data["ref"].path or data["ref"].name,
                "count": data["count"],
                "in_title": in_t
            })
            
        ranked = rank_hits(raw_hits)
        collisions.append({
            "name": c,
            "total_hits": sum(h["count"] for h in raw_hits),
            "best_match": "title" if ranked[0]["in_title"] else "body",
            "hits": ranked
        })
        
    collisions.sort(key=collision_priority)
    new_concepts.sort()
    
    return {
        "file": inbox_file,
        "collisions": collisions,
        "new_concepts": new_concepts
    }


class PayloadArgs(BaseModel):
    recon_report_path: str = Field(description="Path to the recon report JSON file")
    max_concepts: int = Field(default=7, description="Maximum concepts per batch")
    max_bytes: int = Field(default=80 * 1024, description="Maximum bytes (JSON size) per chunk")

@tool(PayloadArgs, cls="composed")
def silica_payload(recon_report_path: str, max_concepts: int = 7, max_bytes: int = 80 * 1024) -> dict[str, Any]:
    """Assembles payloads for the Distiller by pre-extracting snippets from the vault."""
    import orjson
    from silica.kernel.payload import build_payload
    from silica.kernel.partition import partition_by_concepts
    
    try:
        with open(recon_report_path, 'rb') as f:
            recon_reports = orjson.loads(f.read())
    except Exception as e:
        return {"error": f"Failed to read recon report: {e}"}
        
    # We use a default window of 450 chars
    payload = build_payload(recon_reports, window=450)
    
    # C4/S3.1: Always run partition_by_concepts if we have constraints
    if max_concepts > 0 or max_bytes > 0:
        chunks = partition_by_concepts(payload, max_concepts, max_bytes)
        return {"chunks": chunks}
        
    return {"payload": payload}


class SanitizeArgs(BaseModel):
    distiller_output_path: str = Field(description="Path to the raw distiller output JSON file")

@tool(SanitizeArgs, cls="composed")
def silica_sanitize(distiller_output_path: str) -> dict[str, Any]:
    """Validates and sanitizes the JSON returned by Distiller workers."""
    from silica.kernel.sanitize import parse_json, normalize_ops

    try:
        with open(distiller_output_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
    except Exception as e:
        return {"error": f"Failed to read distiller output: {e}"}

    try:
        parsed_obj, was_clean = parse_json(raw_content, strict=False)
    except Exception as e:
        return {"error": f"JSON Parse Error: {e}"}

    # Normalize op content: strip .md from wikilinks, etc.
    if isinstance(parsed_obj, list):
        parsed_obj = normalize_ops(parsed_obj)
    elif isinstance(parsed_obj, dict) and "updates" in parsed_obj:
        parsed_obj["updates"] = normalize_ops(parsed_obj["updates"])

    # Axis enforcement (Layer 2): demote ops whose linked_axis is not in main_thematic_axes.
    # Only activates when the distiller actually emitted axes — graceful degradation otherwise.
    if isinstance(parsed_obj, dict):
        axes = {a.strip().lower() for a in parsed_obj.get("main_thematic_axes", []) if a}
        if axes:
            for op in parsed_obj.get("updates", []):
                if isinstance(op, dict) and op.get("op") in ("write", "patch"):
                    la = (op.get("linked_axis") or "").strip().lower()
                    if la and la not in axes:
                        op["op"] = "skip"
                        op["reason"] = f"unlinked_axis '{op.get('linked_axis')}' not in main_thematic_axes"

    return {
        "success": True,
        "parsed": parsed_obj,
        "was_clean": was_clean
    }


class ValidateOpsArgs(BaseModel):
    ops_json_path: str = Field(description="Path to the consolidated operations JSON file to validate")
    payload_paths: list[str] = Field(default_factory=list, description="Paths to the original payload JSON files")
    target_dir: str = Field(default="", description="Target folder in the vault")
    hub: str = Field(default="", description="Hub note name")
    future_ref_whitelist: list[str] = Field(default_factory=list, description="Optional whitelist of future reference note names")

@tool(ValidateOpsArgs, cls="composed")
def silica_validate_ops(
    ops_json_path: str,
    payload_paths: list[str] | None = None,
    target_dir: str = "",
    hub: str = "",
    future_ref_whitelist: list[str] | None = None,
) -> dict[str, Any]:
    """Pre-write gate: checks structural validity and applies rejection threshold (10%).

    C4: After validation, OVERWRITES ops_json_path with the coerced + deduped
    validated ops. Snapshot and bulk_write MUST read from the same ops_json_path
    after this call — never from a pre-validation snapshot.
    """
    import orjson
    from silica.kernel.validate import validate_operations

    if payload_paths is None:
        payload_paths = []

    try:
        ops = load_ops(ops_json_path)
    except Exception as e:
        return {"error": f"Failed to load operations: {e}"}

    payloads = []
    for path in payload_paths:
        try:
            with open(path, 'rb') as f:
                payloads.append(orjson.loads(f.read()))
        except Exception as e:
            return {"error": f"Failed to load payload {path}: {e}"}

    cleared_parents: list[dict] = []
    validated_ops, rejected_ops = validate_operations(
        ops,
        payloads,
        target_dir,
        hub=hub,
        cleared_parents_out=cleared_parents,
        future_ref_whitelist=future_ref_whitelist,
    )

    total = len(ops)
    rejected_count = len(rejected_ops)
    # C4 denominator: skip ops excluded from rejection rate
    actionable = sum(1 for o in ops if o.op != OpType.skip)
    rejection_rate = rejected_count / actionable if actionable > 0 else 0.0

    # C4: Always overwrite ops_json_path with validated (coerced + deduped) ops —
    # even when rejection_rate exceeds the old 10% threshold.  Policy (abort vs.
    # continue) is the FSM's responsibility; the tool is a pure filter.
    try:
        dump_ops(ops_json_path, validated_ops)
    except Exception as e:
        return {"error": f"Failed to persist validated ops: {e}"}

    return {
        "success": True,
        "total": total,
        "validated_count": len(validated_ops),
        "rejected_count": rejected_count,
        "rejection_rate": rejection_rate,
        "validated_ops": [o.model_dump() for o in validated_ops],
        "rejected_ops": [r.model_dump() for r in rejected_ops],
        "cleared_parents": cleared_parents,
    }


class BulkWriteArgs(BaseModel):
    ops_json_path: str = Field(description="Path to the validated operations JSON file")

@tool(BulkWriteArgs, cls="composed")
def silica_bulk_write(ops_json_path: str) -> dict[str, Any]:
    """Applies write/patch/overwrite/delete operations in batch in the vault."""
    from silica.kernel.bulk import execute_operations

    try:
        ops = load_ops(ops_json_path)
    except Exception as e:
        return {"error": f"Failed to load operations: {e}"}

    res = execute_operations(ops)
    return res.model_dump()


class PatchNoteArgs(BaseModel):
    name: str = Field(description="Name or vault-relative path of the note to patch")
    heading: str = Field(description="Concept/section heading the snippet is filed under")
    snippet: str = Field(description="Distilled body text to append to the note")
    source_basename: str = Field(description="Provenance: source filename this snippet derives from")
    hub: str | None = Field(default=None, description="Optional [[Hub]] to link in frontmatter if missing")

@tool(PatchNoteArgs, cls="composed")
def silica_patch_note(
    name: str,
    heading: str,
    snippet: str,
    source_basename: str,
    hub: str | None = None,
) -> dict[str, Any]:
    """Append a distilled snippet to a single existing note — fast path for
    interactive edits, no temp-file + bulk_write round-trip.

    Reuses the shared single-op executor (silica.kernel.bulk.execute_one), so
    it stays in lockstep with the batch write path and inherits any future
    write-layer changes (e.g. atomic file writes).

    Undo: every successful patch is recorded on the per-note checkpoint stack,
    so it can be reverted later via the REPL ``/undo`` command. This is a
    lightweight user-facing edit history — it is NOT the FSM's transactional
    snapshot/rollback (which guards a whole pipeline run). Crash-safety of the
    write itself relies on atomic writes at the DRIVER level.
    """
    from silica.kernel.bulk import execute_one
    from silica.kernel.checkpoints import get_checkpoint_store

    # Resolve the note and capture its pre-patch content for the undo floor.
    try:
        nc = DRIVER.read_note(name)
    except Exception as e:
        return {"error": f"Failed to read note '{name}': {e}"}

    path = nc.ref.path or name
    prior_content = nc.content

    op = Op(
        op=OpType.patch,
        heading=heading,
        source_basename=source_basename,
        path=path,
        snippet=snippet,
        hub=hub,
    )

    try:
        result = execute_one(op)
    except Exception as e:
        return {"error": f"Failed to patch '{name}': {e}"}

    # Record the resulting on-disk content as a restore point.
    checkpoint_depth = None
    try:
        new_content = DRIVER.read_note(path).content
        checkpoint_depth = get_checkpoint_store().push(path, prior_content, new_content)
    except Exception:
        # A patch that succeeded must not be reported as failed just because
        # the undo bookkeeping hiccuped; undo is best-effort.
        pass

    return {**result, "note": name, "path": path, "checkpoint_depth": checkpoint_depth}


class LintArgs(BaseModel):
    note_name: str = Field(description="Name of the note to lint")
    op_type: str = Field(default="", description="Operation type (write/patch/overwrite) for conditional checks")
    hub: str = Field(default="", description="Hub note name for wikilink validation")

@tool(LintArgs, cls="composed")
def silica_lint(note_name: str, op_type: str = "", hub: str = "") -> dict[str, Any]:
    """Post-write gate: executes the OFM linter to find structural regressions."""
    from silica.kernel.linter import validate_note

    errors, warnings = validate_note(note_name, hub=hub or None, op_type=op_type or None)

    return {
        "success": len(errors) == 0,
        "note": note_name,
        "errors": errors,
        "warnings": warnings,
    }


class RunInjectorArgs(BaseModel):
    inbox_file: str = Field(default="", description="Path to a single inbox file (legacy; use inbox_files for multiple files)")
    inbox_files: list[str] = Field(default_factory=list, description="Paths to one or more inbox files to ingest in a single run")
    target_dir: str = Field(description="Destination directory for the extracted concepts")
    hub: str = Field(default="", description="Optional reference hub note")
    resume_run_id: str = Field(default="", description="Run ID to resume (re-processes only failed chunks, skips done ones)")

@tool(RunInjectorArgs, cls="composed")
def silica_run_injector(
    inbox_file: str = "",
    inbox_files: list[str] | None = None,
    target_dir: str = "",
    hub: str = "",
    resume_run_id: str = "",
    cancel_token: Any = None,
) -> dict[str, Any]:
    """Execute the entire Injector pipeline deterministically with acceptance gates and rollback.

    Accepts one or more inbox files in a single FSM run with per-chunk failure
    containment: a failed chunk is rolled back and marked 'failed' while the
    remaining chunks continue.  Pass resume_run_id to re-run only the chunks
    that failed in a previous partial run (content-addressed idempotency).

    When sub-agents are enabled (CONFIG.subagents_enabled), the run is driven by
    the Coordinator, which fans borderline-pair dedup work out to leashed
    sub-agents on the worker model concurrently with the injection batches.
    """
    from silica.router.coordinator import Coordinator

    files: list[str] = list(inbox_files or [])
    if inbox_file and inbox_file not in files:
        files.insert(0, inbox_file)
    if not files:
        return {"error": "No inbox file(s) specified"}

    coordinator = Coordinator(
        inbox_files=files,
        target_dir=target_dir,
        hub=hub or None,
        resume_run_id=resume_run_id or None,
        cancel_token=cancel_token,
    )
    return coordinator.run()


# ---------------------------------------------------------------------------
# silica_deferred_retry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# silica_graph_export
# ---------------------------------------------------------------------------

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
        default="Silica Knowledge Graph",
        description="Title shown in the visualization header",
    )

@tool(GraphExportArgs, cls="composed")
def silica_graph_export(output_path: str = "graph.html", folder: str = "", title: str = "Silica Knowledge Graph") -> dict[str, Any]:
    """Generates a self-contained vis.js knowledge graph HTML file from the vault's wikilink structure.

    Runs Louvain community detection to cluster notes by topic.
    Works with both cli and fs backends. Ghost nodes mark unresolved wikilinks.
    The output file can be opened directly in any browser — no server needed.
    """
    from silica.kernel.graph_export import export_graph
    return export_graph(output_path=output_path, folder=folder, title=title)


class AutolinkArgs(BaseModel):
    note_paths: list[str] | None = Field(default=None, description="List of vault-relative paths to autolink")
    note_path: str = Field(default="", description="Vault-relative path of the note to autolink (legacy single-file)")
    use_candidates: bool = Field(default=True, description="Use embedding candidates to focus autolinking (requires index)")

@tool(AutolinkArgs, cls="composed")
def silica_autolink(note_paths: list[str] | None = None, note_path: str = "", use_candidates: bool = True) -> dict[str, Any]:
    """Scan notes for mentions of existing vault titles and wrap them as wikilinks.

    Skips frontmatter, code blocks, math, headings, and already-linked text.
    Only links titles that exist in the vault graph (graph-safe by construction).
    Returns the total number of links added.
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
            from silica.kernel.embed import EmbedStore
            store = EmbedStore()
            if len(store) > 0:
                embedder = get_embedder(CONFIG)
        except Exception:
            pass
        # The co-occurrence leg is embedder-free: load it independently so
        # candidates survive (focused) even when the embedder is down.
        try:
            from silica.config import CONFIG
            from silica.kernel.cooccurrence import CooccurStore
            cooccur_store = CooccurStore(lang=CONFIG.cooccurrence_lang)
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

@tool(BacklinkArgs, cls="composed")
def silica_backlink(new_titles: list[str], neighbourhood: list[str]) -> dict[str, Any]:
    """Inject wikilinks to newly-created notes into pre-existing neighbouring notes.

    For each note in `neighbourhood`, wraps mentions of any title in `new_titles`
    with a wikilink — the reverse direction of AUTOLINK.  Skips frontmatter, code,
    math, and already-linked spans.  Returns {path: [titles_added]}.
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


class SemanticSearchArgs(BaseModel):
    query: str = Field(description="Free-form query text to embed and search against the vault index")
    k: int = Field(default=5, description="Number of results to return")

@tool(SemanticSearchArgs, cls="composed")
def silica_semantic_search(query: str, k: int = 5) -> dict[str, Any]:
    """Find vault notes semantically similar to a query using the embedding index.

    Embeddings PROPOSE candidates; the graph DISPOSES (verify links with the driver).
    Returns at most k results ordered by cosine similarity, highest first.
    Requires the embedding index to be built first with silica_embed_refresh.
    """
    from silica.agent.providers import get_embedder
    from silica.config import CONFIG
    from silica.kernel.embed import EmbedStore

    store = EmbedStore()
    if len(store) == 0:
        return {"error": "Embedding index is empty. Run silica_embed_refresh to build it first."}

    try:
        embedder = get_embedder(CONFIG)
        vecs = embedder.embed([query])
    except Exception as e:
        return {"error": f"Embedding call failed: {e}"}

    results = store.cosine_top_k(vecs[0], k=k)
    return {"query": query, "results": results}


class SimilarArgs(BaseModel):
    text: str = Field(description="Text to find similar notes for (title, snippet, or concept description)")
    k: int = Field(default=5, description="Number of results to return")

@tool(SimilarArgs, cls="composed")
def silica_similar(text: str, k: int = 5) -> dict[str, Any]:
    """Find vault notes semantically similar to an arbitrary text snippet.

    Equivalent to silica_semantic_search but signals the intent of finding
    notes *similar to* a specific text rather than searching by intent.
    Requires the embedding index to be built first with silica_embed_refresh.
    """
    from silica.agent.providers import get_embedder
    from silica.config import CONFIG
    from silica.kernel.embed import EmbedStore

    store = EmbedStore()
    if len(store) == 0:
        return {"error": "Embedding index is empty. Run silica_embed_refresh to build it first."}

    try:
        embedder = get_embedder(CONFIG)
        vecs = embedder.embed([text])
    except Exception as e:
        return {"error": f"Embedding call failed: {e}"}

    results = store.cosine_top_k(vecs[0], k=k)
    return {"text": text[:120], "results": results}


class EmbedRefreshArgs(BaseModel):
    folder: str = Field(default="", description="Vault-relative folder to restrict indexing (empty = entire vault)")
    force: bool = Field(default=False, description="Re-embed all notes, even if already indexed")

@tool(EmbedRefreshArgs, cls="composed")
def silica_embed_refresh(folder: str = "", force: bool = False) -> dict[str, Any]:
    """Build or refresh the vault embedding index at ~/.silica/index/embeddings.json.

    Incrementally skips notes already in the index (unless force=True).
    Call this after bulk writes to keep the index fresh.
    The driver reads each note to get its content; works with both cli and fs backends.
    """
    from silica.agent.providers import get_embedder
    from silica.config import CONFIG
    from silica.kernel.embed import build_index

    try:
        all_refs = DRIVER.list_files(folder or None)
    except Exception as e:
        return {"error": f"Failed to list vault files: {e}"}

    from silica.kernel.media import preprocess_text
    notes: list[tuple[str, str, str]] = []
    errors: list[str] = []
    for ref in all_refs:
        path = ref.path or ref.name
        name = ref.name or path
        try:
            nc = DRIVER.read_note(path)
            body = preprocess_text(nc.content or "")
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

@tool(CooccurrenceRefreshArgs, cls="composed")
def silica_cooccurrence_refresh(folder: str = "", force: bool = False) -> dict[str, Any]:
    """Build or refresh the vault co-occurrence index at ~/.silica/index/cooccurrence.json.

    The embedder-free twin of silica_embed_refresh: a deterministic concept
    co-occurrence graph derived purely from note text — no LM Studio, no network.
    Incrementally skips notes already indexed (unless force=True). Run this once
    to seed an existing vault; the post-write hook then keeps it fresh.
    Powers the relatedness facade's co-occurrence leg and the graph delta report.
    """
    from silica.config import CONFIG
    from silica.kernel.cooccurrence import build_index

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

    try:
        store = build_index(notes, lang=CONFIG.cooccurrence_lang, force=force)
    except Exception as e:
        return {"error": f"Index build failed: {e}", "read_errors": errors}

    # Garbage collection: remove stale paths from the store
    current_paths = {idx_path for idx_path, _, _ in notes}
    stale_paths = [
        p for p in store.paths()
        if _in_folder(p, folder) and p not in current_paths
    ]
    for p in stale_paths:
        store.delete_note(p)
    if stale_paths:
        store.save()

    return {
        "indexed": len(store),
        "total_notes": len(notes),
        "read_errors": len(errors),
        "index_path": str(store._path),
    }


class LedgerDigestArgs(BaseModel):
    run_id: str = Field(default="", description="Run ID to inspect (latest saved run if empty)")

@tool(LedgerDigestArgs, cls="composed")
def silica_ledger_digest(run_id: str = "") -> dict[str, Any]:
    """Returns a compact summary of a run's plan and progress (< 500 tokens).

    Loads TaskLedger (immutable plan) and ProgressLedger (execution state) from
    ~/.silica/runs/<run_id>/. Pass run_id="" to inspect the most recently saved run.
    """
    from silica.planner.progress import ProgressLedger, _RUNS_DIR

    resolved_id = run_id.strip()
    if not resolved_id:
        # Find the most recently modified run directory
        runs_root = _RUNS_DIR
        if not runs_root.exists():
            return {"error": "No runs found in ~/.silica/runs/"}
        candidates = [
            d for d in runs_root.iterdir()
            if d.is_dir() and (d / "ledger.json").exists()
        ]
        if not candidates:
            return {"error": "No runs found in ~/.silica/runs/"}
        latest = max(candidates, key=lambda d: d.stat().st_mtime)
        resolved_id = latest.name

    try:
        ledger = ProgressLedger.load(resolved_id)
    except FileNotFoundError:
        return {"error": f"Run '{resolved_id}' not found"}
    except Exception as e:
        return {"error": f"Failed to load ledger: {e}"}

    return {"run_id": resolved_id, "digest": ledger.digest()}


# ---------------------------------------------------------------------------
# silica_vault_report
# ---------------------------------------------------------------------------

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
    """Deterministic structural audit of the vault.

    Computes god-nodes, surprising cross-cluster connections, orphans, dangling
    links, and clusters. Writes GRAPH_REPORT.md and (if seed_ledger=True)
    persists a run whose ProgressLedger is pre-seeded with remediation tasks
    the agent can advance via silica_ledger_next.

    Tier semantics:
      auto     — reversible, graph-safe ops the agent executes without confirmation
      propose  — reversible but borderline; agent asks before executing
      escalate — IssueCards requiring human judgment (create/rename/delete)
    """
    import orjson
    from pathlib import Path

    from silica.config import CONFIG
    from silica.kernel.graph_report import compute_report, to_digest, to_facts, write_report
    from silica.planner.analyst_plan import build_task_plan
    from silica.planner.progress import IssueCard, ProgressLedger, TaskLedger

    # 1. Build report
    report = compute_report(
        folder=folder, top_k=top_k,
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

    progress = ProgressLedger.new(mode="analyst", inputs={"scope": folder or "vault"})
    run_id = progress.run_id
    run_dir = Path.home() / ".silica" / "runs" / run_id
    payloads_dir = run_dir / "payloads"
    payloads_dir.mkdir(parents=True, exist_ok=True)

    # Persist immutable TaskLedger
    tl = TaskLedger.new(
        run_id=run_id,
        user_request=f"audit {folder or 'vault'}",
        checkpoints=plan.checkpoints,
        facts=to_facts(report),
    )
    try:
        tl.save()
    except Exception:
        pass

    # Seed tasks from auto + propose (propose carries needs_confirmation flag)
    for candidate in plan.auto + plan.propose:
        task = progress.add_task(candidate.capability_name)
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
        progress.issues.append(card)

    progress.save()

    result["run_id"] = run_id
    result["auto"] = len(plan.auto)
    result["propose"] = len(plan.propose)
    result["issues"] = len(plan.escalate)

    return result


class DeferredRetryArgs(BaseModel):
    content_hash: str = Field(description="Content hash of the deferred bundle to retry (from silica_deferred_list)")

@tool(DeferredRetryArgs, cls="composed")
def silica_deferred_retry(content_hash: str) -> dict[str, Any]:
    """Retry writing a deferred op bundle: re-validates against the current vault,
    snapshots, writes the ops that now pass, and updates the bundle.

    - Ops that pass validation are written immediately.
    - Ops that still fail remain in the deferred store.
    - If the bundle is fully cleared, it is removed from the deferred store.
    """
    import os
    from silica.kernel.deferred import get_deferred_store
    from silica.kernel.validate import validate_operations
    from silica.kernel.ops_io import parse_ops, dump_ops
    from silica.tools.wrapped import build_txn
    from silica.kernel.bulk import execute_operations

    store = get_deferred_store()
    bundle = store.get(content_hash)
    if not bundle:
        return {"error": f"No deferred bundle found for hash {content_hash[:8]}…"}

    rejected_raw = bundle.get("rejected_ops", [])
    target_dir = bundle.get("target_dir", "")
    hub = bundle.get("hub")

    try:
        ops = parse_ops(rejected_raw)
    except Exception as e:
        return {"error": f"Failed to parse deferred ops: {e}"}

    validated, still_rejected = validate_operations(ops, [], target_dir, hub=hub)

    if not validated:
        return {
            "success": False,
            "message": "All deferred ops still rejected by the validator",
            "rejected": [
                {"path": r.op.path, "heading": r.op.heading, "reason": r.reason}
                for r in still_rejected
            ],
            "still_deferred": len(still_rejected),
        }

    import uuid
    from silica.kernel.paths import silica_tmp_dir
    tmp_path = str(silica_tmp_dir() / f"{uuid.uuid4().hex}.json")
    try:
        dump_ops(tmp_path, validated)

        # Snapshot before writing for rollback safety
        txn = build_txn(validated)

        result = execute_operations(validated)
        if not result.ok:
            from silica.tools.wrapped import silica_restore
            silica_restore(txn_id=txn.id, inverses=[i.model_dump() for i in txn.inverses])
            failures = [f.model_dump() for f in result.failed]
            return {"error": f"Deferred retry write failed: {failures}"}
    except Exception as e:
        return {"error": f"Deferred retry failed: {e}"}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # Update or clear the deferred store
    if still_rejected:
        store.put(
            content_hash=content_hash,
            source_path=bundle.get("source_path", ""),
            target_dir=target_dir,
            hub=hub,
            rejected_ops=[r.op.model_dump() for r in still_rejected],
            rejection_reasons={
                (r.op.path or r.op.heading or "?"): r.reason for r in still_rejected
            },
        )
    else:
        store.remove(content_hash)

    return {
        "success": True,
        "written": len(validated),
        "still_deferred": len(still_rejected),
        "bundle_cleared": len(still_rejected) == 0,
    }


# ---------------------------------------------------------------------------
# silica_dedup / silica_refine — ad-hoc leashed sub-agent passes (out of pipeline)
# ---------------------------------------------------------------------------

def _in_folder(path: str, folder: str) -> bool:
    """True if vault-rel `path` is inside `folder` (empty folder ⇒ whole vault)."""
    if not folder:
        return True
    f = folder.replace("\\", "/").strip("/").lower()
    p = path.replace("\\", "/").removesuffix(".md").lower()
    return p == f or p.startswith(f + "/")


class DedupPairsArgs(BaseModel):
    pairs: list[dict] = Field(description="List of duplicate pairs to merge. Each dict must have 'source' and 'target' keys.")

@tool(DedupPairsArgs, cls="composed")
def silica_dedup_pairs(pairs: list[dict]) -> dict[str, Any]:
    """Merge a provided list of duplicate note pairs.
    
    Delegates the provided duplicate pairs to the leashed dedup sub-agent batch processor.
    The smaller note is appended to the larger note as a single patch.
    """
    from silica.planner.workqueue import WorkItem
    from silica.agent.subagent import run_subagent_batch
    
    if not pairs:
        return {"error": "No pairs provided."}
        
    items: list[WorkItem] = []
    
    for pair in pairs:
        source = pair.get("source")
        target = pair.get("target")
        score = pair.get("score", 0.0)
        
        if not source or not target:
            continue
            
        try:
            body_src = DRIVER.read_note(source).content or ""
            body_tgt = DRIVER.read_note(target).content or ""
        except Exception:
            continue
            
        # The larger note is the merge target; the smaller is the source of new info.
        if len(body_tgt) >= len(body_src):
            larger, smaller, smaller_body = target, source, body_src
        else:
            larger, smaller, smaller_body = source, target, body_tgt
            
        items.append(WorkItem(
            kind="dedup",
            target_path=larger,
            context={
                "concept": smaller.rsplit("/", 1)[-1],
                "excerpt": smaller_body[:4000],
                "candidate": larger.rsplit("/", 1)[-1],
                "score": score,
                "inbox_file": smaller,
            },
            reason=f"ledger_dedup score={score:.3f}",
        ))
        
    if not items:
        return {"success": False, "message": "No valid pairs to process"}
        
    res = run_subagent_batch(items)
    res["pairs_found"] = len(items)
    return res


class DedupFolderArgs(BaseModel):
    folder: str = Field(default="", description="Vault folder to scan for near-duplicate notes (empty = whole vault)")


@tool(DedupFolderArgs, cls="composed")
def silica_dedup(folder: str = "", cancel_token: Any = None) -> dict[str, Any]:
    """Find near-duplicate note pairs and merge the smaller into the larger.

    Uses the embedding index to surface borderline pairs, then runs the leashed
    dedup sub-agent on each pair: it appends only the smaller note's genuinely-new
    info into the larger note (a single append-only patch). Never rewrites/deletes/
    creates. Run /embed first.

    Pair admission criteria (either condition is sufficient):
      • Full-note cosine similarity in (τ_low, τ_high)   ← body-level similarity
      • Title-only cosine similarity ≥ sim_title_threshold ← title-level similarity
    The second criterion catches cases like "ROS" / "JSON in ROS 2" where the bodies
    are topically distinct but the titles share a strong semantic relationship.
    """
    from silica.kernel.embed import EmbedStore, _cosine
    from silica.planner.workqueue import WorkItem
    from silica.agent.subagent import run_subagent_batch
    from silica.config import CONFIG as _C

    store = EmbedStore()
    if len(store) == 0:
        return {"error": "Embedding index empty — run /embed first."}

    τ_high = getattr(_C, "sim_threshold_high", 0.85)
    τ_low = getattr(_C, "sim_threshold_low", 0.65)
    τ_title = getattr(_C, "sim_title_threshold", 0.80)

    scope = [p for p in store.paths() if _in_folder(p, folder)]
    seen_pairs: set[tuple[str, str]] = set()
    items: list[WorkItem] = []

    for p in scope:
        vec = store.get_vec(p)
        if not vec:
            continue
        candidates = store.cosine_top_k(vec, k=_C.dedup_scan_k, exclude={p})
        for match in candidates:
            score = match.get("score", 0.0)
            other = match.get("path", "")
            if not other or not _in_folder(other, folder):
                continue

            # Title-level similarity gate: catches pairs whose bodies diverge
            # but whose titles share a strong semantic relationship.
            title_vec_p = store.get_title_vec(p)
            title_vec_o = store.get_title_vec(other)
            title_score = (
                _cosine(title_vec_p, title_vec_o)
                if title_vec_p and title_vec_o
                else 0.0
            )

            in_full_window = τ_low < score < τ_high
            in_title_gate  = title_score >= τ_title

            # continue (not break): list is sorted descending; a match above τ_high
            # arrives before borderline ones — break would kill the loop too early.
            if not in_full_window and not in_title_gate:
                continue

            key = tuple(sorted((p, other)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            try:
                body_p = DRIVER.read_note(p).content or ""
                body_o = DRIVER.read_note(other).content or ""
            except Exception:
                continue

            # The larger note is the merge target; the smaller is the source of new info.
            if len(body_o) >= len(body_p):
                larger, smaller, smaller_body = other, p, body_p
            else:
                larger, smaller, smaller_body = p, other, body_o

            effective_score = max(score, title_score)
            items.append(WorkItem(
                kind="dedup",
                target_path=larger,
                context={
                    "concept": smaller.removesuffix(".md").rsplit("/", 1)[-1],
                    "excerpt": smaller_body[:4000],
                    "candidate": larger.removesuffix(".md").rsplit("/", 1)[-1],
                    "score": effective_score,
                    "full_score": score,
                    "title_score": title_score,
                    "inbox_file": smaller,
                },
                reason=f"folder_dedup score={effective_score:.3f} (full={score:.3f} title={title_score:.3f})",
            ))

    res = run_subagent_batch(items, cancel_token=cancel_token)
    res["pairs_found"] = len(items)
    res["folder"] = folder or "(vault)"
    return res


class RefineBatchArgs(BaseModel):
    note_paths: list[str] = Field(description="List of vault-relative paths to stylistically refine.")

@tool(RefineBatchArgs, cls="composed")
def silica_refine_batch(note_paths: list[str], cancel_token: Any = None) -> dict[str, Any]:
    """Stylistically refine a batch of notes (leashed refiner sub-agent).

    Each note is reformatted for clarity/Obsidian style WITHOUT information loss.
    """
    if not note_paths:
        return {"error": "No note paths provided."}

    from silica.planner.workqueue import WorkItem
    from silica.agent.subagent import run_subagent_batch

    items = [WorkItem(kind="refine", target_path=p, context={}) for p in note_paths]
    res = run_subagent_batch(items, cancel_token=cancel_token)
    res["notes"] = len(items)
    return res

class EnrichBatchArgs(BaseModel):
    note_paths: list[str] = Field(description="List of vault-relative paths to semantically enrich.")

@tool(EnrichBatchArgs, cls="composed")
def silica_enrich_batch(note_paths: list[str], cancel_token: Any = None) -> dict[str, Any]:
    """Semantically enrich a batch of lean or empty notes (leashed enricher sub-agent)."""
    if not note_paths:
        return {"error": "No note paths provided."}

    from silica.planner.workqueue import WorkItem
    from silica.agent.subagent import run_subagent_batch

    items = [WorkItem(kind="enrich", target_path=p, context={}) for p in note_paths]
    res = run_subagent_batch(items, cancel_token=cancel_token)
    res["notes"] = len(items)
    return res


# ---------------------------------------------------------------------------
# silica_generate_taxonomy
# ---------------------------------------------------------------------------

class GenerateTaxonomyArgs(BaseModel):
    user_intent: str = Field(description="Natural-language description of how the user wants to organize their vault")
    scope: str = Field(
        default="",
        description="Vault-relative subfolder to restrict taxonomy generation and scanning to",
    )
    save_path: str = Field(
        default="",
        description=(
            "Vault-relative path where the taxonomy YAML should be written. "
            "Defaults to '_silica/taxonomy.yaml' inside the configured vault."
        ),
    )

@tool(GenerateTaxonomyArgs, cls="composed")
def silica_generate_taxonomy(user_intent: str, scope: str = "", save_path: str = "") -> dict[str, Any]:
    """Generate a taxonomy YAML from a natural-language organization intent.

    Uses the LLM to translate the user's description into a structured
    FolderRule list, validates it with Pydantic, and writes it to disk
    at _silica/taxonomy.yaml (or the specified path).

    Returns the validated taxonomy dict and the path it was written to.
    The user should review the output before running silica_run_organizer.
    """
    from pathlib import Path

    from silica.agent.llm import call_llm
    from silica.config import CONFIG
    from silica.kernel.sanitize import parse_json
    from silica.kernel.taxonomy import (
        TAXONOMY_GENERATION_PROMPT,
        Taxonomy,
        default_taxonomy_path,
    )
    from silica.driver import DRIVER

    note_titles: list[str] = []
    try:
        refs = DRIVER.list_files(scope or "")
        note_titles = [
            Path(ref.path or ref.name).stem
            for ref in refs
            if (ref.path or ref.name).endswith(".md")
        ]
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("silica_generate_taxonomy: failed to list files for scope %s: %s", scope, exc)

    # Format the titles clearly (e.g., max 400 titles to prevent prompt bloat)
    titles_summary = "\n".join(f"- {t}" for t in note_titles[:400])
    if len(note_titles) > 400:
        titles_summary += f"\n- ... and {len(note_titles) - 400} more notes."

    system_prompt = "You are an expert knowledge manager. Follow the user instructions exactly."
    user_msg = TAXONOMY_GENERATION_PROMPT.format(
        user_intent=user_intent,
        scope=scope or "Entire Vault",
        note_titles=titles_summary or "(No notes found in scope)",
    )

    try:
        response = call_llm(
            model=CONFIG.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            tools=None,
        )
        raw = (response.text or "").strip()
    except Exception as exc:
        return {"error": f"LLM call failed: {exc}"}

    # The LLM is instructed to return raw YAML; try yaml.safe_load first,
    # then fall back to parse_json for robustness.
    import yaml as _yaml

    taxonomy_dict: dict | None = None
    try:
        taxonomy_dict = _yaml.safe_load(raw)
    except Exception:
        pass

    if not isinstance(taxonomy_dict, dict):
        parsed, _ = parse_json(raw, strict=False)
        if isinstance(parsed, dict):
            taxonomy_dict = parsed
        else:
            return {"error": f"LLM returned unparseable output: {raw[:300]}"}

    try:
        taxonomy = Taxonomy.from_dict(taxonomy_dict)
    except Exception as exc:
        return {"error": f"Taxonomy validation failed: {exc}", "raw": taxonomy_dict}

    if save_path:
        dest = Path(save_path)
        if not dest.is_absolute() and CONFIG.vault_path:
            dest = Path(CONFIG.vault_path) / dest
    else:
        dest = default_taxonomy_path()
    try:
        taxonomy.to_yaml(dest)
    except Exception as exc:
        return {"error": f"Failed to write taxonomy to {dest}: {exc}"}

    return {
        "success": True,
        "taxonomy_path": str(dest),
        "taxonomy": taxonomy.model_dump(),
        "rules_count": len(taxonomy.rules),
    }


# ---------------------------------------------------------------------------
# silica_run_organizer
# ---------------------------------------------------------------------------

class RunOrganizerArgs(BaseModel):
    taxonomy_path: str = Field(
        default="",
        description="Path to the taxonomy YAML file. Defaults to '_silica/taxonomy.yaml' in the vault.",
    )
    scope: str = Field(
        default="",
        description="Vault-relative subfolder to restrict organization to (empty = vault-wide)",
    )
    dry_run: bool = Field(
        default=True,
        description=(
            "If True (default), compute and return the move plan without executing any moves. "
            "Set to False to actually move notes."
        ),
    )
    llm_arbiter: bool = Field(
        default=True,
        description="If True, use the LLM to classify borderline notes (ambiguous band)",
    )

@tool(RunOrganizerArgs, cls="composed")
def silica_run_organizer(
    taxonomy_path: str = "",
    scope: str = "",
    dry_run: bool = True,
    llm_arbiter: bool = True,
) -> dict[str, Any]:
    """Execute the /organize pipeline — classify vault notes and move them to taxonomy folders.

    Phase 1 (dry_run=True): returns a plan showing which notes would move and where.
    Phase 2 (dry_run=False): executes the moves via DRIVER.move() (graph-safe, wikilinks updated).

    The pipeline runs the OrganizeFSM which:
      1. SCAN   — lists notes in scope
      2. CLASSIFY — L1 co-occurrence matching (zero LLM cost)
      3. ARBITRATE — LLM arbiter for borderline notes (optional)
      4. PLAN   — generates MoveOps for notes that need relocation
      5. SNAPSHOT — captures pre-move state for rollback
      6. MOVE   — executes DRIVER.move() calls
      7. LINT   — graph regression gate
      8. CLEANUP — ledger commit

    Rollback is automatic if the LINT gate fails.
    """
    from silica.kernel.taxonomy import load_taxonomy
    from silica.router.organize_fsm import OrganizerFSM

    taxonomy = load_taxonomy(taxonomy_path or None)
    if not taxonomy.rules:
        return {
            "error": (
                "Taxonomy has no rules. Run silica_generate_taxonomy first or "
                "create _silica/taxonomy.yaml manually."
            )
        }

    fsm = OrganizerFSM(
        taxonomy=taxonomy,
        scope=scope,
        dry_run=dry_run,
        llm_arbiter=llm_arbiter,
    )
    return fsm.run()

