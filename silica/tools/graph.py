"""Graph & relatedness tools — indexes, search, linking, and the vault audit.

Embedding and co-occurrence index refresh, semantic search, autolink/backlink
passes, the vis.js graph export, and the structural vault report.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from silica.driver import DRIVER
from silica.tools import tool


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
    from silica.kernel.analyst_plan import build_task_plan
    from silica.kernel.progress import IssueCard, ProgressLedger, TaskLedger

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
