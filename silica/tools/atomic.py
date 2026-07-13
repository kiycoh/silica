# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Atomic tools — L0 façades, 1:1 on Obsidian CLI commands.

From SILICA.md §4.2:
  Atomic tools are single Obsidian-native operations, 1:1 on a CLI command
  or a pure kernel function. They are the base vocabulary — called by both
  the agent and the pipeline.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from silica.driver import DRIVER
from silica.tools import tool


# ---------------------------------------------------------------------------
# Read / Discovery
# ---------------------------------------------------------------------------

class SearchArgs(BaseModel):
    query: str = Field(description="Text to search for in note names in the vault")

@tool(SearchArgs, cls="atomic")
def silica_search(query: str) -> list:
    """Search for notes by NAME/title match. Returns the names of matching notes.

    For text inside note bodies use silica_search_context; for meaning-based
    search when you don't know the exact words use silica_semantic_search.
    """
    refs = DRIVER.search_names(query)
    return [{"name": r.name, "path": r.path} for r in refs]


class SearchContextArgs(BaseModel):
    query: str = Field(description="Text to search for within the content of vault notes")

@tool(SearchContextArgs, cls="atomic")
def silica_search_context(query: str) -> list:
    """Search note BODIES for exact text; returns snippets with line numbers.

    Use to find literal mentions of a term. When the exact wording is unknown,
    use silica_semantic_search instead; to match note titles use silica_search.
    """
    hits = DRIVER.search_context(query)
    return [
        {"name": h.ref.name, "path": h.ref.path, "line": h.line, "snippet": h.snippet}
        for h in hits
    ]


class ReadNoteArgs(BaseModel):
    name: str = Field(description="Name of the note to read (wikilink style, without file extension)")

@tool(ReadNoteArgs, cls="atomic")
def silica_read_note(name: str) -> str:
    """Reads the complete content of a note in the vault by name (wikilink-style resolution). DO NOT use paths."""
    nc = DRIVER.read_note(name)
    return nc.content


class PropsArgs(BaseModel):
    name: str = Field(description="Name of the note to read the frontmatter properties from")

@tool(PropsArgs, cls="atomic")
def silica_props(name: str) -> dict:
    """Reads the frontmatter properties of a note (saves tokens, does not read the body)."""
    return DRIVER.props_of(name)


class OutlineArgs(BaseModel):
    name: str = Field(description="Name of the note to display the heading tree of")

@tool(OutlineArgs, cls="atomic")
def silica_outline(name: str) -> list:
    """Displays the heading tree (H1-H6) of a note."""
    headings = DRIVER.outline(name)
    return [{"level": h.level, "text": h.text} for h in headings]


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class LinksArgs(BaseModel):
    name: str = Field(description="Name of the note to list outgoing links from")

@tool(LinksArgs, cls="atomic")
def silica_links(name: str) -> list:
    """Lists outgoing links from a note (connected notes)."""
    refs = DRIVER.links(name)
    return [{"name": r.name, "path": r.path} for r in refs]


class BacklinksArgs(BaseModel):
    name: str = Field(description="Name of the note to list incoming links (backlinks) for")

@tool(BacklinksArgs, cls="atomic")
def silica_backlinks(name: str) -> list:
    """Lists incoming links (backlinks) pointing to a note."""
    refs = DRIVER.backlinks(name)
    return [{"name": r.name, "path": r.path} for r in refs]


class EmptyArgs(BaseModel):
    pass

@tool(EmptyArgs, cls="atomic")
def silica_orphans() -> list:
    """Lists orphan notes (notes with no incoming links) in the vault."""
    refs = DRIVER.orphans()
    return [{"name": r.name, "path": r.path} for r in refs]


@tool(EmptyArgs, cls="atomic")
def silica_unresolved() -> list:
    """Lists unresolved wikilinks in the vault (links pointing to non-existent notes)."""
    links = DRIVER.unresolved()
    return [{"target": l.target} for l in links]


# ---------------------------------------------------------------------------
# List files
# ---------------------------------------------------------------------------

class ListFilesArgs(BaseModel):
    folder: str = Field(default="", description="Optional folder path to filter results")

# ponytail: flat cap defends the context window (a 1000-note vault ≈ 20k tokens
# uncapped); no paging — narrowing by folder covers the real use cases.
_FILES_CAP = 200


@tool(ListFilesArgs, cls="atomic")
def silica_files(folder: str = "") -> dict:
    """Lists markdown files in the vault, optionally filtered by folder.

    Returns {"total": N, "files": [{name, path}, ...]}. The listing is capped
    at 200 entries: when "truncated" is true, narrow with folder= instead of
    re-calling. For a bare count ("how many notes?") use the returned "total"
    — or the '## Vault map' block already in context, without any call.
    """
    refs = DRIVER.list_files(folder)
    files = [{"name": r.name, "path": r.path} for r in refs]
    result: dict = {"total": len(files), "files": files[:_FILES_CAP]}
    if len(files) > _FILES_CAP:
        result["truncated"] = True
        result["hint"] = "Listing capped at 200 entries; pass folder= to narrow."
    return result


class ExistsArgs(BaseModel):
    path: str = Field(description="Relative path of the note in the vault")

@tool(ExistsArgs, cls="atomic")
def silica_exists(path: str) -> bool:
    """Verifies if a note exists in the vault (including the inbox) given its relative path."""
    try:
        DRIVER.read_note(path)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Deferred Op Store
# ---------------------------------------------------------------------------

@tool(EmptyArgs, cls="atomic")
def silica_deferred_list() -> list:
    """List all pending deferred op bundles (concepts rejected by the validator in previous runs).

    Returns summary rows — use silica_deferred_retry(content_hash) to attempt
    writing them, or silica_deferred_flush(content_hash) to discard them.
    """
    from silica.kernel.deferred import get_deferred_store
    return get_deferred_store().list_all()


class DeferredFlushArgs(BaseModel):
    content_hash: str = Field(description="Content hash of the deferred bundle to permanently discard")

@tool(DeferredFlushArgs, cls="atomic", collapse="eager")
def silica_deferred_flush(content_hash: str) -> dict:
    """Discard a deferred op bundle — marks those rejected ops as permanently skipped."""
    from silica.kernel.deferred import get_deferred_store
    removed = get_deferred_store().remove(content_hash)
    if removed:
        return {"removed": True, "content_hash": content_hash}
    return {"removed": False, "error": f"No deferred bundle found for {content_hash[:8]}…"}


@tool(EmptyArgs, cls="atomic")
def silica_inbox_ls() -> list:
    """Lists all files in the Inbox folder (inbox_dir)."""
    refs = DRIVER.list_inbox_files()
    return [{"name": r.name, "path": r.path} for r in refs]


# ---------------------------------------------------------------------------
# Graph path / explain
# ---------------------------------------------------------------------------

class GraphPathArgs(BaseModel):
    source: str = Field(description="Source note name or vault-relative path")
    target: str = Field(description="Target note name or vault-relative path")
    max_paths: int = Field(default=1, description="Maximum number of shortest paths to return")

@tool(GraphPathArgs, cls="atomic")
def silica_graph_path(source: str, target: str, max_paths: int = 1) -> dict:
    """Shortest connection(s) between two notes over the resolved wikilink graph.

    Returns path(s) as lists of note ids, or an error dict if no path exists.
    Uses the undirected view of the resolved (EXTRACTED) wikilink graph.
    """
    import networkx as nx
    from silica.kernel.graph_export import build_graph_data

    try:
        nodes, edges = build_graph_data(folder="")
    except Exception as exc:
        return {"error": f"Failed to build graph: {exc}"}

    real_ids: set[str] = {n["id"] for n in nodes if n.get("type") != "ghost"}

    G = nx.Graph()
    G.add_nodes_from(real_ids)
    for e in edges:
        if e.get("type") == "EXTRACTED" and e["from"] in real_ids and e["to"] in real_ids:
            G.add_edge(e["from"], e["to"])

    # Resolve source/target: accept path or name substring match
    def _resolve(query: str) -> str | None:
        if query in real_ids:
            return query
        q_lower = query.lower().removesuffix(".md")
        for nid in real_ids:
            stem = nid.rsplit("/", 1)[-1].removesuffix(".md").lower()
            if stem == q_lower:
                return nid
        return None

    src_id = _resolve(source)
    tgt_id = _resolve(target)

    if src_id is None:
        return {"error": f"Source note not found in graph: '{source}'"}
    if tgt_id is None:
        return {"error": f"Target note not found in graph: '{target}'"}
    if src_id == tgt_id:
        return {"paths": [[src_id]], "length": 0}

    try:
        if max_paths == 1:
            path = nx.shortest_path(G, src_id, tgt_id)
            return {"paths": [path], "length": len(path) - 1}
        else:
            import itertools
            gen = nx.all_shortest_paths(G, src_id, tgt_id)
            paths = list(itertools.islice(gen, max_paths))
            return {"paths": paths, "length": len(paths[0]) - 1 if paths else 0}
    except nx.NetworkXNoPath:
        return {"error": f"No path between '{source}' and '{target}'"}
    except nx.NodeNotFound as exc:
        return {"error": f"Node not found: {exc}"}


class GraphExplainArgs(BaseModel):
    note: str = Field(description="Note name or vault-relative path to explain")
    depth: int = Field(default=1, description="Neighbourhood depth (1=direct links only)")

@tool(GraphExplainArgs, cls="atomic")
def silica_graph_explain(note: str, depth: int = 1) -> dict:
    """Explain a note's structural position: cluster, degree rank, betweenness,
    out-links, backlinks, and any cross-cluster bridges it participates in.

    `betweenness` is the fraction of shortest paths running through the note — a
    bottleneck signal distinct from degree. A note with LOW degree but HIGH
    betweenness is a bridge whose removal fragments the vault: worth reinforcing
    even though it has few links.
    """
    from silica.kernel.graph_report import compute_report

    try:
        report = compute_report(analytics=True)  # on-demand: needs god_nodes/bridges
    except Exception as exc:
        return {"error": f"Failed to compute graph report: {exc}"}

    # Find the node in god_nodes or clusters
    q_lower = note.lower().removesuffix(".md")
    node_stat = None
    for n in report.god_nodes:
        if n.id.lower() == q_lower or n.id.rsplit("/", 1)[-1].removesuffix(".md").lower() == q_lower:
            node_stat = n
            break

    # Resolve via cluster members if not in god_nodes
    resolved_id: str | None = None
    if node_stat:
        resolved_id = node_stat.id
    else:
        for c in report.clusters:
            for m in c.members:
                if m.lower() == q_lower or m.rsplit("/", 1)[-1].removesuffix(".md").lower() == q_lower:
                    resolved_id = m
                    break
            if resolved_id:
                break

    if resolved_id is None:
        # last attempt: check orphans
        for o in report.orphans:
            if o.lower() == q_lower or o.rsplit("/", 1)[-1].removesuffix(".md").lower() == q_lower:
                resolved_id = o
                break

    if resolved_id is None:
        return {"error": f"Note '{note}' not found in the graph"}

    # Degree rank (rank among all nodes by degree)
    try:
        out_links = [r.path or r.name for r in DRIVER.links(resolved_id)]
        backlinks = [r.path or r.name for r in DRIVER.backlinks(resolved_id)]
    except Exception:
        out_links = []
        backlinks = []

    bridges_involving = [
        {"source": b.source, "target": b.target, "clusters": f"{b.source_cluster}↔{b.target_cluster}", "weight": b.weight}
        for b in report.bridges
        if b.source == resolved_id or b.target == resolved_id
    ]

    cluster_id = -1
    for c in report.clusters:
        if resolved_id in c.members:
            cluster_id = c.cluster_id
            break

    # Degree rank
    all_degrees = sorted(
        [(n.id, n.degree) for n in report.god_nodes],
        key=lambda x: -x[1],
    )
    degree_rank = next(
        (i + 1 for i, (nid, _) in enumerate(all_degrees) if nid == resolved_id),
        None,
    )

    degree = (node_stat.degree if node_stat else len(out_links) + len(backlinks))
    return {
        "note": resolved_id,
        "cluster": cluster_id,
        "degree": degree,
        "degree_rank": degree_rank,
        "betweenness": report.betweenness_map.get(resolved_id, 0.0),
        "out_links": out_links[:depth * 10],
        "backlinks": backlinks[:depth * 10],
        "bridges": bridges_involving,
    }


# ---------------------------------------------------------------------------
# Ledger steering — silica_ledger_next / silica_ledger_update
# ---------------------------------------------------------------------------

class LedgerNextArgs(BaseModel):
    run_id: str = Field(description="Run ID returned by silica_vault_report")

@tool(LedgerNextArgs, cls="atomic")
def silica_ledger_next(run_id: str) -> dict:
    """Return the next actionable task for a run: capability (tool name), validated
    payload, and reason. Returns {"done": true} when the plan is exhausted.

    The agent should call the named tool with the payload, then call
    silica_ledger_update to record the outcome.
    """
    import orjson
    from pathlib import Path
    from silica.kernel.progress import ProgressLedger

    try:
        progress = ProgressLedger.load(run_id)
    except FileNotFoundError:
        return {"error": f"Run '{run_id}' not found"}
    except Exception as exc:
        return {"error": f"Failed to load ledger: {exc}"}

    t = progress.next_pending()
    if t is None:
        return {"done": True}

    # Load payload from disk if available
    payload: dict = {}
    if t.input_ref:
        try:
            payload = orjson.loads(Path(t.input_ref).read_bytes())
        except Exception:
            pass

    return {
        "task_id": t.id,
        "capability": t.capability_name,
        "payload": payload,
        "reason": payload.get("_reason", ""),
        "needs_confirmation": payload.get("needs_confirmation", False),
        "attempts": t.attempts,
    }


class LedgerUpdateArgs(BaseModel):
    run_id: str = Field(description="Run ID")
    task_id: str = Field(description="Task ID returned by silica_ledger_next")
    status: str = Field(description="Outcome: done | failed | skipped | blocked")
    error: str = Field(default="", description="Error message if status is 'failed'")

@tool(LedgerUpdateArgs, cls="atomic")
def silica_ledger_update(run_id: str, task_id: str, status: str, error: str = "") -> dict:
    """Mark a task's outcome on the run's ProgressLedger and persist it.

    Returns {"ok": true, "digest": ...} so the agent has the updated state
    for the next iteration.
    """
    from silica.kernel.progress import ProgressLedger

    try:
        progress = ProgressLedger.load(run_id)
    except FileNotFoundError:
        return {"error": f"Run '{run_id}' not found"}
    except Exception as exc:
        return {"error": f"Failed to load ledger: {exc}"}

    try:
        if status == "done":
            progress.mark_done(task_id)
        elif status == "failed":
            progress.mark_failed(task_id, error=error)
        else:
            progress.set_status(task_id, status, error=error or None)  # type: ignore[arg-type]
        progress.save()
    except KeyError:
        return {"error": f"Task '{task_id}' not found in run '{run_id}'"}
    except Exception as exc:
        return {"error": f"Failed to update ledger: {exc}"}

    return {"ok": True, "digest": progress.digest()}

