"""Pipeline tools — the mechanical injector stages as system tools.

Recon → payload → sanitize → validate → bulk-write → lint, plus the
deferred-ops retry path. These are the per-stage building blocks the
InjectorFSM (and the LLM, ad hoc) drive; the full run lives in
silica.tools.runners.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from silica.driver import DRIVER
from silica.tools import tool
from silica.kernel.ops import OpType
from silica.kernel.ops_io import load_ops, dump_ops


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
    from silica.kernel.recon import is_title_match, rank_hits, collision_priority
    from silica.kernel.keyphrase import extract_keyphrases
    from silica.kernel.overlay import get_active_overlay
    from silica.config import CONFIG

    try:
        nc = DRIVER.read_note(inbox_file)
    except RuntimeError:
        return {"error": f"File not found: {inbox_file}"}

    cands = extract_keyphrases(nc.content, overlay=get_active_overlay(), lang=CONFIG.cooccurrence_lang)
    concepts = [c.phrase for c in cands]
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
    cleared_links: list[dict] = []
    validated_ops, rejected_ops = validate_operations(
        ops,
        payloads,
        target_dir,
        hub=hub,
        cleared_parents_out=cleared_parents,
        future_ref_whitelist=future_ref_whitelist,
        cleared_links_out=cleared_links,
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
        "cleared_links": cleared_links,
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
