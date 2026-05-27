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
from silica.kernel.ops import OpType
from silica.kernel.ops_io import load_ops, dump_ops
from silica.tools import tool


def _same_note(ref_a, ref_b) -> bool:
    """Confronto path-safe tra due NoteRef — gestisce slash, case, .md suffix."""
    import os
    def norm(r):
        p = r.path or r.name
        return os.path.normcase(p.replace("\\", "/").removesuffix(".md").strip("/"))
    return norm(ref_a) == norm(ref_b)


class ReconArgs(BaseModel):
    inbox_file: str = Field(description="Percorso file inbox da analizzare")
    limit: int = Field(default=0, description="Limite estrazione concetti")

@tool(ReconArgs, cls="composed")
def silica_recon(inbox_file: str, limit: int = 0) -> dict[str, Any]:
    """Estrazione meccanica di concetti da un file inbox e ricerca collisioni nel vault."""
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
    recon_report_path: str = Field(description="Path al JSON report di recon")
    max_concepts: int = Field(default=7, description="Massimo concetti per batch")
    max_bytes: int = Field(default=80 * 1024, description="Massimo byte (dimensione JSON) per chunk")

@tool(PayloadArgs, cls="composed")
def silica_payload(recon_report_path: str, max_concepts: int = 7, max_bytes: int = 80 * 1024) -> dict[str, Any]:
    """Assembla i payload per il Distiller pre-estraendo estratti dal vault."""
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
    distiller_output_path: str = Field(description="Output raw del Distiller")

@tool(SanitizeArgs, cls="composed")
def silica_sanitize(distiller_output_path: str) -> dict[str, Any]:
    """Valida e pulisce il JSON restituito dai worker Distiller."""
    from silica.kernel.sanitize import parse_json
    
    try:
        with open(distiller_output_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
    except Exception as e:
        return {"error": f"Failed to read distiller output: {e}"}
        
    try:
        parsed_obj, was_clean = parse_json(raw_content, strict=False)
    except Exception as e:
        return {"error": f"JSON Parse Error: {e}"}
        
    return {
        "success": True,
        "parsed": parsed_obj,
        "was_clean": was_clean
    }


class ValidateOpsArgs(BaseModel):
    ops_json_path: str = Field(description="Operazioni consolidate da validare")
    payload_paths: list[str] = Field(default_factory=list, description="Percorsi ai file payload originali")
    target_dir: str = Field(default="", description="Target folder in the vault")
    hub: str = Field(default="", description="Hub note name")

@tool(ValidateOpsArgs, cls="composed")
def silica_validate_ops(ops_json_path: str, payload_paths: list[str] | None = None, target_dir: str = "", hub: str = "") -> dict[str, Any]:
    """Gate pre-scrittura: controlla validità strutturale e applica threshold rigetti (10%).

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

    validated_ops, rejected_ops = validate_operations(ops, payloads, target_dir, hub=hub)

    total = len(ops)
    rejected_count = len(rejected_ops)
    # C4 denominator: skip ops excluded from rejection rate
    actionable = sum(1 for o in ops if o.op != OpType.skip)
    rejection_rate = rejected_count / actionable if actionable > 0 else 0.0

    success = rejection_rate <= 0.1

    # C4: Overwrite ops_json_path with validated (coerced + deduped) ops.
    # SNAPSHOT and WRITE read this same file — single source of truth.
    if success:
        try:
            dump_ops(ops_json_path, validated_ops)
        except Exception as e:
            return {"error": f"Failed to persist validated ops: {e}"}

    return {
        "success": success,
        "total": total,
        "validated_count": len(validated_ops),
        "rejected_count": rejected_count,
        "rejection_rate": rejection_rate,
        "validated_ops": [o.model_dump() for o in validated_ops],
        "rejected_ops": [r.model_dump() for r in rejected_ops],
    }


class BulkWriteArgs(BaseModel):
    ops_json_path: str = Field(description="Percorso al file JSON delle operazioni validate")

@tool(BulkWriteArgs, cls="composed")
def silica_bulk_write(ops_json_path: str) -> dict[str, Any]:
    """Applica in batch write/patch/overwrite/delete nel vault."""
    from silica.kernel.bulk import execute_operations
    
    try:
        ops = load_ops(ops_json_path)
    except Exception as e:
        return {"error": f"Failed to load operations: {e}"}
        
    res = execute_operations(ops)
    return res.model_dump()


class LintArgs(BaseModel):
    note_name: str = Field(description="Nome della nota da lintare")
    op_type: str = Field(default="", description="Op type (write/patch/overwrite) for conditional checks")
    hub: str = Field(default="", description="Hub note name for wikilink validation")

@tool(LintArgs, cls="composed")
def silica_lint(note_name: str, op_type: str = "", hub: str = "") -> dict[str, Any]:
    """Gate post-scrittura: esegue l'OFM linter per trovare regressioni strutturali."""
    from silica.kernel.linter import validate_note

    errors, warnings = validate_note(note_name, hub=hub or None, op_type=op_type or None)

    return {
        "success": len(errors) == 0,
        "note": note_name,
        "errors": errors,
        "warnings": warnings,
    }


class RunInjectorArgs(BaseModel):
    inbox_file: str = Field(description="Percorso file inbox da ingerire (es. Inbox/meeting_notes.md)")
    target_dir: str = Field(description="Directory di destinazione per i concetti estratti")
    hub: str = Field(default="", description="Hub di riferimento opzionale")

@tool(RunInjectorArgs, cls="composed")
def silica_run_injector(inbox_file: str, target_dir: str, hub: str = "") -> dict[str, Any]:
    """Azione singola per l'agente: esegue l'intera pipeline Injector (10 fasi) in modo deterministico con gate di accettazione e rollback in caso di fallimento."""
    from silica.router.orchestrator import InjectorFSM
    
    fsm = InjectorFSM(inbox_file=inbox_file, target_dir=target_dir, hub=hub or None)
    return fsm.run()

