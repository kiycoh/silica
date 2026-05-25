"""L3 Router / Orchestrator for Silica — Injector FSM (S2.3 complete).

From SILICA.md §3 L3 & §7.3:
  Deterministic state machine for the Injector pipeline.
  Gates: >= 10% rejection rate -> abort + rollback.

Contracts applied (see silica_architecture_addendum.md):
  C1 — ops_path carries list[Op]-compatible dicts after VALIDATE.
  C2 — freshness via per-op postconditions in CLI backend.
  C3 — build_txn() builds InverseOp entries; ROLLBACK applies them.
  C4 — VALIDATE overwrites ops_path; SNAPSHOT/WRITE read that same file.
  C5 — ledger records ops; CLEANUP only reachable from DONE state.

S2.3 change: DELEGATE calls the real Distiller LLM via prep_delegation.
S2.3 change: SNAPSHOT uses build_txn() directly (no _txn_obj leak).
S2.3 change: ledger.py integrated (CLEANUP writes 'committed', ROLLBACK marks 'rolled_back').
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from enum import Enum, auto
from typing import Any

from silica.driver import DRIVER
from silica.tools.composed import (
    silica_bulk_write,
    silica_lint,
    silica_payload,
    silica_recon,
    silica_sanitize,
    silica_validate_ops,
)
from silica.tools.wrapped import silica_move, build_txn

logger = logging.getLogger(__name__)


class InjectorState(Enum):
    INIT = auto()
    RECON = auto()         # Phase 1
    PAYLOAD = auto()       # Phase 2.0
    DELEGATE = auto()      # Phase 2.1 — real Distiller LLM
    SANITIZE = auto()      # Phase 2.2
    VALIDATE = auto()      # Phase 2.3 (Gate) — C4: overwrites ops_path
    SNAPSHOT = auto()      # Phase 2.5 — C3: builds InverseOp Txn
    WRITE = auto()         # Phase 3
    LINT = auto()          # Phase 4 (Gate)
    CLEANUP = auto()       # Phase 5 — C5: only from DONE
    ROLLBACK = auto()      # On gate fail — C3: apply inverses
    DONE = auto()
    ERROR = auto()


class InjectorFSM:
    """Deterministic state machine for the Injector pipeline (S2.3 complete)."""

    def __init__(self, inbox_file: str, target_dir: str, hub: str | None = None):
        self.inbox_file = inbox_file
        self.target_dir = target_dir
        self.hub = hub

        self.state = InjectorState.INIT
        self.context: dict[str, Any] = {}
        self._tmp_files: list[str] = []
        self._txn = None  # holds the live Txn object for ROLLBACK
        self._pre_graph = None  # S3.2 pre-write graph snapshot

        # S3.3: Load the recipe for dynamic configuration
        from silica.router.recipe_parser import load_recipe
        try:
            self._recipe = load_recipe("injector")
        except Exception as e:
            logger.warning("Failed to load recipe 'injector', using defaults: %s", e)
            self._recipe = {}

    def _get_recipe_gate(self, name: str, default: Any) -> Any:
        return self._recipe.get("gates", {}).get(name, default)

    def _get_recipe_phase(self, phase_id: str) -> dict:
        for phase in self._recipe.get("phases", []):
            if phase.get("id") == phase_id:
                return phase
        return {}



    def _make_tmp(self, content: Any, suffix: str = ".json") -> str:
        """Write content as JSON to a temp file and track for cleanup."""
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(content, f, ensure_ascii=False)
        except Exception:
            os.close(fd)
            raise
        self._tmp_files.append(path)
        return path

    def _cleanup_tmp(self) -> None:
        for path in self._tmp_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._tmp_files.clear()

    def run(self) -> dict[str, Any]:
        """Execute the pipeline end-to-end."""
        self.state = InjectorState.RECON

        try:
            while self.state not in (InjectorState.DONE, InjectorState.ERROR):
                try:
                    self.step()
                except Exception as e:
                    logger.error("FSM Error in state %s: %s", self.state, e)
                    self.context["error"] = str(e)
                    if self.state in (InjectorState.WRITE, InjectorState.LINT) and self._txn:
                        self.context["abort_reason"] = str(e)
                        self.state = InjectorState.ROLLBACK
                    else:
                        self.state = InjectorState.ERROR
        finally:
            self._cleanup_tmp()

        return self.context

    def step(self) -> None:
        """Execute the current state and transition."""
        logger.info("Injector phase: %s", self.state.name)

        # ------------------------------------------------------------------
        if self.state == InjectorState.RECON:
            res = silica_recon(self.inbox_file)
            if "error" in res:
                raise RuntimeError(f"Recon failed: {res['error']}")
            self.context["recon"] = res
            self.state = InjectorState.PAYLOAD

        # ------------------------------------------------------------------
        elif self.state == InjectorState.PAYLOAD:
            recon_path = self._make_tmp([self.context["recon"]])
            phase_conf = self._get_recipe_phase("payload")
            max_concepts = phase_conf.get("partition_if_over", 7)
            res = silica_payload(recon_path, max_concepts=max_concepts)
            if "error" in res:
                raise RuntimeError(f"Payload failed: {res['error']}")
            self.context["payload"] = res
            self.state = InjectorState.DELEGATE

        # ------------------------------------------------------------------
        elif self.state == InjectorState.DELEGATE:
            # S3.1: call parallel Distiller LLM workers via delegate
            from silica.agent.delegate import delegate
            from silica.kernel.prep_delegation import run_distiller

            payload_data = self.context["payload"]

            if "chunks" in payload_data:
                chunks = payload_data["chunks"]
            else:
                p = payload_data.get("payload", payload_data)
                chunks = [p]

            # run_one closure for delegate
            def run_one(chunk: dict) -> dict:
                return run_distiller(
                    payload=chunk,
                    target=self.target_dir,
                    hub=self.hub,
                )

            phase_conf = self._get_recipe_phase("distill")
            max_workers = phase_conf.get("max_workers", 7)
            # Parallel fan-out
            results = delegate(chunks, run_one, max_workers=max_workers)

            merged_updates = []
            for idx, r in enumerate(results):
                if "error" in r:
                    raise RuntimeError(f"Distiller chunk {idx} failed: {r['error']}")
                merged_updates.extend(r.get("updates", []))

            # Deduplicate by path (C4)
            path_groups = {}
            for op in merged_updates:
                path = op.get("path")
                if path:
                    norm = os.path.abspath(path)
                    if norm not in path_groups:
                        path_groups[norm] = []
                    path_groups[norm].append(op)

            for norm, group in path_groups.items():
                if len(group) > 1:
                    richest = max(group, key=lambda o: len(o.get("snippet", "")))
                    for op in group:
                        if op is not richest:
                            op["op"] = "skip"
                            op["reason"] = f"Duplicate write/patch to the same path '{op.get('path')}' during multi-batch merge"

            merged_result = {"updates": merged_updates}
            distiller_path = self._make_tmp(merged_result)
            self.context["distiller_output_path"] = distiller_path
            self.state = InjectorState.SANITIZE

        # ------------------------------------------------------------------
        elif self.state == InjectorState.SANITIZE:
            res = silica_sanitize(self.context["distiller_output_path"])
            if "error" in res:
                raise RuntimeError(f"Sanitize failed: {res['error']}")
            self.context["sanitized"] = res
            self.state = InjectorState.VALIDATE

        # ------------------------------------------------------------------
        elif self.state == InjectorState.VALIDATE:
            # Write sanitized ops to temp file; validate_ops will OVERWRITE it (C4)
            sanitized = self.context["sanitized"]["parsed"]
            ops_raw = sanitized.get("updates", sanitized) if isinstance(sanitized, dict) else sanitized
            if not isinstance(ops_raw, list):
                ops_raw = [ops_raw]

            ops_path = self._make_tmp(ops_raw)

            # Collect payload paths for concept cross-check (C4: include all chunks)
            payload_paths: list[str] = []
            payload_data = self.context["payload"]
            if "chunks" in payload_data:
                for chunk in payload_data["chunks"]:
                    payload_paths.append(self._make_tmp(chunk))
            elif "payload" in payload_data:
                payload_paths.append(self._make_tmp(payload_data["payload"]))

            res = silica_validate_ops(
                ops_path,
                payload_paths=payload_paths,
                target_dir=self.target_dir,
            )

            if "error" in res:
                raise RuntimeError(f"Validate failed: {res['error']}")

            self.context["validate"] = res
            max_rate = self._get_recipe_gate("rejection_rate_max", 0.10)
            if not res["success"] or res.get("rejection_rate", 0) >= max_rate:
                self.context["abort_reason"] = (
                    f"Rejection rate {res.get('rejection_rate', 0):.1%} >= {max_rate:.1%}"
                )
                self.state = InjectorState.ERROR
            else:
                # C4: silica_validate_ops has already overwritten ops_path with
                # the coerced + deduped ops. SNAPSHOT and WRITE read this file.
                self.context["ops_path"] = ops_path
                self.state = InjectorState.SNAPSHOT

        # ------------------------------------------------------------------
        elif self.state == InjectorState.SNAPSHOT:
            from silica.tools.wrapped import silica_snapshot
            res = silica_snapshot(self.context["ops_path"])
            if "error" in res:
                raise RuntimeError(f"SNAPSHOT failed: {res['error']}")
            
            self.context["snapshot"] = res
            self.context["txn_id"] = res["txn_id"]

            # Retain live Txn for internal FSM needs (ledger metadata, etc.)
            try:
                with open(self.context["ops_path"], "r", encoding="utf-8") as f:
                    ops_data = json.load(f)
                ops = ops_data if isinstance(ops_data, list) else ops_data.get("updates", [])
                self._txn = build_txn(ops)
            except Exception as e:
                raise RuntimeError(f"SNAPSHOT rebuild failed: {e}")

            # S3.2: Take pre-write graph snapshot
            try:
                self._pre_graph = DRIVER.graph_snapshot()
            except Exception as e:
                logger.warning("Failed to take pre-write graph snapshot: %s", e)
                self._pre_graph = None

            self.state = InjectorState.WRITE

        # ------------------------------------------------------------------
        elif self.state == InjectorState.WRITE:
            res = silica_bulk_write(self.context["ops_path"])

            if "error" in res:
                raise RuntimeError(f"Write failed: {res['error']}")
            # C4: partial failure = full abort (B4 fix)
            if not res.get("success", False):
                failed = res.get("failed_operations", "?")
                total = res.get("total_operations", "?")
                raise RuntimeError(
                    f"Write partially failed: {failed}/{total} operations failed. "
                    f"Results: {res.get('results', [])}"
                )

            self.context["write"] = res
            self.state = InjectorState.LINT

        # ------------------------------------------------------------------
        elif self.state == InjectorState.LINT:
            # B1: read ops from the validated file, derive paths via op["path"]
            try:
                with open(self.context["ops_path"], "r", encoding="utf-8") as f:
                    ops_raw = json.load(f)
                ops = ops_raw if isinstance(ops_raw, list) else ops_raw.get("updates", [])
            except Exception as e:
                raise RuntimeError(f"LINT: failed to read ops: {e}")

            # Build (path, op_type, hub) tuples — exclude delete and skip
            touched = [
                (op["path"], op.get("op"), op.get("hub"))
                for op in ops
                if op.get("path") and op.get("op") not in ("delete", "skip")
            ]

            for path, op_type, hub in touched:
                res = silica_lint(path, op_type=op_type or "", hub=hub or "")
                if not res["success"]:
                    self.context["abort_reason"] = (
                        f"Lint failed for {path}: {res['errors']}"
                    )
                    self.state = InjectorState.ROLLBACK
                    return

            # S3.2: Run graph-diff check
            regression_rule = self._get_recipe_gate("graph_regression", "forbid_new_orphans")
            if regression_rule != "allow" and self._pre_graph is not None:
                try:
                    post_graph = DRIVER.graph_snapshot()
                    from silica.kernel.graph_diff import check_graph_regression
                    
                    created_paths = self._txn.created_paths if self._txn else []
                    success, errors = check_graph_regression(self._pre_graph, post_graph, created_paths)
                    if not success:
                        self.context["abort_reason"] = (
                            f"Graph regression gate failed: {'; '.join(errors)}"
                        )
                        self.state = InjectorState.ROLLBACK
                        return
                except Exception as e:
                    logger.error("Failed to perform graph-diff check: %s", e)

            self.state = InjectorState.CLEANUP

        # ------------------------------------------------------------------
        elif self.state == InjectorState.CLEANUP:
            # C5: only reachable from LINT -> CLEANUP (i.e. all gates green)
            # Move inbox to done/ using composed tool
            from silica.tools.wrapped import silica_cleanup
            res = silica_cleanup(self.inbox_file, "done")
            if "error" in res:
                self.context["cleanup_warning"] = res["error"]

            # C5: write committed ledger entries
            self._write_ledger("committed")

            self.context["final_status"] = "Success"
            self.state = InjectorState.DONE

        # ------------------------------------------------------------------
        elif self.state == InjectorState.ROLLBACK:
            snapshot_res = self.context.get("snapshot", {})
            inverses = snapshot_res.get("inverses", [])
            txn_id = snapshot_res.get("txn_id")
            
            if txn_id and inverses:
                from silica.tools.wrapped import silica_restore
                try:
                    # S3.3: Use silica_restore with inverses list as the single source of truth
                    res = silica_restore(txn_id=txn_id, inverses=inverses)
                    if not res.get("success", False):
                        err_msg = "; ".join(res.get("errors", []))
                        logger.error("Rollback partially failed: %s", err_msg)
                        self.context["rollback_error"] = err_msg
                    else:
                        logger.info("Rollback complete for txn %s", txn_id)
                except Exception as e:
                    logger.error("Rollback failed: %s", e)
                    self.context["rollback_error"] = str(e)
                # C5: mark ops as rolled_back in ledger
                self._write_ledger_rollback(txn_id)

            self.context["final_status"] = (
                f"Rolled Back: {self.context.get('abort_reason', 'unknown reason')}"
            )
            self.state = InjectorState.ERROR

    # ------------------------------------------------------------------
    # Ledger helpers (C5)
    # ------------------------------------------------------------------

    def _write_ledger(self, status: str) -> None:
        """Record all ops from ops_path into the ledger."""
        try:
            from silica.kernel.ledger import get_ledger
            ledger = get_ledger()
            txn_id = self.context.get("txn_id", "unknown")

            with open(self.context["ops_path"], "r", encoding="utf-8") as f:
                ops_raw = json.load(f)
            ops = ops_raw if isinstance(ops_raw, list) else ops_raw.get("updates", [])

            for op in ops:
                if op.get("op") == "skip":
                    continue
                ledger.record(
                    txn_id=txn_id,
                    source_basename=op.get("source_basename", ""),
                    path=op.get("path"),
                    op=op.get("op", ""),
                    status=status,
                )
        except Exception as e:
            logger.warning("Failed to write ledger: %s", e)

    def _write_ledger_rollback(self, txn_id: str) -> None:
        try:
            from silica.kernel.ledger import get_ledger
            get_ledger().mark_rolled_back(txn_id)
        except Exception as e:
            logger.warning("Failed to mark rollback in ledger: %s", e)
