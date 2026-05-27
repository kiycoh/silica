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

import hashlib
import logging
import os
import tempfile
from enum import Enum, auto
from typing import Any, TYPE_CHECKING
import orjson

if TYPE_CHECKING:
    from silica.driver.base import Txn, GraphSnapshot

from silica.driver import DRIVER
from silica.config import CONFIG
from silica.tools.composed import (
    silica_bulk_write,
    silica_lint,
    silica_payload,
    silica_recon,
    silica_sanitize,
    silica_validate_ops,
)
from silica.kernel.ops import OpType
from silica.kernel.ops_io import load_ops
from silica.kernel.paths import to_vault_relative

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
        # Canonicalize to vault-relative POSIX once at ingress so the RECON
        # driver read, _source_canonical (ledger key), and every downstream
        # consumer operate on the same form. Absolute paths under the vault
        # are relativized; absolute paths outside the vault fail loudly here
        # instead of degrading into a silent "File not found" at RECON.
        self.inbox_file = to_vault_relative(inbox_file)
        self.target_dir = target_dir

        # Hub sanity check: if not specified, inherit the folder name of target_dir
        if not hub and target_dir:
            import os
            hub = os.path.basename(target_dir.rstrip("/\\"))
        self.hub = hub

        self.state = InjectorState.INIT
        self.context: dict[str, Any] = {}
        self._tmp_files: list[str] = []
        self._txn: Txn | None = None  # holds the live Txn object for ROLLBACK
        self._pre_graph: GraphSnapshot | None = None  # S3.2 pre-write graph snapshot

        # S3.3: Load the recipe for dynamic configuration
        from silica.router.recipe_parser import load_recipe
        try:
            self._recipe = load_recipe("injector")
        except Exception as e:
            logger.warning("Failed to load recipe 'injector', using defaults: %s", e)
            self._recipe = {}

        if not self._recipe or "phases" not in self._recipe:
            self._recipe = {
                "name": "injector",
                "gates": {
                    "rejection_rate_max": 0.10,
                    "graph_regression": "forbid_new_orphans"
                },
                "phases": [
                    { "id": "recon",        "kind": "mechanical", "tool": "silica_recon" },
                    { "id": "payload",      "kind": "mechanical", "tool": "silica_payload", "partition_if_over": 200 },
                    { "id": "distill",      "kind": "semantic",   "worker": "distiller", "fanout": True, "max_workers": 7 },
                    { "id": "sanitize",     "kind": "mechanical", "tool": "silica_sanitize" },
                    { "id": "validate",     "kind": "gate",       "tool": "silica_validate_ops", "abort_code": 2 },
                    { "id": "snapshot",     "kind": "txn",        "tool": "silica_snapshot" },
                    { "id": "write",        "kind": "mechanical", "tool": "silica_bulk_write" },
                    { "id": "lint",         "kind": "gate",       "tool": "silica_lint" },
                    { "id": "cleanup",      "kind": "mechanical", "tool": "silica_cleanup", "on_success_only": True },
                    { "id": "rollback",     "kind": "txn",        "tool": "silica_restore", "on_gate_fail": True }
                ]
            }

        # S2.2.1: Handlers mapping and error policy
        self._HANDLERS = {
            InjectorState.RECON: self._handle_recon,
            InjectorState.PAYLOAD: self._handle_payload,
            InjectorState.DELEGATE: self._handle_delegate,
            InjectorState.SANITIZE: self._handle_sanitize,
            InjectorState.VALIDATE: self._handle_validate,
            InjectorState.SNAPSHOT: self._handle_snapshot,
            InjectorState.WRITE: self._handle_write,
            InjectorState.LINT: self._handle_lint,
            InjectorState.CLEANUP: self._handle_cleanup,
            InjectorState.ROLLBACK: self._handle_rollback,
        }

        self._ON_ERROR = {
            InjectorState.RECON: InjectorState.ERROR,
            InjectorState.PAYLOAD: InjectorState.ERROR,
            InjectorState.DELEGATE: InjectorState.ERROR,
            InjectorState.SANITIZE: InjectorState.ERROR,
            InjectorState.VALIDATE: InjectorState.ERROR,
            InjectorState.SNAPSHOT: InjectorState.ERROR,
            InjectorState.WRITE: InjectorState.ROLLBACK,
            InjectorState.LINT: InjectorState.ROLLBACK,
        }

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
            with os.fdopen(fd, "wb") as f:
                if isinstance(content, list) and len(content) > 0 and hasattr(content[0], "model_dump"):
                    f.write(orjson.dumps([item.model_dump() for item in content], option=orjson.OPT_INDENT_2))
                elif hasattr(content, "model_dump"):
                    f.write(orjson.dumps(content.model_dump(), option=orjson.OPT_INDENT_2))
                else:
                    f.write(orjson.dumps(content, option=orjson.OPT_INDENT_2))
        except Exception:
            os.close(fd)
            raise
        self._tmp_files.append(path)
        logger.debug("Creato file temporaneo di stage in: %s", path)
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
        from silica.kernel.ledger import get_ledger
        # Compute stable identity: vault-relative path, no .md, lowercase
        source_canonical = self._source_canonical()
        # Hash the inbox file content for content-aware skip (C2.2)
        try:
            content_bytes = open(self.inbox_file, "rb").read()
            content_hash = hashlib.sha256(content_bytes).hexdigest()
        except OSError:
            content_hash = ""
        self.context["source_canonical"] = source_canonical
        self.context["source_content_hash"] = content_hash

        if get_ledger().is_committed(source_canonical, content_hash=content_hash):
            self.context["final_status"] = "already_ingested"
            return self.context

        self.state = InjectorState.RECON

        try:
            while self.state not in (InjectorState.DONE, InjectorState.ERROR):
                try:
                    logger.debug("FSM Transizione: %s -> eseguendo handler", self.state.name)
                    self.step()
                except Exception as e:
                    logger.error("FSM Error in state %s: %s", self.state, e)
                    self.context["error"] = str(e)
                    
                    next_state = self._ON_ERROR.get(self.state, InjectorState.ERROR)
                    if next_state == InjectorState.ROLLBACK and self._txn:
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
        handler = self._HANDLERS.get(self.state)
        if handler:
            handler()
        else:
            raise RuntimeError(f"No handler defined for state {self.state}")

    def _transition_success(self) -> None:
        """Advance to the next state according to the recipe phases sequence."""
        phases = self._recipe.get("phases", [])
        
        PHASE_TO_STATE = {
            "recon": InjectorState.RECON,
            "payload": InjectorState.PAYLOAD,
            "distill": InjectorState.DELEGATE,
            "sanitize": InjectorState.SANITIZE,
            "validate": InjectorState.VALIDATE,
            "snapshot": InjectorState.SNAPSHOT,
            "write": InjectorState.WRITE,
            "lint": InjectorState.LINT,
            "cleanup": InjectorState.CLEANUP,
            "rollback": InjectorState.ROLLBACK,
        }

        # Normal sequential flow excludes cleanup and rollback from direct transition
        sequence = [p["id"] for p in phases if not p.get("on_gate_fail") and p.get("id") != "rollback" and p.get("id") != "cleanup"]
        
        current_phase_id = None
        for k, v in PHASE_TO_STATE.items():
            if v == self.state:
                current_phase_id = k
                break
                
        if current_phase_id in sequence:
            idx = sequence.index(current_phase_id)
            if idx + 1 < len(sequence):
                next_phase_id = sequence[idx + 1]
                self.state = PHASE_TO_STATE[next_phase_id]
            else:
                # After the sequence, go to cleanup if defined, else DONE
                if "cleanup" in [p["id"] for p in phases]:
                    self.state = InjectorState.CLEANUP
                else:
                    self.state = InjectorState.DONE
        elif self.state == InjectorState.CLEANUP:
            self.state = InjectorState.DONE
        elif self.state == InjectorState.ROLLBACK:
            self.state = InjectorState.ERROR

    # ------------------------------------------------------------------
    # State Handlers
    # ------------------------------------------------------------------

    def _handle_recon(self) -> None:
        res = silica_recon(self.inbox_file)
        if "error" in res:
            raise RuntimeError(f"Recon failed: {res['error']}")
        self.context["recon"] = res
        self._transition_success()

    def _source_canonical(self) -> str:
        """Vault-relative canonical path for the inbox file (no .md, lowercase)."""
        vault_path = getattr(CONFIG, "vault_path", None) or ""
        inbox = self.inbox_file
        if vault_path:
            try:
                from pathlib import Path as _P
                rel = _P(inbox).relative_to(_P(vault_path)).as_posix()
                return rel.removesuffix(".md").lower()
            except ValueError:
                pass
        # Fallback: basename without extension
        return os.path.splitext(os.path.basename(inbox))[0].lower()

    def _handle_payload(self) -> None:
        recon_path = self._make_tmp([self.context["recon"]])
        phase_conf = self._get_recipe_phase("payload")
        max_concepts = phase_conf.get("partition_if_over", 200)
        res = silica_payload(recon_path, max_concepts=max_concepts)
        if "error" in res:
            raise RuntimeError(f"Payload failed: {res['error']}")
        self.context["payload"] = res
        self._transition_success()

    def _handle_delegate(self) -> None:
        from silica.agent.delegate import delegate
        from silica.kernel.prep_delegation import run_distiller

        payload_data = self.context["payload"]
        if "chunks" in payload_data:
            chunks = payload_data["chunks"]
        elif "payload" in payload_data:
            chunks = [payload_data["payload"]]
        else:
            chunks = [payload_data]

        def run_one(chunk: dict) -> dict:
            return run_distiller(
                payload=chunk,
                target=self.target_dir,
                hub=self.hub,
            )

        phase_conf = self._get_recipe_phase("distill")
        max_workers = phase_conf.get("max_workers", 7)

        results = delegate(chunks, run_one, max_workers=max_workers)

        merged_updates = []
        for idx, r in enumerate(results):
            if "error" in r:
                raise RuntimeError(f"Distiller chunk {idx} failed: {r['error']}")
            merged_updates.extend(r.get("updates", []))

        # Deduplicate by path (C4)
        path_groups: dict[str, list[dict]] = {}
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
                has_write = any(op.get("op") == "write" for op in group)
                for op in group:
                    if op is not richest:
                        op["op"] = "skip"
                        op["reason"] = f"Duplicate write/patch to the same path '{op.get('path')}' during multi-batch merge"
                if has_write:
                    richest["op"] = "write"

        merged_result = {"updates": merged_updates}
        distiller_path = self._make_tmp(merged_result)
        self.context["distiller_output_path"] = distiller_path
        self._transition_success()

    def _handle_sanitize(self) -> None:
        res = silica_sanitize(self.context["distiller_output_path"])
        if "error" in res:
            raise RuntimeError(f"Sanitize failed: {res['error']}")
        self.context["sanitized"] = res
        self._transition_success()

    def _handle_validate(self) -> None:
        sanitized = self.context["sanitized"]["parsed"]
        ops_raw = sanitized.get("updates", sanitized) if isinstance(sanitized, dict) else sanitized
        if not isinstance(ops_raw, list):
            ops_raw = [ops_raw]

        ops_path = self._make_tmp(ops_raw)

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
            hub=self.hub,
        )

        if "error" in res:
            raise RuntimeError(f"Validate failed: {res['error']}")

        self.context["validate"] = res

        max_rate = self._get_recipe_gate("rejection_rate_max", 0.10)

        if CONFIG.verbose:
            total_ops = res.get("validated_count", 0) + res.get("rejected_count", 0)
            logger.info(
                "[DEBUG VALIDATE Gate]: Success: %s | Total evaluated ops: %d | Validated (accepted): %d | Rejected: %d | Rejection Rate: %.1f%% (Max Allowed: %.1f%%)",
                res.get("success"),
                total_ops,
                res.get("validated_count", 0),
                res.get("rejected_count", 0),
                res.get("rejection_rate", 0) * 100,
                max_rate * 100,
            )

        if res.get("validated_count", 0) == 0 and res.get("rejected_count", 0) == 0:
            logger.info("VALIDATE: no actionable ops (all skip) — short-circuit to CLEANUP")
            self.context["final_status"] = "no_ops"
            self.context["ops_path"] = ops_path
            self.state = InjectorState.CLEANUP
            return

        if not res["success"] or res.get("rejection_rate", 0) >= max_rate:
            self.context["abort_reason"] = (
                f"Rejection rate {res.get('rejection_rate', 0):.1%} >= {max_rate:.1%}"
            )
            self.state = InjectorState.ERROR
        else:
            self.context["ops_path"] = ops_path
            self._transition_success()

    def _handle_snapshot(self) -> None:
        from silica.tools.wrapped import silica_snapshot
        res = silica_snapshot(self.context["ops_path"])
        if "error" in res:
            raise RuntimeError(f"SNAPSHOT failed: {res['error']}")
        
        self.context["snapshot"] = res
        self.context["txn_id"] = res["txn_id"]
        try:
            from silica.driver.base import NoteRef, Txn
            from silica.kernel.ops import InverseOp
            inv = [InverseOp(**d) for d in res["inverses"]]
            
            # Reconstruct refs for Txn from inverses
            refs = []
            for d in res["inverses"]:
                if d.get("kind") == "restore_version":
                    path = d.get("path")
                    name = path.rsplit("/", 1)[-1].removesuffix(".md")
                    refs.append(NoteRef(name=name, path=path))
                    
            self._txn = Txn(
                id=res["txn_id"],
                refs=refs,
                versions=res.get("versions", {}),
                created_paths=res.get("created_paths", []),
                inverses=inv
            )
        except Exception as e:
            raise RuntimeError(f"SNAPSHOT rebuild failed: {e}")

        # S3.2: Take pre-write graph snapshot incrementally
        try:
            ops = load_ops(self.context["ops_path"])
            touched_refs = []
            snapshot_domain = set()
            
            for op in ops:
                path = op.touched_ref()
                if path:
                    name = os.path.splitext(os.path.basename(path))[0]
                    ref = NoteRef(name=name, path=path)
                    touched_refs.append(ref)
                    snapshot_domain.add(ref)
                    
                    # For mutating ops on existing files (patch/overwrite/delete),
                    # capture their current outgoing targets to see if they become orphans
                    if op.op in (OpType.patch, OpType.overwrite, OpType.delete):
                        try:
                            for target_ref in DRIVER.links(ref):
                                snapshot_domain.add(target_ref)
                        except Exception as ex:
                            logger.warning("Failed to fetch pre-write links for %s: %s", path, ex)
                            
            snapshot_domain_list = list(snapshot_domain)
            self.context["snapshot_domain"] = [{"name": r.name, "path": r.path} for r in snapshot_domain_list]
            self._pre_graph = DRIVER.graph_snapshot(snapshot_domain_list)
        except Exception as e:
            logger.error("Failed to take pre-write graph snapshot: %s", e)
            raise RuntimeError(f"Pre-write graph snapshot failed: {e}")

        self._transition_success()

    def _handle_write(self) -> None:
        res = silica_bulk_write(self.context["ops_path"])

        if "error" in res:
            raise RuntimeError(f"Write failed: {res['error']}")
        if not res.get("success", False):
            failed = res.get("failed_operations", "?")
            total = res.get("total_operations", "?")
            raise RuntimeError(
                f"Write partially failed: {failed}/{total} operations failed. "
                f"Results: {res.get('results', [])}"
            )

        self.context["write"] = res
        self._transition_success()

    def _handle_lint(self) -> None:
        try:
            ops = load_ops(self.context["ops_path"])
        except Exception as e:
            raise RuntimeError(f"LINT: failed to read ops: {e}")

        touched = [
            (op.touched_ref(), op.op.value if op.op else "", op.hub or "")
            for op in ops
            if op.touched_ref() and op.op not in (OpType.delete, OpType.skip)
        ]

        for path, op_type, hub in touched:
            res = silica_lint(path, op_type=op_type or "", hub=hub or "")
            if CONFIG.verbose:
                logger.info(
                    "[DEBUG LINT Gate]: File: %s | Type: %s | Hub: %s | Success: %s | Errors: %s",
                    path,
                    op_type,
                    hub,
                    res["success"],
                    res.get("errors", []),
                )
            if not res["success"]:
                self.context["abort_reason"] = (
                    f"Lint failed for {path}: {res['errors']}"
                )
                self.state = InjectorState.ROLLBACK
                return

        # S3.2: Run graph-diff check
        regression_rule = self._get_recipe_gate("graph_regression", "forbid_new_orphans")
        if regression_rule != "allow":
            if self._pre_graph is None:
                self.context["abort_reason"] = "Graph regression gate failed: pre-write snapshot is missing"
                self.state = InjectorState.ROLLBACK
                return
            try:
                from silica.driver.base import NoteRef
                snapshot_domain_dicts = self.context.get("snapshot_domain", [])
                if snapshot_domain_dicts:
                    snapshot_domain = [NoteRef(**d) for d in snapshot_domain_dicts]
                else:
                    # Fallback to touched refs if snapshot_domain is missing
                    snapshot_domain = []
                    for op in ops:
                        path = op.touched_ref()
                        if path:
                            name = os.path.splitext(os.path.basename(path))[0]
                            snapshot_domain.append(NoteRef(name=name, path=path))
                
                post_graph = DRIVER.graph_snapshot(snapshot_domain)
                from silica.kernel.graph_diff import check_graph_regression
                
                created_paths = self._txn.created_paths if self._txn else []
                success, errors = check_graph_regression(self._pre_graph, post_graph, created_paths)

                if CONFIG.verbose:
                    logger.info(
                        "[DEBUG Graph Regression Gate]: Pre-write graph size: %d nodes | Post-write graph size: %d nodes | Rule: %s | Result: %s",
                        len(self._pre_graph.notes) if self._pre_graph and self._pre_graph.notes else 0,
                        len(post_graph.notes) if post_graph and post_graph.notes else 0,
                        regression_rule,
                        "PASSED" if success else f"FAILED: {errors}"
                    )

                if not success:
                    self.context["abort_reason"] = (
                        f"Graph regression gate failed: {'; '.join(errors)}"
                    )
                    self.state = InjectorState.ROLLBACK
                    return
            except Exception as e:
                logger.error("Failed to perform graph-diff check: %s", e)
                self.context["abort_reason"] = f"Graph regression gate error during check: {e}"
                self.state = InjectorState.ROLLBACK
                return

        self._transition_success()

    def _handle_cleanup(self) -> None:
        from silica.tools.wrapped import silica_cleanup
        res = silica_cleanup(self.inbox_file, "done")
        if "error" in res:
            self.context["cleanup_warning"] = res["error"]

        self._write_ledger("committed")
        if self.context.get("final_status") != "no_ops":
            self.context["final_status"] = "Success"
        self._transition_success()

    def _handle_rollback(self) -> None:
        snapshot_res = self.context.get("snapshot", {})
        inverses = snapshot_res.get("inverses", [])
        txn_id = snapshot_res.get("txn_id")
        
        if txn_id and inverses:
            from silica.tools.wrapped import silica_restore
            try:
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
            self._write_ledger_rollback(txn_id)

        self.context["final_status"] = (
            f"Rolled Back: {self.context.get('abort_reason', 'unknown reason')}"
        )
        self._transition_success()

    # ------------------------------------------------------------------
    # Ledger helpers (C5)
    # ------------------------------------------------------------------

    def _write_ledger(self, status: str) -> None:
        """Record all ops from ops_path into the ledger."""
        try:
            from silica.kernel.ledger import get_ledger
            ledger = get_ledger()
            txn_id = self.context.get("txn_id", "unknown")
            source_canonical = self.context.get("source_canonical", "")
            content_hash = self.context.get("source_content_hash")

            ops = load_ops(self.context["ops_path"])

            for op in ops:
                if op.op == OpType.skip:
                    continue
                ledger.record(
                    txn_id=txn_id,
                    source_canonical=source_canonical,
                    path=op.touched_ref(),
                    op=op.op.value if op.op else "",
                    status=status,
                    content_hash=content_hash,
                )
        except Exception as e:
            logger.warning("Failed to write ledger: %s", e)

    def _write_ledger_rollback(self, txn_id: str) -> None:
        try:
            from silica.kernel.ledger import get_ledger
            get_ledger().mark_rolled_back(txn_id)
        except Exception as e:
            logger.warning("Failed to mark rollback in ledger: %s", e)
