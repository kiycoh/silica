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
import time
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
from silica.kernel.paths import to_vault_relative, silica_tmp_dir

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
    HUB_UPDATE = auto()    # Phase 3.5 — patch Hub note with MOC links
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

        # Iterative chunk processing state fields
        self._chunks: list[dict] = []
        self._current_chunk_idx: int = 0

        # Shadow ProgressLedger — mirrors FSM state on disk; FSM remains canonical
        from silica.planner.progress import ProgressLedger
        self.progress = ProgressLedger.new(
            mode="inject",
            inputs={"inbox_file": inbox_file, "target_dir": target_dir, "hub": hub or ""},
        )
        self.progress.add_task("recon",   task_id="recon")
        self.progress.add_task("payload", task_id="payload", depends_on=["recon"])

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
                    { "id": "hub_update",   "kind": "mechanical", "tool": "silica_hub_update" },
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
            InjectorState.HUB_UPDATE: self._handle_hub_update,
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
            InjectorState.HUB_UPDATE: InjectorState.ROLLBACK,
            InjectorState.LINT: InjectorState.ROLLBACK,
        }

    def _get_chunks_from_context_if_empty(self) -> None:
        """Helper to extract chunks from self.context['payload'] if self._chunks is empty."""
        if not self._chunks and "payload" in self.context:
            res = self.context["payload"]
            if "chunks" in res:
                self._chunks = res["chunks"]
            elif "payload" in res:
                self._chunks = [res["payload"]]
            else:
                self._chunks = [res]

    def _get_recipe_gate(self, name: str, default: Any) -> Any:
        return self._recipe.get("gates", {}).get(name, default)

    def _get_recipe_phase(self, phase_id: str) -> dict:
        for phase in self._recipe.get("phases", []):
            if phase.get("id") == phase_id:
                return phase
        return {}

    def _make_tmp(self, content: Any, suffix: str = ".json") -> str:
        """Write content as JSON to ~/.silica/tmp/ and track for cleanup."""
        import uuid
        tmp_dir = silica_tmp_dir()
        path = str(tmp_dir / f"{uuid.uuid4().hex}{suffix}")
        with open(path, "wb") as f:
            if isinstance(content, list) and len(content) > 0 and hasattr(content[0], "model_dump"):
                f.write(orjson.dumps([item.model_dump() for item in content], option=orjson.OPT_INDENT_2))
            elif hasattr(content, "model_dump"):
                f.write(orjson.dumps(content.model_dump(), option=orjson.OPT_INDENT_2))
            else:
                f.write(orjson.dumps(content, option=orjson.OPT_INDENT_2))
        self._tmp_files.append(path)
        logger.debug("Created staging file: %s", path)
        return path

    def _cleanup_tmp(self) -> None:
        for path in self._tmp_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._tmp_files.clear()

    def _progress_note(
        self,
        task_id: str,
        capability_name: str,
        status: str,
        *,
        output_ref: str | None = None,
        error: str | None = None,
    ) -> None:
        """Shadow: record FSM progress in ProgressLedger; never affects FSM control flow."""
        try:
            from silica.planner.progress import TaskStatus
            if not any(t.id == task_id for t in self.progress.tasks):
                self.progress.add_task(capability_name, task_id=task_id)
            if status == "done":
                self.progress.mark_done(task_id, output_ref=output_ref)
            elif status == "failed":
                self.progress.mark_failed(task_id, error or "")
            else:
                self.progress.set_status(task_id, status, error=error)  # type: ignore[arg-type]
            self.progress.save()
        except Exception as _e:
            logger.debug("progress shadow error (suppressed): %s", _e)

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
                    logger.debug("FSM Transition: %s -> executing handler", self.state.name)
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
            "hub_update": InjectorState.HUB_UPDATE,
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
                # After the sequence, go to cleanup if defined, else DONE/LOOP
                if "cleanup" in [p["id"] for p in phases]:
                    self.state = InjectorState.CLEANUP
                else:
                    self._eval_loop_or_done()
        elif self.state == InjectorState.CLEANUP:
            self._eval_loop_or_done()
        elif self.state == InjectorState.ROLLBACK:
            self.state = InjectorState.ERROR

    def _eval_loop_or_done(self) -> None:
        """Check if there are more chunks to process or if the queue is empty."""
        self._get_chunks_from_context_if_empty()
        if self._current_chunk_idx + 1 < len(self._chunks):
            self._current_chunk_idx += 1
            logger.info(f"✔ Batch completed successfully. Advancing to batch {self._current_chunk_idx + 1}")
            # Reset FSM state back to DELEGATE to process next chunk
            self.state = InjectorState.DELEGATE
        else:
            logger.info("🎉 All batched chunks have been successfully injected and verified!")
            self.state = InjectorState.DONE

    # ------------------------------------------------------------------
    # State Handlers
    # ------------------------------------------------------------------

    def _handle_recon(self) -> None:
        self._progress_note("recon", "recon", "running")
        res = silica_recon(self.inbox_file)
        if "error" in res:
            self._progress_note("recon", "recon", "failed", error=res["error"])
            raise RuntimeError(f"Recon failed: {res['error']}")
        self.context["recon"] = res
        self._progress_note("recon", "recon", "done")

        # Surface any deferred ops from a previous run of this source file.
        content_hash = self.context.get("source_content_hash", "")
        if content_hash:
            from silica.kernel.deferred import get_deferred_store
            bundle = get_deferred_store().get(content_hash)
            if bundle:
                rejected_count = len(bundle.get("rejected_ops", []))
                logger.info(
                    "RECON: %d deferred op(s) from a previous run of '%s' are waiting. "
                    "Call silica_deferred_retry('%s') to attempt them.",
                    rejected_count,
                    self.inbox_file,
                    content_hash[:8],
                )
                self.context["deferred"] = {
                    "content_hash": content_hash,
                    "rejected_count": rejected_count,
                }

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
        self._progress_note("payload", "payload", "running")
        recon_path = self._make_tmp([self.context["recon"]])
        phase_conf = self._get_recipe_phase("payload")
        max_concepts = phase_conf.get("partition_if_over", 200)
        res = silica_payload(recon_path, max_concepts=max_concepts)
        if "error" in res:
            self._progress_note("payload", "payload", "failed", error=res["error"])
            raise RuntimeError(f"Payload failed: {res['error']}")
        self.context["payload"] = res

        # Deterministically isolate chunks at payload source
        if "chunks" in res:
            self._chunks = res["chunks"]
        elif "payload" in res:
            self._chunks = [res["payload"]]
        else:
            self._chunks = [res]

        self._current_chunk_idx = 0

        # Register per-chunk tasks now that chunk count is known
        prev = "payload"
        for idx in range(len(self._chunks)):
            for cap in ("distill", "sanitize", "validate", "snapshot", "write", "hub_update", "lint", "cleanup"):
                tid = f"chunk_{idx}_{cap}"
                dep = prev
                self.progress.add_task(cap, task_id=tid, depends_on=[dep])
                prev = tid
        try:
            self.progress.save()
        except Exception as _e:
            logger.debug("progress save error (suppressed): %s", _e)

        self._progress_note("payload", "payload", "done")
        logger.info(f"Pipeline initialized with {len(self._chunks)} independent chunks.")
        self._transition_success()

    def _handle_delegate(self) -> None:
        from silica.kernel.prep_delegation import run_distiller

        self._get_chunks_from_context_if_empty()

        if not self._chunks or self._current_chunk_idx >= len(self._chunks):
            raise RuntimeError("No chunks available for iterative processing.")

        current_chunk = self._chunks[self._current_chunk_idx]
        idx = self._current_chunk_idx
        logger.info(f"--- DISTILLING BATCH {idx + 1}/{len(self._chunks)} ---")
        self._progress_note(f"chunk_{idx}_distill", "distill", "running")

        try:
            chunk_result = run_distiller(
                payload=current_chunk,
                target=self.target_dir,
                hub=self.hub,
            )
            if "error" in chunk_result:
                self._progress_note(f"chunk_{idx}_distill", "distill", "failed", error=chunk_result["error"])
                raise RuntimeError(f"Distiller error on batch {idx}: {chunk_result['error']}")

            distiller_path = self._make_tmp(chunk_result)
            self.context["distiller_output_path"] = distiller_path
            self._progress_note(f"chunk_{idx}_distill", "distill", "done", output_ref=distiller_path)
            self._transition_success()

        except Exception as e:
            raise RuntimeError(f"Critical failure delegating batch {idx}: {e}")

    def _handle_sanitize(self) -> None:
        idx = self._current_chunk_idx
        self._progress_note(f"chunk_{idx}_sanitize", "sanitize", "running")
        res = silica_sanitize(self.context["distiller_output_path"])
        if "error" in res:
            self._progress_note(f"chunk_{idx}_sanitize", "sanitize", "failed", error=res["error"])
            raise RuntimeError(f"Sanitize failed: {res['error']}")
        self.context["sanitized"] = res
        self._progress_note(f"chunk_{idx}_sanitize", "sanitize", "done")
        self._transition_success()

    def _handle_validate(self) -> None:
        idx = self._current_chunk_idx
        self._progress_note(f"chunk_{idx}_validate", "validate", "running")
        sanitized = self.context["sanitized"]["parsed"]
        ops_raw = sanitized.get("updates", sanitized) if isinstance(sanitized, dict) else sanitized
        if not isinstance(ops_raw, list):
            ops_raw = [ops_raw]

        ops_path = self._make_tmp(ops_raw)

        self._get_chunks_from_context_if_empty()

        payload_paths: list[str] = []
        if self._chunks and self._current_chunk_idx < len(self._chunks):
            payload_paths.append(self._make_tmp(self._chunks[self._current_chunk_idx]))
        else:
            # Fallback to general payload if _chunks is not populated
            payload_data = self.context.get("payload", {})
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
            self._progress_note(f"chunk_{idx}_validate", "validate", "done")
            self.state = InjectorState.CLEANUP
            return

        # Persist rejected ops to the deferred store so the model can retry them
        # later without re-running the expensive RECON → DELEGATE cycle.
        rejected_raw = res.get("rejected_ops", [])
        if rejected_raw:
            content_hash = self.context.get("source_content_hash", "")
            if content_hash:
                from silica.kernel.deferred import get_deferred_store
                deferred_ops = [
                    r.get("op", r) if isinstance(r, dict) and "op" in r else r
                    for r in rejected_raw
                ]
                rejection_reasons = {
                    (r.get("op", {}).get("path") or r.get("op", {}).get("heading") or "?"): r.get("reason", "")
                    for r in rejected_raw if isinstance(r, dict)
                }
                get_deferred_store().put(
                    content_hash=content_hash,
                    source_path=self.inbox_file,
                    target_dir=self.target_dir,
                    hub=self.hub,
                    rejected_ops=deferred_ops,
                    rejection_reasons=rejection_reasons,
                )
                logger.warning(
                    "VALIDATE: %d op(s) rejected and saved to deferred store (hash=%s…). "
                    "Use silica_deferred_retry to attempt them later.",
                    len(rejected_raw),
                    content_hash[:8],
                )

        rejection_rate = res.get("rejection_rate", 0)
        if rejection_rate >= max_rate:
            logger.warning(
                "VALIDATE: rejection rate %.1f%% exceeds threshold %.1f%% — continuing with %d validated op(s).",
                rejection_rate * 100,
                max_rate * 100,
                res.get("validated_count", 0),
            )

        # Abort only when no validated ops remain — partial success is fine.
        if res.get("validated_count", 0) == 0:
            self.context["abort_reason"] = "All ops rejected — nothing to write"
            self.context["final_status"] = "no_ops"
            self.context["ops_path"] = ops_path
            self._progress_note(f"chunk_{idx}_validate", "validate", "done")
            self.state = InjectorState.CLEANUP
            return

        self.context["ops_path"] = ops_path
        self._progress_note(f"chunk_{self._current_chunk_idx}_validate", "validate", "done")
        self._transition_success()

    def _handle_snapshot(self) -> None:
        idx = self._current_chunk_idx
        self._progress_note(f"chunk_{idx}_snapshot", "snapshot", "running")
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
            from silica.kernel.wikilink import extract_links as _extract_links
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

                    if op.op in (OpType.patch, OpType.overwrite, OpType.delete):
                        # Capture current outgoing targets so we can detect orphaning.
                        try:
                            for target_ref in DRIVER.links(ref):
                                snapshot_domain.add(target_ref)
                        except Exception as ex:
                            logger.warning("Failed to fetch pre-write links for %s: %s", path, ex)

                    elif op.op == OpType.write:
                        # A write op creates a new note that didn't exist at pre-snapshot
                        # time.  After the write, graph_snapshot expands its neighborhood
                        # to include every vault note the new note links to.  If those
                        # linked notes carry pre-existing unresolved links, they appear as
                        # new_unres in graph_diff Rule 2 — a false positive.
                        # Fix: add those link targets to the pre-snapshot domain now so
                        # their existing ghost links cancel out in the diff.
                        content = op.snippet or op.content or ""
                        for link_target in _extract_links(content):
                            target_stem = link_target.removesuffix(".md")
                            target_key = target_stem.lower()
                            try:
                                if "/" in target_stem:
                                    target_name = os.path.splitext(os.path.basename(target_stem))[0]
                                    snapshot_domain.add(NoteRef(name=target_name, path=target_stem + ".md"))
                                else:
                                    for match in DRIVER.search_names(target_stem):
                                        if match.name.lower() == target_key:
                                            snapshot_domain.add(match)
                            except Exception as ex:
                                logger.debug("Snapshot domain expansion: could not resolve '%s': %s", link_target, ex)

            snapshot_domain_list = list(snapshot_domain)
            self.context["snapshot_domain"] = [{"name": r.name, "path": r.path} for r in snapshot_domain_list]
            self._pre_graph = DRIVER.graph_snapshot(snapshot_domain_list)
        except Exception as e:
            logger.error("Failed to take pre-write graph snapshot: %s", e)
            raise RuntimeError(f"Pre-write graph snapshot failed: {e}")

        self._progress_note(f"chunk_{idx}_snapshot", "snapshot", "done")
        self._transition_success()

    def _handle_write(self) -> None:
        idx = self._current_chunk_idx
        self._progress_note(f"chunk_{idx}_write", "write", "running")
        res = silica_bulk_write(self.context["ops_path"])

        if "error" in res:
            self._progress_note(f"chunk_{idx}_write", "write", "failed", error=res["error"])
            raise RuntimeError(f"Write failed: {res['error']}")
        if not res.get("success", False):
            failed = res.get("failed_operations", "?")
            total = res.get("total_operations", "?")
            raise RuntimeError(
                f"Write partially failed: {failed}/{total} operations failed. "
                f"Results: {res.get('results', [])}"
            )

        self.context["write"] = res
        self._progress_note(f"chunk_{idx}_write", "write", "done")
        self._transition_success()

    def _handle_hub_update(self) -> None:
        """Append MOC links to the Hub note for all newly written notes."""
        idx = self._current_chunk_idx
        self._progress_note(f"chunk_{idx}_hub_update", "hub_update", "running")
        if not self.hub:
            logger.info("HUB_UPDATE: no hub configured, skipping")
            self._progress_note(f"chunk_{idx}_hub_update", "hub_update", "done")
            self._transition_success()
            return

        try:
            ops = load_ops(self.context["ops_path"])
        except Exception as e:
            raise RuntimeError(f"HUB_UPDATE: failed to read ops: {e}")

        hub_name = self.hub.strip("[]")
        hub_name_lower = hub_name.lower()

        # Collect write ops for new notes, excluding the hub auto-creation itself
        new_notes: list[tuple[str, str]] = []
        for op in ops:
            if op.op != OpType.write:
                continue
            path = op.touched_ref()
            if not path:
                continue
            note_name = os.path.splitext(os.path.basename(path))[0]
            if note_name.lower() == hub_name_lower:
                continue
            snippet = (op.snippet or "").strip()
            desc = snippet.split("\n")[0][:120] if snippet else ""
            new_notes.append((note_name, desc))

        if not new_notes:
            logger.info("HUB_UPDATE: no new notes to link, skipping")
            self._progress_note(f"chunk_{idx}_hub_update", "hub_update", "done")
            self._transition_success()
            return

        hub_path = f"{self.target_dir}/{hub_name}.md".replace("//", "/")
        from silica.driver.base import NoteRef
        hub_ref = NoteRef(name=hub_name, path=hub_path)

        try:
            hub_note = DRIVER.read_note(hub_ref)
        except Exception as e:
            logger.warning("HUB_UPDATE: hub '%s' not readable: %s — skipping", hub_path, e)
            self._progress_note(f"chunk_{idx}_hub_update", "hub_update", "done")
            self._transition_success()
            return

        # If hub pre-existed (not created in this txn), register a content-based
        # rollback inverse using the content we just read — more reliable than
        # history:restore whose version positions shift after each new write.
        hub_path_norm = hub_path.replace("\\", "/")
        hub_is_new = self._txn is not None and any(
            p.replace("\\", "/") == hub_path_norm
            for p in (self._txn.created_paths or [])
        )
        if not hub_is_new and self._txn is not None:
            from silica.kernel.ops import InverseOp, InverseOpKind
            hub_inverse = InverseOp(
                kind=InverseOpKind.restore_version,
                path=hub_path,
                prior_content=hub_note.content,
            )
            self._txn.inverses.append(hub_inverse)
            if "snapshot" in self.context and "inverses" in self.context["snapshot"]:
                self.context["snapshot"]["inverses"].append(hub_inverse.model_dump())

        # Build MOC block and merge with existing content.
        # Use overwrite (not append) to avoid the create→append settle race:
        # append's _wait_for_content_contains must find the full fragment
        # within 2 s, which fails when the note was just created in WRITE.
        # overwrite's settle check (first 120 chars) is satisfied more quickly.
        source_name = os.path.splitext(os.path.basename(self.inbox_file))[0]
        lines = [f"\n## From: {source_name}\n"]
        for note_name, desc in new_notes:
            if desc:
                lines.append(f"- [[{note_name}]] — {desc}")
            else:
                lines.append(f"- [[{note_name}]]")
        moc_block = "\n".join(lines) + "\n"
        new_hub_content = hub_note.content.rstrip() + "\n" + moc_block

        try:
            DRIVER.overwrite(hub_path, new_hub_content)
            # The overwrite settle check only verifies the first 120 chars, which
            # for a long pre-existing hub equals the unchanged prefix — it would
            # pass immediately before the MOC block is flushed.  Explicitly wait
            # until the unique section header is readable.
            unique_marker = f"## From: {source_name}"
            _deadline = time.monotonic() + 5.0
            while time.monotonic() < _deadline:
                try:
                    if unique_marker in DRIVER.read_note(hub_ref).content:
                        break
                except Exception:
                    pass
                time.sleep(0.15)
            else:
                logger.warning("HUB_UPDATE: MOC block settle timeout for hub '%s' — graph may lag", hub_path)
            logger.info("HUB_UPDATE: updated hub '%s' with %d links", hub_path, len(new_notes))
        except Exception as e:
            raise RuntimeError(f"HUB_UPDATE: failed to update hub '{hub_path}': {e}")

        # Extend snapshot_domain so LINT's graph regression check covers the hub's new links
        existing_paths = {d["path"] for d in self.context.get("snapshot_domain", [])}
        if hub_path not in existing_paths:
            self.context.setdefault("snapshot_domain", []).append({"name": hub_name, "path": hub_path})

        self._progress_note(f"chunk_{idx}_hub_update", "hub_update", "done")
        self._transition_success()

    def _handle_lint(self) -> None:
        idx = self._current_chunk_idx
        self._progress_note(f"chunk_{idx}_lint", "lint", "running")
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
                self._progress_note(f"chunk_{idx}_lint", "lint", "failed", error=self.context["abort_reason"])
                self.state = InjectorState.ROLLBACK
                return

        # S3.2: Run graph-diff check
        regression_rule = self._get_recipe_gate("graph_regression", "forbid_new_orphans")
        if regression_rule != "allow":
            if self._pre_graph is None:
                self.context["abort_reason"] = "Graph regression gate failed: pre-write snapshot is missing"
                self._progress_note(f"chunk_{idx}_lint", "lint", "failed", error=self.context["abort_reason"])
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
                        len(self._pre_graph.link_counts) if self._pre_graph and self._pre_graph.link_counts else 0,
                        len(post_graph.link_counts) if post_graph and post_graph.link_counts else 0,
                        regression_rule,
                        "PASSED" if success else f"FAILED: {errors}"
                    )

                if not success:
                    self.context["abort_reason"] = (
                        f"Graph regression gate failed: {'; '.join(errors)}"
                    )
                    self._progress_note(f"chunk_{idx}_lint", "lint", "failed", error=self.context["abort_reason"])
                    self.state = InjectorState.ROLLBACK
                    return
            except Exception as e:
                logger.error("Failed to perform graph-diff check: %s", e)
                self.context["abort_reason"] = f"Graph regression gate error during check: {e}"
                self._progress_note(f"chunk_{idx}_lint", "lint", "failed", error=self.context["abort_reason"])
                self.state = InjectorState.ROLLBACK
                return

        self._progress_note(f"chunk_{idx}_lint", "lint", "done")
        self._transition_success()

    def _handle_cleanup(self) -> None:
        from silica.tools.wrapped import silica_cleanup

        self._get_chunks_from_context_if_empty()
        idx = self._current_chunk_idx
        self._progress_note(f"chunk_{idx}_cleanup", "cleanup", "running")

        # Only archive the physical inbox file on the very last chunk
        if not self._chunks or self._current_chunk_idx + 1 >= len(self._chunks):
            res = silica_cleanup(self.inbox_file, "done")
            if "error" in res:
                self.context["cleanup_warning"] = res["error"]
        else:
            logger.info("Batch %d completed. Skipping file move cleanup until final batch.", self._current_chunk_idx + 1)

        self._write_ledger("committed")
        if self.context.get("final_status") != "no_ops":
            self.context["final_status"] = "Success"
        self._progress_note(f"chunk_{idx}_cleanup", "cleanup", "done")
        self._transition_success()

    def _handle_rollback(self) -> None:
        self._progress_note("rollback", "rollback", "running")
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
        self._progress_note("rollback", "rollback", "done")
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
