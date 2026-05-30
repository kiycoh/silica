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
from silica.router.base_fsm import BaseFSM

logger = logging.getLogger(__name__)


class InjectorState(Enum):
    INIT = auto()
    RECON = auto()         # Phase 1
    PAYLOAD = auto()       # Phase 2.0
    SALIENCE = auto()      # Phase 2.05 — thematic salience gate (drop off-theme concepts)
    COLLISION = auto()     # Phase 5 — dedup routing: high-sim→patch, borderline→defer, low→write
    DELEGATE = auto()      # Phase 2.1 — real Distiller LLM
    SANITIZE = auto()      # Phase 2.2
    VALIDATE = auto()      # Phase 2.3 (Gate) — C4: overwrites ops_path
    SNAPSHOT = auto()      # Phase 2.5 — C3: builds InverseOp Txn
    WRITE = auto()         # Phase 3
    HUB_UPDATE = auto()    # Phase 3.5 — patch Hub note with MOC links
    AUTOLINK = auto()      # Phase 4 — inject wikilinks into touched notes
    LINT = auto()          # Phase 5 (Gate)
    CLEANUP = auto()       # Phase 5 — C5: only from DONE
    ROLLBACK = auto()      # On gate fail — C3: apply inverses
    DONE = auto()
    ERROR = auto()


class InjectorFSM(BaseFSM[InjectorState]):
    """Deterministic state machine for the Injector pipeline (S2.3 complete)."""

    def __init__(
        self,
        inbox_file: str = "",
        target_dir: str = "",
        hub: str | None = None,
        *,
        inbox_files: list[str] | None = None,
        resume_run_id: str | None = None,
    ):
        # Normalize to a list. inbox_files takes precedence; inbox_file is a
        # compat shim inserted at position 0 if not already present.
        files: list[str] = list(inbox_files or [])
        if inbox_file and inbox_file not in files:
            files.insert(0, inbox_file)
        if not files:
            raise ValueError("At least one inbox file must be provided")
        self.inbox_files: list[str] = [to_vault_relative(f) for f in files]
        self.inbox_file: str = self.inbox_files[0]  # first file; compat with single-file callers
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

        # Per-file content info — populated by run() before _run_loop starts
        self._file_canonicals: list[str] = []
        self._file_content_hashes: list[str] = []

        # Iterative chunk processing state fields
        self._chunks: list[dict] = []
        self._current_chunk_idx: int = 0
        # Multi-file hierarchy (§3.6): per-file chunk groups + flat→(fi,ci) mapping
        self._file_chunks: list[dict] = []  # [{"source_file": str, "chunks": [...]}, ...]
        self._chunk_flat_to_fi_ci: dict[int, tuple[int, int]] = {}  # flat_idx → (file_idx, chunk_idx)

        # Shadow ProgressLedger — mirrors FSM state on disk; FSM remains canonical
        from silica.planner.progress import ProgressLedger
        _resumed = False
        if resume_run_id:
            try:
                self.progress = ProgressLedger.load(resume_run_id)
                logger.info("Resuming run %s", resume_run_id)
                _resumed = True
            except Exception as _re:
                logger.warning("Failed to load run '%s', starting fresh: %s", resume_run_id, _re)
        if not _resumed:
            self.progress = ProgressLedger.new(
                mode="inject",
                inputs={
                    "inbox_files": self.inbox_files,
                    "inbox_file": self.inbox_file,
                    "target_dir": target_dir,
                    "hub": hub or "",
                },
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
                    { "id": "collision",    "kind": "mechanical", "best_effort": True },
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

        # BaseFSM contract
        self._phase_label = "Injector"
        self._done_state = InjectorState.DONE
        self._error_state = InjectorState.ERROR
        self._rollback_state = InjectorState.ROLLBACK
        self._phase_to_state: dict[str, InjectorState] = {
            "recon":      InjectorState.RECON,
            "payload":    InjectorState.PAYLOAD,
            "salience":   InjectorState.SALIENCE,
            "collision":  InjectorState.COLLISION,
            "distill":    InjectorState.DELEGATE,
            "sanitize":   InjectorState.SANITIZE,
            "validate":   InjectorState.VALIDATE,
            "snapshot":   InjectorState.SNAPSHOT,
            "write":      InjectorState.WRITE,
            "hub_update": InjectorState.HUB_UPDATE,
            "autolink":   InjectorState.AUTOLINK,
            "lint":       InjectorState.LINT,
            "cleanup":    InjectorState.CLEANUP,
            "rollback":   InjectorState.ROLLBACK,
        }

        # S2.2.1: Handlers mapping and error policy
        self._HANDLERS = {
            InjectorState.RECON: self._handle_recon,
            InjectorState.PAYLOAD: self._handle_payload,
            InjectorState.SALIENCE: self._handle_salience,
            InjectorState.COLLISION: self._handle_collision,
            InjectorState.DELEGATE: self._handle_delegate,
            InjectorState.SANITIZE: self._handle_sanitize,
            InjectorState.VALIDATE: self._handle_validate,
            InjectorState.SNAPSHOT: self._handle_snapshot,
            InjectorState.WRITE: self._handle_write,
            InjectorState.HUB_UPDATE: self._handle_hub_update,
            InjectorState.AUTOLINK: self._handle_autolink,
            InjectorState.LINT: self._handle_lint,
            InjectorState.CLEANUP: self._handle_cleanup,
            InjectorState.ROLLBACK: self._handle_rollback,
        }

        self._ON_ERROR = {
            # Setup phases: abort the whole run on failure
            InjectorState.RECON: InjectorState.ERROR,
            InjectorState.PAYLOAD: InjectorState.ERROR,
            # Per-chunk phases: contain failure at chunk level via rollback
            InjectorState.DELEGATE: InjectorState.ROLLBACK,
            InjectorState.SANITIZE: InjectorState.ROLLBACK,
            InjectorState.VALIDATE: InjectorState.ROLLBACK,
            InjectorState.SNAPSHOT: InjectorState.ROLLBACK,
            InjectorState.WRITE: InjectorState.ROLLBACK,
            InjectorState.HUB_UPDATE: InjectorState.ROLLBACK,
            InjectorState.LINT: InjectorState.ROLLBACK,
        }

        # Build and persist the immutable TaskLedger from the loaded recipe.
        # Shares run_id with ProgressLedger so both sides of the ledger are
        # co-located under ~/.silica/runs/<run_id>/.
        from silica.planner.progress import TaskLedger, CheckpointSpec
        _checkpoints = [
            CheckpointSpec(
                id=p["id"],
                kind=p.get("kind", "mechanical"),
                objective=p.get("tool", p.get("worker", p["id"])),
            )
            for p in self._recipe.get("phases", [])
        ]
        self.task_ledger = TaskLedger.new(
            run_id=self.progress.run_id,
            user_request=f"inject {', '.join(self.inbox_files)} → {target_dir}",
            checkpoints=_checkpoints,
        )
        try:
            self.task_ledger.save()
        except Exception as _e:
            logger.debug("TaskLedger save failed (suppressed): %s", _e)

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

    def _progress_note(
        self,
        task_id: str,
        capability_name: str,
        status: str,
        *,
        output_ref: str | None = None,
        content_hash: str | None = None,
        error: str | None = None,
    ) -> None:
        """Shadow: record FSM progress in ProgressLedger; never affects FSM control flow."""
        try:
            from silica.planner.progress import TaskStatus
            if not any(t.id == task_id for t in self.progress.tasks):
                self.progress.add_task(capability_name, task_id=task_id)
            if status == "done":
                self.progress.mark_done(task_id, output_ref=output_ref, content_hash=content_hash)
            elif status == "failed":
                self.progress.mark_failed(task_id, error or "")
            else:
                self.progress.set_status(task_id, status, error=error)  # type: ignore[arg-type]
            self.progress.save()
        except Exception as _e:
            logger.debug("progress shadow error (suppressed): %s", _e)

    def _save_knowledge_block(self, chunk_idx: int, ops_path: str) -> str:
        """Persist validated ops to a stable (non-tmp) path in the run directory.

        Returns the persistent path so it can be stored as a task output_ref
        and reused on re-runs (content-addressed idempotency).
        """
        import shutil
        kb_dir = self.progress.run_dir / "checkpoints" / f"chunk_{chunk_idx}"
        kb_dir.mkdir(parents=True, exist_ok=True)
        kb_path = str(kb_dir / "validated_ops.json")
        shutil.copy2(ops_path, kb_path)
        return kb_path

    def run(self) -> dict[str, Any]:
        """Execute the pipeline end-to-end (single or multi-file)."""
        from silica.kernel.ledger import get_ledger
        ledger = get_ledger()

        # Compute per-file canonicals and content hashes; track committed status
        self._file_canonicals = []
        self._file_content_hashes = []
        all_committed = True
        for inbox_file in self.inbox_files:
            canonical = self._source_canonical_for(inbox_file)
            self._file_canonicals.append(canonical)
            try:
                content_bytes = open(inbox_file, "rb").read()
                content_hash = hashlib.sha256(content_bytes).hexdigest()
            except OSError:
                content_hash = ""
            self._file_content_hashes.append(content_hash)
            if not ledger.is_committed(canonical, content_hash=content_hash):
                all_committed = False

        # Compat keys for first file (used by single-file code paths and RECON)
        self.context["source_canonical"] = self._file_canonicals[0] if self._file_canonicals else ""
        self.context["source_content_hash"] = self._file_content_hashes[0] if self._file_content_hashes else ""

        if all_committed:
            self.context["final_status"] = "already_ingested"
            return self.context

        self.state = InjectorState.RECON
        return self._run_loop()

    def _run_loop(self) -> dict[str, Any]:
        """Override: remove txn guard so per-chunk phases route to ROLLBACK without a live txn."""
        try:
            while self.state not in (self._done_state, self._error_state):
                try:
                    logger.debug("FSM Transition: %s -> executing handler", self.state.name)
                    self.step()
                except Exception as e:
                    logger.error("FSM Error in state %s: %s", self.state, e)
                    self.context["error"] = str(e)
                    next_state = self._ON_ERROR.get(self.state, self._error_state)
                    if next_state == self._rollback_state:
                        self.context["abort_reason"] = str(e)
                        self.state = self._rollback_state
                    else:
                        self.state = self._error_state
        finally:
            self._cleanup_tmp()
        return self.context

    def _on_sequence_end(self) -> None:
        self._eval_loop_or_done()

    def _on_cleanup_done(self) -> None:
        self._eval_loop_or_done()

    def _eval_loop_or_done(self) -> None:
        """Check if there are more chunks to process or if the queue is empty."""
        self._get_chunks_from_context_if_empty()
        if self._current_chunk_idx + 1 < len(self._chunks):
            self._current_chunk_idx += 1
            logger.info(f"✔ Batch completed successfully. Advancing to batch {self._current_chunk_idx + 1}")
            # Restart per-chunk loop from COLLISION (Phase 5) if present, else DELEGATE
            has_collision = any(
                p.get("id") == "collision"
                for p in self._recipe.get("phases", [])
            )
            self.state = InjectorState.COLLISION if has_collision else InjectorState.DELEGATE
        else:
            logger.info("🎉 All batched chunks have been successfully injected and verified!")
            self.state = InjectorState.DONE

    # ------------------------------------------------------------------
    # State Handlers
    # ------------------------------------------------------------------

    def _handle_recon(self) -> None:
        self._progress_note("recon", "recon", "running")

        # Iterate all inbox files and aggregate recon reports into a list
        recon_list: list[dict] = []
        deferred_notices: list[dict] = []
        for fi, inbox_file in enumerate(self.inbox_files):
            res = silica_recon(inbox_file)
            if "error" in res:
                self._progress_note("recon", "recon", "failed", error=res["error"])
                raise RuntimeError(f"Recon failed for {inbox_file}: {res['error']}")
            recon_list.append(res)

            # Surface any deferred ops from a previous run of this file
            content_hash = self._file_content_hashes[fi] if fi < len(self._file_content_hashes) else ""
            if content_hash:
                from silica.kernel.deferred import get_deferred_store
                bundle = get_deferred_store().get(content_hash)
                if bundle:
                    rejected_count = len(bundle.get("rejected_ops", []))
                    logger.info(
                        "RECON: %d deferred op(s) from a previous run of '%s' are waiting. "
                        "Call silica_deferred_retry('%s') to attempt them.",
                        rejected_count, inbox_file, content_hash[:8],
                    )
                    deferred_notices.append({
                        "inbox_file": inbox_file,
                        "content_hash": content_hash,
                        "rejected_count": rejected_count,
                    })

        # Always a list — even for single-file runs — so _handle_payload is uniform
        self.context["recon"] = recon_list
        if deferred_notices:
            self.context["deferred"] = deferred_notices[0] if len(deferred_notices) == 1 else deferred_notices

        self._progress_note("recon", "recon", "done")
        self._transition_success()

    def _source_canonical_for(self, inbox_file: str) -> str:
        """Vault-relative canonical path for an arbitrary inbox file (no .md, lowercase)."""
        vault_path = getattr(CONFIG, "vault_path", None) or ""
        if vault_path:
            try:
                from pathlib import Path as _P
                rel = _P(inbox_file).relative_to(_P(vault_path)).as_posix()
                return rel.removesuffix(".md").lower()
            except ValueError:
                pass
        return os.path.splitext(os.path.basename(inbox_file))[0].lower()

    def _source_canonical(self) -> str:
        """Vault-relative canonical path for the first inbox file (compat wrapper)."""
        return self._source_canonical_for(self.inbox_file)

    def _chunk_task_id(self, cap: str) -> str:
        """Return the task ID for the current chunk using the f{fi}_c{ci}_{cap} scheme."""
        fi, ci = self._chunk_flat_to_fi_ci.get(self._current_chunk_idx, (0, self._current_chunk_idx))
        return f"f{fi}_c{ci}_{cap}"

    def _handle_payload(self) -> None:
        self._progress_note("payload", "payload", "running")
        # self.context["recon"] is now always a list of per-file recon dicts
        recon_path = self._make_tmp(self.context["recon"])
        phase_conf = self._get_recipe_phase("payload")
        max_concepts = phase_conf.get("partition_if_over", 200)
        res = silica_payload(recon_path, max_concepts=max_concepts)
        if "error" in res:
            self._progress_note("payload", "payload", "failed", error=res["error"])
            raise RuntimeError(f"Payload failed: {res['error']}")
        self.context["payload"] = res

        # Build per-file chunk hierarchy (§3.6).
        # Try to use partition_by_file when the payload has proper batch structure;
        # fall back to the legacy flat-chunk path when batches are absent (e.g. tests).
        from silica.kernel.partition import partition_by_file

        raw_payload: dict | None = None
        if "chunks" in res and res["chunks"]:
            all_batches: list[dict] = []
            for chunk in res["chunks"]:
                all_batches.extend(chunk.get("batches", []))
            if all_batches:
                raw_payload = {
                    "schema_version": res["chunks"][0].get("schema_version", 1),
                    "batches": all_batches,
                }
        elif "payload" in res:
            raw_payload = res["payload"]

        if raw_payload and max_concepts > 0:
            attempt = partition_by_file(raw_payload, max_concepts)
            if attempt:
                self._file_chunks = attempt

        if not self._file_chunks:
            # Fallback: all chunks belong to the first (or only) inbox file.
            # Do NOT split by chunk — one physical file = one file group.
            raw_chunks = res.get("chunks", [])
            if not raw_chunks and "payload" in res:
                raw_chunks = [res["payload"]]
            if not raw_chunks:
                raw_chunks = [res]
            self._file_chunks.append({"source_file": self.inbox_file, "chunks": raw_chunks})

        # Build flat chunk list preserving file order (for existing handler logic)
        self._chunks = []
        self._chunk_flat_to_fi_ci = {}
        flat_idx = 0
        for fi, fg in enumerate(self._file_chunks):
            for ci, chunk in enumerate(fg.get("chunks", [])):
                self._chunks.append(chunk)
                self._chunk_flat_to_fi_ci[flat_idx] = (fi, ci)
                flat_idx += 1

        if not self._chunks:
            self._chunks = [res]
            self._chunk_flat_to_fi_ci = {0: (0, 0)}

        self._current_chunk_idx = 0

        # Build facts["sources"] with per-file concept + chunk counts
        sources_facts: list[dict] = []
        for fi, fg in enumerate(self._file_chunks):
            n_chunks = len(fg.get("chunks", []))
            n_concepts = sum(
                len(b.get("concepts", []))
                for chunk in fg.get("chunks", [])
                for b in chunk.get("batches", [])
            )
            sources_facts.append({
                "inbox_file": fg["source_file"],
                "concepts": n_concepts,
                "chunks": n_chunks,
            })

        # Stash per-file stats in progress inputs for the digest
        self.progress.inputs["sources"] = sources_facts

        # Register per-chunk tasks with f{fi}_c{ci}_{cap} IDs and intra-file deps
        caps = ("collision", "distill", "sanitize", "validate", "snapshot", "write", "hub_update", "autolink", "lint", "cleanup")
        for fi, fg in enumerate(self._file_chunks):
            prev_in_file = "payload"
            for ci in range(len(fg.get("chunks", []))):
                for cap in caps:
                    tid = f"f{fi}_c{ci}_{cap}"
                    self.progress.add_task(cap, task_id=tid, depends_on=[prev_in_file])
                    prev_in_file = tid
        try:
            self.progress.save()
        except Exception as _e:
            logger.debug("progress save error (suppressed): %s", _e)

        self._progress_note("payload", "payload", "done")
        logger.info(
            "Pipeline initialized: %d file(s), %d total chunk(s). Files: %s",
            len(self._file_chunks),
            len(self._chunks),
            [fg["source_file"] for fg in self._file_chunks],
        )
        self._transition_success()

    def _handle_salience(self) -> None:
        """Thematic salience gate — Phase 2.05.

        Single-pass over ALL chunks: drops concepts whose embedding is too far
        from the document's thematic centroid.  Best-effort: any failure
        (embedder down, empty index) is logged and chunks pass unchanged.
        Does NOT re-run on subsequent chunk iterations — _eval_loop_or_done
        restarts from COLLISION, which is correct.
        """
        if not getattr(CONFIG, "salience_gate_enabled", True):
            self._transition_success()
            return

        τ_theme = getattr(CONFIG, "sim_threshold_theme", 0.35)
        try:
            from silica.agent.providers import get_embedder
            from silica.kernel.embed import document_theme_vector, _cosine
            from silica.kernel.recon import _strip_frontmatter
            embedder = get_embedder(CONFIG)
        except Exception as _e:
            logger.warning("SALIENCE: embedder unavailable (%s) — skipping", _e)
            self._transition_success()
            return

        self._get_chunks_from_context_if_empty()
        theme_cache: dict[str, list[float]] = {}
        dropped = 0

        for chunk in self._chunks:
            for batch in chunk.get("batches", []):
                inbox_file = batch.get("inbox_file", self.inbox_file)
                if inbox_file not in theme_cache:
                    try:
                        body = _strip_frontmatter(DRIVER.read_note(inbox_file).content)
                    except Exception:
                        body = ""
                    theme_cache[inbox_file] = document_theme_vector(embedder, body)
                theme = theme_cache[inbox_file]
                if not theme:
                    continue

                concepts = batch.get("concepts", [])
                texts = [
                    (c.get("name", "") + "\n" + c.get("inbox_excerpt", "")) if isinstance(c, dict) else str(c)
                    for c in concepts
                ]
                if not texts:
                    continue
                try:
                    vecs = embedder.embed(texts)
                except Exception as _e:
                    logger.debug("SALIENCE: embed failed (%s) — keeping batch", _e)
                    continue

                kept = []
                for c, v in zip(concepts, vecs):
                    score = _cosine(v, theme)
                    name = c.get("name", "") if isinstance(c, dict) else str(c)
                    if score < τ_theme:
                        logger.info(
                            "SALIENCE: drop '%s' (score=%.3f < τ_theme=%.2f)", name, score, τ_theme
                        )
                        dropped += 1
                    else:
                        kept.append(c)
                batch["concepts"] = kept

        self.context["salience_dropped"] = dropped
        if dropped:
            logger.info("SALIENCE: %d concept(s) below thematic threshold removed", dropped)
        self._transition_success()

    def _handle_collision(self) -> None:
        """Dedup/collision routing — Phase 5.

        For each concept in the current chunk:
        - score ≥ τ_high  → pre-route as a 'patch' op on the existing note
                            (graph check: note must exist in vault)
        - τ_low < score < τ_high → defer (borderline, ambiguous)
        - score ≤ τ_low   → keep for normal distillation (new write)

        Best-effort: any failure (missing index, embedder down) silently skips
        the check and lets the chunk flow to DELEGATE unchanged.
        """
        idx = self._current_chunk_idx
        self._progress_note(self._chunk_task_id("collision"), "collision", "running")

        τ_high = getattr(CONFIG, "sim_threshold_high", 0.85)
        τ_low = getattr(CONFIG, "sim_threshold_low", 0.65)

        try:
            from silica.agent.providers import get_embedder
            from silica.kernel.embed import EmbedStore

            store = EmbedStore()
            if len(store) == 0:
                logger.info("COLLISION: embedding index empty — skipping (build with silica_embed_refresh)")
                self._progress_note(self._chunk_task_id("collision"), "collision", "done")
                self._transition_success()
                return
            embedder = get_embedder(CONFIG)
        except Exception as _e:
            logger.warning("COLLISION: embedder unavailable (%s) — skipping", _e)
            self._progress_note(self._chunk_task_id("collision"), "collision", "done")
            self._transition_success()
            return

        self._get_chunks_from_context_if_empty()
        chunk = self._chunks[idx]

        pre_routed_ops: list[dict] = []
        deferred_concepts: list[dict] = []
        modified_batches: list[dict] = []

        for batch in chunk.get("batches", []):
            inbox_file = batch.get("inbox_file", self.inbox_file)
            kept: list = []

            for concept in batch.get("concepts", []):
                concept_text = concept.get("name", "") if isinstance(concept, dict) else str(concept)
                if not concept_text:
                    kept.append(concept)
                    continue

                try:
                    vecs = embedder.embed([concept_text])
                    results = store.cosine_top_k(vecs[0], k=1)
                except Exception as _embed_err:
                    logger.debug("COLLISION: embed failed for '%s': %s", concept_text, _embed_err)
                    kept.append(concept)
                    continue

                if not results:
                    kept.append(concept)
                    continue

                top = results[0]
                score: float = top.get("score", 0.0)

                if score >= τ_high:
                    existing_path = top.get("path", "")
                    try:
                        DRIVER.read_note(existing_path)
                        # Graph confirms node exists — safe to patch
                        logger.info(
                            "COLLISION: '%s' → patch '%s' (score=%.3f ≥ τ_high=%.2f)",
                            concept_text, existing_path, score, τ_high,
                        )
                        pre_routed_ops.append({
                            "op": "patch",
                            "path": existing_path,
                            "heading": concept_text,
                            "source_basename": os.path.basename(inbox_file),
                            "snippet": concept.get("excerpt", "") if isinstance(concept, dict) else "",
                            "hub": self.hub,
                            "reason": f"collision_routed score={score:.3f}",
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
            content_hash = self.context.get("source_content_hash", "")
            if content_hash:
                try:
                    from silica.kernel.deferred import get_deferred_store
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
                    get_deferred_store().put(
                        content_hash=content_hash,
                        source_path=self.inbox_file,
                        target_dir=self.target_dir,
                        hub=self.hub,
                        rejected_ops=deferred_op_dicts,
                        rejection_reasons={
                            (d["concept"].get("name", str(i)) if isinstance(d["concept"], dict) else str(i)):
                            f"borderline_similarity score={d['score']:.3f}"
                            for i, d in enumerate(deferred_concepts)
                        },
                    )
                except Exception as _de:
                    logger.warning("COLLISION: failed to save deferred concepts: %s", _de)

        # Store pre-routed ops for merging in VALIDATE (Phase 5)
        self.context[f"chunk_{idx}_collision_ops"] = pre_routed_ops

        # Replace chunk with filtered version (remove patched/deferred concepts)
        self._chunks[idx] = {
            "schema_version": chunk.get("schema_version", 1),
            "batches": modified_batches,
        }

        self._progress_note(
            self._chunk_task_id("collision"), "collision", "done",
            output_ref=f"{len(pre_routed_ops)} patch-routed, {len(deferred_concepts)} deferred",
        )
        self._transition_success()

    def _handle_delegate(self) -> None:
        from silica.kernel.prep_delegation import run_distiller

        self._get_chunks_from_context_if_empty()

        if not self._chunks or self._current_chunk_idx >= len(self._chunks):
            raise RuntimeError("No chunks available for iterative processing.")

        current_chunk = self._chunks[self._current_chunk_idx]
        idx = self._current_chunk_idx

        # Content-addressed idempotency (Phase 2): if this chunk was already
        # processed in a prior run with identical input, skip DELEGATE→SANITIZE→VALIDATE
        # and reuse the persisted knowledge-block ops file.
        import json as _json
        chunk_hash = hashlib.sha256(
            _json.dumps(current_chunk, sort_keys=True).encode()
        ).hexdigest()
        saved_ops_path = self.progress.is_checkpoint_done(self._chunk_task_id("validate"), chunk_hash)
        if saved_ops_path and os.path.exists(saved_ops_path):
            logger.info(
                "DELEGATE chunk %d: content-addressed hit (hash=%s…) — skipping to SNAPSHOT",
                idx,
                chunk_hash[:8],
            )
            self.context["ops_path"] = saved_ops_path
            self.state = InjectorState.SNAPSHOT
            return

        logger.info(f"--- DISTILLING BATCH {idx + 1}/{len(self._chunks)} ---")
        self._progress_note(self._chunk_task_id("distill"), "distill", "running")

        # Assemble compact ledger digest for LLM context (Phase 2 rails).
        ledger_digest: str | None = None
        try:
            ledger_digest = self.progress.digest()
        except Exception:
            pass

        # Phase 6: pass steering correction if VALIDATE sent us back here
        steer_context: str | None = self.context.get(f"chunk_{idx}_steer_context")
        if steer_context:
            logger.info("DELEGATE chunk %d: re-attempt with steering correction", idx)

        try:
            chunk_result = run_distiller(
                payload=current_chunk,
                target=self.target_dir,
                hub=self.hub,
                ledger_digest=ledger_digest,
                steer_context=steer_context,
            )
            if "error" in chunk_result:
                self._progress_note(self._chunk_task_id("distill"), "distill", "failed", error=chunk_result["error"])
                raise RuntimeError(f"Distiller error on batch {idx}: {chunk_result['error']}")

            distiller_path = self._make_tmp(chunk_result)
            self.context["distiller_output_path"] = distiller_path
            # Store chunk hash for knowledge-block write at VALIDATE
            self.context[f"chunk_{idx}_hash"] = chunk_hash
            self._progress_note(self._chunk_task_id("distill"), "distill", "done", output_ref=distiller_path)
            self._transition_success()

        except Exception as e:
            raise RuntimeError(f"Critical failure delegating batch {idx}: {e}")

    def _handle_sanitize(self) -> None:
        idx = self._current_chunk_idx
        self._progress_note(self._chunk_task_id("sanitize"), "sanitize", "running")
        res = silica_sanitize(self.context["distiller_output_path"])
        if "error" in res:
            self._progress_note(self._chunk_task_id("sanitize"), "sanitize", "failed", error=res["error"])
            raise RuntimeError(f"Sanitize failed: {res['error']}")
        self.context["sanitized"] = res
        self._progress_note(self._chunk_task_id("sanitize"), "sanitize", "done")
        self._transition_success()

    def _handle_validate(self) -> None:
        idx = self._current_chunk_idx
        self._progress_note(self._chunk_task_id("validate"), "validate", "running")
        sanitized = self.context["sanitized"]["parsed"]
        ops_raw = sanitized.get("updates", sanitized) if isinstance(sanitized, dict) else sanitized
        if not isinstance(ops_raw, list):
            ops_raw = [ops_raw]

        # Merge collision-routed patch ops (Phase 5): prepend so they go through
        # the same validate→snapshot→write path as distiller-generated ops.
        collision_ops = self.context.get(f"chunk_{idx}_collision_ops", [])
        if collision_ops:
            ops_raw = list(collision_ops) + list(ops_raw)

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
            self._progress_note(self._chunk_task_id("validate"), "validate", "done")
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
            # Phase 6 steering arc: re-delegate with rejection reason injected (max 2 attempts).
            steer_attempts = self.context.get(f"chunk_{idx}_steer_attempts", 0)
            _max_steer = self._get_recipe_gate("max_steer_attempts", 2)
            if steer_attempts < _max_steer:
                steer_attempts += 1
                self.context[f"chunk_{idx}_steer_attempts"] = steer_attempts
                # Build a short rejection summary to inject as corrective context.
                rejected_raw = res.get("rejected_ops", [])
                reasons = "; ".join(
                    r.get("reason", "") for r in rejected_raw if isinstance(r, dict) and r.get("reason")
                )
                steer_msg = (
                    f"|attempt={steer_attempts}|"
                    f" All {res.get('rejected_count', '?')} ops were rejected."
                    f" Reasons: {reasons or 'no reason provided'}."
                    f" Produce valid ops that satisfy the pipeline constraints."
                )
                self.context[f"chunk_{idx}_steer_context"] = steer_msg
                logger.warning(
                    "VALIDATE: steer attempt %d/%d for chunk %d — re-delegating with correction.",
                    steer_attempts, _max_steer, idx,
                )
                self._progress_note(self._chunk_task_id("validate"), "validate", "running",
                                    error=f"steer {steer_attempts}/{_max_steer}")
                try:
                    self.progress.set_status(  # type: ignore[union-attr]
                        self._chunk_task_id("distill"), "in_progress",
                        error=f"steer attempt {steer_attempts}"
                    )
                except Exception:
                    pass
                self.state = InjectorState.DELEGATE
                return
            # Exhausted steering budget → defer and short-circuit.
            logger.warning("VALIDATE: steer budget exhausted (%d/%d) — deferring chunk %d.", steer_attempts, _max_steer, idx)
            self.context["abort_reason"] = "All ops rejected — nothing to write"
            self.context["final_status"] = "no_ops"
            self.context["ops_path"] = ops_path
            self._progress_note(self._chunk_task_id("validate"), "validate", "done")
            self.state = InjectorState.CLEANUP
            return

        # Knowledge-block consolidation (Phase 2): persist the validated ops to
        # a stable path in the run directory so they survive tmp cleanup and
        # enable content-addressed skip on re-runs.
        chunk_hash = self.context.get(f"chunk_{idx}_hash", "")
        kb_path: str = ops_path  # fallback to tmp if save fails
        if chunk_hash:
            try:
                kb_path = self._save_knowledge_block(idx, ops_path)
            except Exception as _kb_e:
                logger.debug("Knowledge-block save failed (non-fatal): %s", _kb_e)

        self.context["ops_path"] = kb_path
        self._progress_note(
            f"chunk_{self._current_chunk_idx}_validate",
            "validate",
            "done",
            output_ref=kb_path,
            content_hash=chunk_hash or None,
        )
        self._transition_success()

    def _handle_snapshot(self) -> None:
        idx = self._current_chunk_idx
        self._progress_note(self._chunk_task_id("snapshot"), "snapshot", "running")
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

        self._progress_note(self._chunk_task_id("snapshot"), "snapshot", "done")
        self._transition_success()

    def _handle_write(self) -> None:
        idx = self._current_chunk_idx
        self._progress_note(self._chunk_task_id("write"), "write", "running")
        res = silica_bulk_write(self.context["ops_path"])

        if "error" in res:
            self._progress_note(self._chunk_task_id("write"), "write", "failed", error=res["error"])
            raise RuntimeError(f"Write failed: {res['error']}")
        if not res.get("success", False):
            failed = res.get("failed_operations", "?")
            total = res.get("total_operations", "?")
            raise RuntimeError(
                f"Write partially failed: {failed}/{total} operations failed. "
                f"Results: {res.get('results', [])}"
            )

        self.context["write"] = res
        self._progress_note(self._chunk_task_id("write"), "write", "done")
        self._transition_success()

    def _handle_hub_update(self) -> None:
        """Append MOC links to the Hub note for all newly written notes."""
        idx = self._current_chunk_idx
        self._progress_note(self._chunk_task_id("hub_update"), "hub_update", "running")
        if not self.hub:
            logger.info("HUB_UPDATE: no hub configured, skipping")
            self._progress_note(self._chunk_task_id("hub_update"), "hub_update", "done")
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
            self._progress_note(self._chunk_task_id("hub_update"), "hub_update", "done")
            self._transition_success()
            return

        hub_path = f"{self.target_dir}/{hub_name}.md".replace("//", "/")
        from silica.driver.base import NoteRef
        hub_ref = NoteRef(name=hub_name, path=hub_path)

        try:
            hub_note = DRIVER.read_note(hub_ref)
        except Exception as e:
            logger.warning("HUB_UPDATE: hub '%s' not readable: %s — skipping", hub_path, e)
            self._progress_note(self._chunk_task_id("hub_update"), "hub_update", "done")
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

        self._progress_note(self._chunk_task_id("hub_update"), "hub_update", "done")
        self._transition_success()

    def _handle_autolink(self) -> None:
        """Best-effort wikilink injection into touched notes (Phase 4).

        Runs autolink on every note written by this chunk.  Failures are
        non-fatal: they are logged and the FSM continues to LINT.  This is
        intentional — autolink only ADDs links; it can never break a valid note.
        """
        idx = self._current_chunk_idx
        self._progress_note(self._chunk_task_id("autolink"), "autolink", "running")

        try:
            from silica.kernel.autolink import autolink, build_title_index

            ops = load_ops(self.context["ops_path"])
            touched_paths = [
                op.touched_ref()
                for op in ops
                if op.touched_ref() and op.op not in (OpType.delete, OpType.skip)
            ]

            if not touched_paths:
                self._progress_note(self._chunk_task_id("autolink"), "autolink", "done")
                self._transition_success()
                return

            all_refs = DRIVER.list_files()
            title_index = build_title_index(all_refs)

            total_added = 0
            for path in touched_paths:
                try:
                    note_title = os.path.splitext(os.path.basename(path))[0]
                    nc = DRIVER.read_note(path)
                    new_body, added = autolink(nc.content or "", title_index, self_title=note_title)
                    if added:
                        DRIVER.overwrite(path, new_body)
                        total_added += len(added)
                        logger.info("AUTOLINK: %s — added %d link(s): %s", path, len(added), added)
                except Exception as _ae:
                    logger.debug("AUTOLINK: skipped '%s' (non-fatal): %s", path, _ae)

            logger.info("AUTOLINK: finished — %d link(s) added across %d note(s)", total_added, len(touched_paths))
        except Exception as e:
            # AUTOLINK is best-effort: log and continue to LINT
            logger.warning("AUTOLINK: phase failed (non-fatal): %s", e)

        self._progress_note(self._chunk_task_id("autolink"), "autolink", "done")
        self._transition_success()

    def _handle_lint(self) -> None:
        idx = self._current_chunk_idx
        self._progress_note(self._chunk_task_id("lint"), "lint", "running")
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
                self._progress_note(self._chunk_task_id("lint"), "lint", "failed", error=self.context["abort_reason"])
                self.state = InjectorState.ROLLBACK
                return

        # S3.2: Run graph-diff check
        regression_rule = self._get_recipe_gate("graph_regression", "forbid_new_orphans")
        if regression_rule != "allow":
            if self._pre_graph is None:
                self.context["abort_reason"] = "Graph regression gate failed: pre-write snapshot is missing"
                self._progress_note(self._chunk_task_id("lint"), "lint", "failed", error=self.context["abort_reason"])
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
                    self._progress_note(self._chunk_task_id("lint"), "lint", "failed", error=self.context["abort_reason"])
                    self.state = InjectorState.ROLLBACK
                    return
            except Exception as e:
                logger.error("Failed to perform graph-diff check: %s", e)
                self.context["abort_reason"] = f"Graph regression gate error during check: {e}"
                self._progress_note(self._chunk_task_id("lint"), "lint", "failed", error=self.context["abort_reason"])
                self.state = InjectorState.ROLLBACK
                return

        self._progress_note(self._chunk_task_id("lint"), "lint", "done")
        self._transition_success()

    def _handle_cleanup(self) -> None:
        from silica.tools.wrapped import silica_cleanup

        self._get_chunks_from_context_if_empty()
        fi, ci = self._chunk_flat_to_fi_ci.get(self._current_chunk_idx, (0, self._current_chunk_idx))
        self._progress_note(self._chunk_task_id("cleanup"), "cleanup", "running")

        # Always write ledger for this chunk's ops (per chunk)
        self._write_ledger_for_file(fi, "committed")

        # Archive the physical file only on the last chunk of its file group
        file_group = self._file_chunks[fi] if fi < len(self._file_chunks) else {}
        n_chunks_in_file = len(file_group.get("chunks", []))
        is_last_chunk_of_file = (ci + 1 >= n_chunks_in_file)

        if is_last_chunk_of_file:
            # Only archive if no chunk of this file failed
            fi_prefix = f"f{fi}_"
            file_has_failure = any(
                t.status == "failed" for t in self.progress.tasks
                if t.id.startswith(fi_prefix)
            )
            if not file_has_failure:
                inbox_file_for_fi = file_group.get("source_file", self.inbox_file)
                res = silica_cleanup(inbox_file_for_fi, "done")
                if "error" in res:
                    self.context["cleanup_warning"] = res["error"]
            else:
                logger.info(
                    "File %d (%s) had chunk failures — not archiving.",
                    fi, file_group.get("source_file", "?"),
                )
        else:
            logger.info(
                "Chunk f%d_c%d done. Archiving deferred until last chunk of file %d.",
                fi, ci, fi,
            )

        if self.context.get("final_status") != "no_ops":
            if self.context.get("has_partial_failure"):
                self.context["final_status"] = "partial"
            else:
                self.context["final_status"] = "Success"
        self._progress_note(self._chunk_task_id("cleanup"), "cleanup", "done")
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

        self._progress_note("rollback", "rollback", "done")
        # Contain the failure at chunk level instead of aborting the whole run
        self._contain_chunk_failure()

    def _contain_chunk_failure(self) -> None:
        """Contain a per-chunk failure: mark failed tasks, reset context, advance.

        Called after rollback completes (or as a no-op when no txn existed).
        Preserves all previously committed chunks — only the failing chunk is
        affected.  If more chunks remain, the FSM restarts from COLLISION;
        otherwise it concludes with final_status="partial".
        """
        idx = self._current_chunk_idx
        fi, ci = self._chunk_flat_to_fi_ci.get(idx, (0, idx))
        abort_reason = self.context.get("abort_reason", "chunk failure")

        # Mark all f{fi}_c{ci}_* tasks that are not already done as failed
        prefix = f"f{fi}_c{ci}_"
        for task in self.progress.tasks:
            if task.id.startswith(prefix) and task.status not in ("done",):
                try:
                    self.progress.mark_failed(task.id, error=abort_reason[:200])
                except Exception:
                    pass
        try:
            self.progress.save()
        except Exception:
            pass

        # Reset per-chunk context to prevent state leakage to the next chunk
        for key in ("ops_path", "sanitized", "distiller_output_path", "snapshot", "txn_id", "abort_reason"):
            self.context.pop(key, None)
        self.context.pop(f"chunk_{idx}_collision_ops", None)
        self.context.pop(f"chunk_{idx}_hash", None)
        self.context.pop(f"chunk_{idx}_steer_context", None)
        self.context.pop(f"chunk_{idx}_steer_attempts", None)
        self._txn = None
        self._pre_graph = None

        # Record that at least one chunk failed (used by cleanup to set "partial")
        self.context["has_partial_failure"] = True

        # Advance to next chunk, or conclude the run as partial
        self._get_chunks_from_context_if_empty()
        if self._current_chunk_idx + 1 < len(self._chunks):
            self._current_chunk_idx += 1
            logger.info(
                "Chunk f%d_c%d failed — advancing to chunk %d of %d.",
                fi, ci, self._current_chunk_idx + 1, len(self._chunks),
            )
            has_collision = any(
                p.get("id") == "collision"
                for p in self._recipe.get("phases", [])
            )
            self.state = InjectorState.COLLISION if has_collision else InjectorState.DELEGATE
        else:
            logger.info(
                "Chunk f%d_c%d failed (last chunk). Run concludes with partial success.", fi, ci
            )
            self.context["final_status"] = "partial"
            self.state = InjectorState.DONE

    # ------------------------------------------------------------------
    # Ledger helpers (C5)
    # ------------------------------------------------------------------

    def _write_ledger_for_file(self, fi: int, status: str) -> None:
        """Record this chunk's ops into the ledger, attributed to file fi."""
        try:
            from silica.kernel.ledger import get_ledger
            ledger = get_ledger()
            txn_id = self.context.get("txn_id", "unknown")

            # Use per-file canonical/hash when available; fall back to context
            if fi < len(self._file_canonicals):
                source_canonical = self._file_canonicals[fi]
                content_hash = self._file_content_hashes[fi] if fi < len(self._file_content_hashes) else None
            else:
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
            logger.warning("Failed to write ledger for file %d: %s", fi, e)

    def _write_ledger(self, status: str) -> None:
        """Record all ops from ops_path into the ledger (single-file compat wrapper)."""
        self._write_ledger_for_file(0, status)

    def _write_ledger_rollback(self, txn_id: str) -> None:
        try:
            from silica.kernel.ledger import get_ledger
            get_ledger().mark_rolled_back(txn_id)
        except Exception as e:
            logger.warning("Failed to mark rollback in ledger: %s", e)
