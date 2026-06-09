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
import re
import time
from enum import Enum, auto
from typing import Any, Callable, TYPE_CHECKING

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


def _refresh_cooccurrence_for_ops(
    ops: list,
    committed_paths: set,
    *,
    read_body: Callable[[str], str],
    lang: str = "english",
    store: Any | None = None,
) -> int:
    """Refresh the embedder-free co-occurrence index for committed write/patch ops.

    The freshness twin of the embedding refresh, but the STABLE leg: it imports
    only the cooccurrence module (never the embedder/provider stack), so the
    index stays fresh even when LM Studio is down. Uses build_index(force=True)
    so a note's prior contribution is replaced, never inflated, with a single
    save. Best-effort: a per-note read failure is skipped and the whole call
    never raises. Returns the number of notes refreshed.
    """
    from silica.kernel.cooccurrence import build_index as _cooccur_build

    notes: list[tuple[str, str, str]] = []
    concepts_by_path: dict[str, list[str]] = {}
    seen: set[str] = set()
    for op in ops:
        path = op.touched_ref()
        if op.op not in (OpType.write, OpType.patch) or not path:
            continue
        if path not in committed_paths or path in seen:
            continue
        seen.add(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        idx_path = path.removesuffix(".md")
        try:
            body = read_body(path) or ""
        except Exception:
            continue
        notes.append((idx_path, stem, body))
        # #9: forward LLM-extracted concept phrases to reinforce this note.
        op_concepts = getattr(op, "concepts", None)
        if op_concepts:
            concepts_by_path[idx_path] = op_concepts

    if not notes:
        return 0
    try:
        _cooccur_build(notes, store=store, lang=lang, force=True,
                       concepts_by_path=concepts_by_path or None)
    except Exception as exc:
        logger.debug("WRITE: cooccur refresh skipped (%s)", exc)
        return 0
    return len(notes)


def _inject_graph_ctx(chunk: dict, vault_ctx: dict) -> dict:
    """Return a shallow-enriched copy of chunk with graph_context added to concepts.

    For every concept that has a vault_collision, adds a graph_context field:
        {"cluster_id": int, "hub": str|None, "is_hub": bool}
    Concepts without a vault_collision get graph_context=null.
    The original chunk dict is never mutated.
    """
    if not vault_ctx:
        return chunk

    enriched_batches = []
    for batch in chunk.get("batches", []):
        enriched_concepts = []
        for concept in batch.get("concepts", []):
            if not isinstance(concept, dict):
                enriched_concepts.append(concept)
                continue
            collision = concept.get("vault_collision") or {}
            cpath = collision.get("path", "").removesuffix(".md") if collision else ""
            gctx = vault_ctx.get(cpath)
            graph_context = (
                {
                    "cluster_id": gctx["cluster_id"],
                    "hub": gctx["hub"],
                    "is_hub": gctx["is_hub"],
                }
                if gctx and cpath
                else None
            )
            enriched_concepts.append({**concept, "graph_context": graph_context})
        enriched_batches.append({**batch, "concepts": enriched_concepts})
    return {**chunk, "batches": enriched_batches}


# Italian function-word markers used for language detection in MOC headings.
_ITALIAN_MARKERS_RE = re.compile(
    r'\b(della|dello|degli|delle|del|dal|nel|sul|per|con|una|questo|questa|sono|hanno|viene|vengono)\b',
    re.IGNORECASE,
)


def _moc_prefix(sample: str) -> str:
    """Return 'Da' (Italian) or 'From' (English) based on marker density in sample."""
    return "Da" if len(_ITALIAN_MARKERS_RE.findall(sample)) >= 3 else "From"


def _moc_heading(source_name: str, sample: str) -> str:
    """Language-aware MOC section heading: '## Da: {name}' or '## From: {name}'."""
    return f"## {_moc_prefix(sample)}: {source_name}"


def _merge_moc_section(content: str, heading: str, note_lines: list[str]) -> str:
    """Append note_lines to an existing MOC section or create a new one.

    When the same source file produces multiple chunks, each chunk calls
    HUB_UPDATE.  Rather than duplicating the heading, new links are appended
    inside the existing section.
    """
    if heading + "\n" in content or heading + "\r\n" in content:
        # Append new links just before the next same-level heading or end of file.
        pattern = re.compile(re.escape(heading) + r'(.*?)(?=\n##\s|\Z)', re.DOTALL)
        def _append(m: re.Match) -> str:
            return m.group(0).rstrip() + "\n" + "\n".join(note_lines) + "\n"
        return pattern.sub(_append, content, count=1)
    moc_block = f"\n{heading}\n\n" + "\n".join(note_lines) + "\n"
    return content.rstrip() + "\n" + moc_block


class InjectorState(Enum):
    INIT = auto()
    RECON = auto()         # Phase 1
    CROSSDEDUP = auto()    # Phase 1.5 — cross-file concept deduplication
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
    BACKLINK = auto()      # Phase 4.5 — reverse: inject links to new notes into pre-existing ones
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
        self._undo_run_id: str | None = None          # journal run for this inject
        self._run_inverses: list[tuple[str, "InverseOp", str | None]] = []  # (path, inverse, post_hash)
        self._pre_graph: GraphSnapshot | None = None  # S3.2 pre-write graph snapshot

        # Optional producer channel to the leashed sub-agent pool.  Set by the
        # Coordinator; when None the FSM runs standalone (legacy behaviour) and
        # never produces work items.
        self.work_queue: Any | None = None

        # Optional run-scoped memory of non-blocking warnings (orphans).  Set by
        # the Coordinator; drained for repair at end of run.  None ⇒ no recording.
        self.warning_ledger: Any | None = None

        # Per-file content info — populated by run() before _run_loop starts
        self._file_canonicals: list[str] = []
        self._file_content_hashes: list[str] = []
        self._committed_file_indices: set[int] = set()  # indices of already-committed files

        # Iterative chunk processing state fields
        self._chunks: list[dict] = []
        self._current_chunk_idx: int = 0
        # Multi-file hierarchy (§3.6): per-file chunk groups + flat→(fi,ci) mapping
        self._file_chunks: list[dict] = []  # [{"source_file": str, "chunks": [...]}, ...]
        self._chunk_flat_to_fi_ci: dict[int, tuple[int, int]] = {}  # flat_idx → (file_idx, chunk_idx)

        # Shadow ProgressLedger — mirrors FSM state on disk; FSM remains canonical
        from silica.planner.progress import ProgressLedger, RunManifest
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

        # RunManifest — short-term memory of what was injected in this run
        self.manifest = RunManifest(run_id=self.progress.run_id)

        # S3.3: Load the recipe for dynamic configuration
        from silica.router.recipe_parser import load_recipe
        try:
            self._recipe = load_recipe("injector", domain=getattr(CONFIG, "domain", None))
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
                    { "id": "crossdedup",   "kind": "mechanical", "best_effort": True },
                    { "id": "payload",      "kind": "mechanical", "tool": "silica_payload", "partition_if_over": 200 },
                    { "id": "collision",    "kind": "mechanical", "best_effort": True },
                    { "id": "distill",      "kind": "semantic",   "worker": "distiller", "fanout": True, "max_workers": 7 },
                    { "id": "sanitize",     "kind": "mechanical", "tool": "silica_sanitize" },
                    { "id": "validate",     "kind": "gate",       "tool": "silica_validate_ops", "abort_code": 2 },
                    { "id": "snapshot",     "kind": "txn",        "tool": "silica_snapshot" },
                    { "id": "write",        "kind": "mechanical", "tool": "silica_bulk_write" },
                    { "id": "hub_update",   "kind": "mechanical", "tool": "silica_hub_update" },
                    { "id": "autolink",     "kind": "mechanical", "tool": "silica_autolink",  "best_effort": True },
                    { "id": "backlink",     "kind": "mechanical", "tool": "silica_backlink",  "best_effort": True },
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
            "crossdedup": InjectorState.CROSSDEDUP,
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
            "backlink":   InjectorState.BACKLINK,
            "lint":       InjectorState.LINT,
            "cleanup":    InjectorState.CLEANUP,
            "rollback":   InjectorState.ROLLBACK,
        }

        # S2.2.1: Handlers mapping and error policy
        self._HANDLERS = {
            InjectorState.RECON: self._handle_recon,
            InjectorState.CROSSDEDUP: self._handle_crossdedup,
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
            InjectorState.BACKLINK: self._handle_backlink,
            InjectorState.LINT: self._handle_lint,
            InjectorState.CLEANUP: self._handle_cleanup,
            InjectorState.ROLLBACK: self._handle_rollback,
        }

        self._ON_ERROR = {
            # Setup phases: abort the whole run on failure
            InjectorState.RECON: InjectorState.ERROR,
            InjectorState.CROSSDEDUP: InjectorState.ERROR,
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
        if _resumed:
            # On resume: load the original TaskLedger to preserve its immutable
            # user_request / created_at / checkpoints.  Fall back to creating a
            # fresh one only if the file is missing (e.g. run dir was pruned).
            try:
                self.task_ledger = TaskLedger.load(self.progress.run_id)
            except Exception:
                self.task_ledger = TaskLedger.new(
                    run_id=self.progress.run_id,
                    user_request=f"inject {', '.join(self.inbox_files)} → {target_dir}",
                    checkpoints=_checkpoints,
                )
                try:
                    self.task_ledger.save()
                except Exception as _e:
                    logger.debug("TaskLedger save failed (suppressed): %s", _e)
        else:
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

        # Emit phase event to TUI (no-op if no hook is registered)
        try:
            from silica.agent.progress import emit_pipeline_phase
            emit_pipeline_phase(capability_name, status)
        except Exception:
            pass

    @property
    def _chunk_ctx(self) -> dict:
        """Per-chunk volatile state namespace — cleared atomically on each chunk boundary."""
        return self.context.setdefault("chunk", {})

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

    def _defer_ops(
        self,
        rejected_ops: list[dict],
        rejection_reasons: dict[str, str],
        *,
        phase: str,
    ) -> bool:
        """Persist rejected/failed ops to the deferred store, merging with any
        bundle already saved under this source file's content hash.

        Every defer site (COLLISION, VALIDATE, WRITE) funnels through here so
        the bundle's merge semantics live in exactly one place: because all
        phases of all chunks of one file share the same content_hash, a later
        phase (or a later chunk) must NOT clobber ops an earlier one deferred —
        they accumulate. Returns True iff a bundle was written.
        """
        if not rejected_ops:
            return False
        content_hash = self._current_content_hash
        if not content_hash:
            logger.warning(
                "%s: %d op(s) to defer but no content_hash — deferred store skipped.",
                phase, len(rejected_ops),
            )
            return False
        try:
            from silica.kernel.deferred import get_deferred_store
            store = get_deferred_store()
            existing = store.get(content_hash) or {}
            store.put(
                content_hash=content_hash,
                source_path=self._current_source_file,
                target_dir=self.target_dir,
                hub=self.hub,
                rejected_ops=list(existing.get("rejected_ops", [])) + rejected_ops,
                rejection_reasons={**existing.get("rejection_reasons", {}), **rejection_reasons},
            )
            return True
        except Exception as _de:
            logger.warning("%s: failed to save deferred ops: %s", phase, _de)
            return False

    def run(self) -> dict[str, Any]:
        """Execute the pipeline end-to-end (single or multi-file)."""
        from silica.kernel.ledger import get_ledger
        ledger = get_ledger()

        from silica.kernel.undo_journal import get_undo_journal
        self._undo_run_id = get_undo_journal().start_run(source=self.inbox_file)

        # Compute per-file canonicals and content hashes; track committed status
        self._file_canonicals = []
        self._file_content_hashes = []
        all_committed = True
        for inbox_file in self.inbox_files:
            canonical = self._source_canonical_for(inbox_file)
            self._file_canonicals.append(canonical)
            try:
                content_bytes = DRIVER.read_note(inbox_file).content.encode("utf-8")
                content_hash = hashlib.sha256(content_bytes).hexdigest()
            except Exception:
                try:
                    content_bytes = open(inbox_file, "rb").read()
                    content_hash = hashlib.sha256(content_bytes).hexdigest()
                except OSError:
                    content_hash = ""
            self._file_content_hashes.append(content_hash)
            if not ledger.is_committed(canonical, content_hash=content_hash):
                all_committed = False

        # Build set of already-committed file indices so chunk-advance logic can skip them
        self._committed_file_indices = {
            i for i, (canonical, h) in enumerate(zip(self._file_canonicals, self._file_content_hashes))
            if ledger.is_committed(canonical, content_hash=h)
        }

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
                        self._chunk_ctx["abort_reason"] = str(e)
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
        # Clear the per-chunk volatile namespace atomically before advancing
        self.context.pop("chunk", None)
        self._txn = None
        self._pre_graph = None
        self._get_chunks_from_context_if_empty()
        next_idx = self._next_uncommitted_chunk_idx(self._current_chunk_idx + 1)
        if next_idx < len(self._chunks):
            self._current_chunk_idx = next_idx
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

    def _handle_crossdedup(self) -> None:
        """Cross-file concept deduplication — Phase 1.5.

        Embeds concept names extracted by RECON across all inbox files.
        Near-duplicate concepts from different files (cosine ≥ τ_high) are
        merged: the first-file occurrence is kept, the duplicate is removed.
        Best-effort: silently skips when the embedder is unavailable or
        fewer than two inbox files are present.
        """
        recon_list: list[dict] = self.context.get("recon", [])

        if len(recon_list) < 2:
            self._transition_success()
            return

        # Collect (file_index, concept_name) for all new_concepts across files
        all_concepts: list[tuple[int, str]] = [
            (fi, name)
            for fi, rec in enumerate(recon_list)
            for name in rec.get("new_concepts", [])
        ]

        if len(all_concepts) < 2:
            self._transition_success()
            return

        try:
            from silica.agent.providers import get_embedder
            from silica.kernel.embed import _cosine
            embedder = get_embedder(CONFIG)
        except Exception as _e:
            logger.warning("CROSSDEDUP: embedder unavailable (%s) — skipping", _e)
            self._transition_success()
            return

        texts = [name for _, name in all_concepts]
        try:
            vecs = embedder.embed(texts)
        except Exception as _e:
            logger.warning("CROSSDEDUP: embed call failed (%s) — skipping", _e)
            self._transition_success()
            return

        τ_high = getattr(CONFIG, "sim_threshold_high", 0.85)

        # Greedy O(n²) clustering: mark cross-file near-duplicates for removal.
        # The first occurrence (lowest file index) is always the winner.
        losers: set[int] = set()
        for i in range(len(all_concepts)):
            if i in losers:
                continue
            fi, name_i = all_concepts[i]
            for j in range(i + 1, len(all_concepts)):
                if j in losers:
                    continue
                fj, name_j = all_concepts[j]
                if fi == fj:
                    continue
                if _cosine(vecs[i], vecs[j]) >= τ_high:
                    losers.add(j)
                    logger.info(
                        "CROSSDEDUP: '%s' (file %d) merged into '%s' (file %d, score=%.3f)",
                        name_j, fj, name_i, fi, _cosine(vecs[i], vecs[j]),
                    )

        if not losers:
            self._transition_success()
            return

        for idx in losers:
            fi, name = all_concepts[idx]
            nc = recon_list[fi].get("new_concepts", [])
            if name in nc:
                nc.remove(name)

        self.context["recon"] = recon_list
        self.context["crossdedup_merged"] = len(losers)
        logger.info(
            "CROSSDEDUP: %d duplicate concept(s) removed across %d files",
            len(losers), len(recon_list),
        )
        self._transition_success()

    def _build_vault_graph_ctx(self) -> dict[str, dict]:
        """Compute per-note graph context (cluster/hub/pagerank) from the current vault state.

        Returns a dict keyed by vault-relative path without .md extension:
            {"cluster_id": int, "hub": str|None, "is_hub": bool, "pagerank": float}
        Empty dict on any failure — all consumers treat missing context as a no-op.
        """
        try:
            from silica.kernel.graph_report import compute_report
            _t = time.monotonic()
            report = compute_report()
            ctx: dict[str, dict] = {}
            for cs in report.clusters:
                for member in cs.members:
                    ctx[member] = {
                        "cluster_id": cs.cluster_id,
                        "hub": cs.hub,
                        "is_hub": member == cs.hub,
                        "pagerank": report.pagerank_map.get(member, 0.0),
                    }
            # Include isolated nodes (not in any cluster) so pagerank is available
            for node_id, pr_val in report.pagerank_map.items():
                if node_id not in ctx:
                    ctx[node_id] = {"cluster_id": -1, "hub": None, "is_hub": False, "pagerank": pr_val}
            logger.info(
                "PAYLOAD: vault graph context built — %d nodes, %d clusters (%.2fs)",
                len(ctx), len(report.clusters), time.monotonic() - _t,
            )
            return ctx
        except Exception as _e:
            logger.info("PAYLOAD: vault graph context unavailable (%s) — graph features disabled", _e)
            return {}

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
        max_bytes = int(os.getenv("DISTILLER_CHUNK_MAX_BYTES", str(30 * 1024)))
        res = silica_payload(recon_path, max_concepts=max_concepts, max_bytes=max_bytes)
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
        caps = ("collision", "distill", "sanitize", "validate", "snapshot", "write", "hub_update", "autolink", "backlink", "lint", "cleanup")
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

        # Build vault graph context (cluster/hub/pagerank) once per run.
        # Stored in context["vault_graph_ctx"] and consumed by COLLISION,
        # DELEGATE (distiller enrichment), AUTOLINK, and HUB_UPDATE.
        self.context["vault_graph_ctx"] = self._build_vault_graph_ctx()

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

        # Embed every concept in the chunk in a SINGLE call (one network
        # round-trip per chunk instead of one per concept).  Falls back to
        # per-concept embedding only if the embedder returns a ragged response,
        # so a short/odd reply can never silently drop concepts.
        all_texts: list[str] = []
        for batch in chunk.get("batches", []):
            for concept in batch.get("concepts", []):
                ct = concept.get("name", "") if isinstance(concept, dict) else str(concept)
                if ct:
                    all_texts.append(ct)

        vec_by_text: dict[str, Any] = {}
        uniq_texts = list(dict.fromkeys(all_texts))
        if uniq_texts:
            try:
                embedded = embedder.embed(uniq_texts)
                batched_ok = len(embedded) == len(uniq_texts)
            except Exception as _embed_err:
                logger.debug("COLLISION: batch embed failed (%s) — keeping concepts unrouted", _embed_err)
                embedded, batched_ok = [], False
            if batched_ok:
                vec_by_text = dict(zip(uniq_texts, embedded))
            else:
                for _t in uniq_texts:
                    try:
                        _ev = embedder.embed([_t])
                        vec_by_text[_t] = _ev[0]
                    except Exception as _embed_err:
                        logger.debug("COLLISION: embed failed for '%s': %s", _t, _embed_err)

        for batch in chunk.get("batches", []):
            inbox_file = batch.get("inbox_file", self.inbox_file)
            kept: list = []

            for concept in batch.get("concepts", []):
                concept_text = concept.get("name", "") if isinstance(concept, dict) else str(concept)
                if not concept_text:
                    kept.append(concept)
                    continue

                vec = vec_by_text.get(concept_text)
                if vec is None:
                    # Embedding unavailable for this concept (batch failed or
                    # missing) — keep it for normal distillation.
                    kept.append(concept)
                    continue
                try:
                    results = store.cosine_top_k(vec, k=1)
                except Exception as _search_err:
                    logger.debug("COLLISION: search failed for '%s': %s", concept_text, _search_err)
                    kept.append(concept)
                    continue

                if not results:
                    kept.append(concept)
                    continue

                top = results[0]
                score: float = top.get("score", 0.0)
                existing_path = top.get("path", "")

                # Lower effective threshold for cluster hubs: merging into an
                # anchor note is safer than creating a competing shadow note.
                _vault_ctx = self.context.get("vault_graph_ctx", {})
                _match_key = existing_path.removesuffix(".md")
                _is_hub = _vault_ctx.get(_match_key, {}).get("is_hub", False)
                τ_eff = τ_high - (0.08 if _is_hub else 0.0)

                if score >= τ_eff:
                    try:
                        DRIVER.read_note(existing_path)
                        # Graph confirms node exists — safe to patch
                        logger.info(
                            "COLLISION: '%s' → patch '%s' (score=%.3f ≥ τ_eff=%.2f%s)",
                            concept_text, existing_path, score, τ_eff,
                            " [hub]" if _is_hub else "",
                        )
                        pre_routed_ops.append({
                            "op": "patch",
                            "path": existing_path,
                            "heading": concept_text,
                            "source_basename": os.path.basename(inbox_file),
                            "snippet": concept.get("excerpt", "") if isinstance(concept, dict) else "",
                            "hub": self.hub,
                            "reason": f"collision_routed score={score:.3f}{' [hub]' if _is_hub else ''}",
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
            self._defer_ops(
                deferred_op_dicts,
                {
                    (d["concept"].get("name", str(i)) if isinstance(d["concept"], dict) else str(i)):
                    f"borderline_similarity score={d['score']:.3f}"
                    for i, d in enumerate(deferred_concepts)
                },
                phase="COLLISION",
            )

        # Producer: hand each borderline pair to the leashed dedup sub-agent so it
        # can run concurrently while the Injector keeps writing its other batches.
        # The candidate match is a pre-existing (committed) vault note, so the
        # sub-agent's append-only patch never races the Injector's new-note writes;
        # the per-path lease covers the rare same-note overlap.
        if deferred_concepts and self.work_queue is not None:
            from silica.planner.workqueue import WorkItem
            for d in deferred_concepts:
                concept = d["concept"]
                match = d.get("top_match", {})
                candidate_path = match.get("path", "")
                if not candidate_path:
                    continue
                name = concept.get("name", "") if isinstance(concept, dict) else str(concept)
                excerpt = concept.get("excerpt", "") if isinstance(concept, dict) else ""
                try:
                    self.work_queue.enqueue(WorkItem(
                        kind="dedup",
                        target_path=candidate_path,
                        context={
                            "concept": name,
                            "excerpt": excerpt,
                            "candidate": match.get("name", candidate_path),
                            "score": d.get("score"),
                            "inbox_file": d.get("inbox_file", self.inbox_file),
                            "hub": self.hub,
                        },
                        reason=f"borderline_similarity score={d.get('score', 0):.3f}",
                    ))
                except Exception as _qe:
                    logger.debug("COLLISION: failed to enqueue dedup item: %s", _qe)

        # Store pre-routed ops for merging in VALIDATE (Phase 5)
        self.context[f"chunk_{idx}_collision_ops"] = pre_routed_ops

        # Capture the idempotency hash BEFORE mutating the chunk.
        # COLLISION re-routes concepts based on what is currently in the vault,
        # which changes between a partial run and its resume (done chunks have
        # already written their notes).  Hashing the pre-COLLISION chunk means
        # the key is stable across runs with the same source input.
        import json as _json
        self.context[f"chunk_{idx}_input_hash"] = hashlib.sha256(
            _json.dumps(chunk, sort_keys=True).encode()
        ).hexdigest()

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
        #
        # Use the pre-COLLISION hash stored by _handle_collision so the key is
        # based on the original source input, not the vault-state-dependent
        # post-COLLISION chunk (which changes when resumed after a partial run).
        import json as _json
        chunk_hash = self.context.get(f"chunk_{idx}_input_hash") or hashlib.sha256(
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
        # Include the RunManifest so the distiller knows what was injected in prior chunks.
        ledger_digest: str | None = None
        try:
            ledger_digest = self.progress.digest(manifest=self.manifest)
        except Exception:
            pass

        # Phase 6: pass steering correction if VALIDATE sent us back here
        steer_context: str | None = self.context.get(f"chunk_{idx}_steer_context")
        if steer_context:
            logger.info("DELEGATE chunk %d: re-attempt with steering correction", idx)

        # Enrich the payload with graph context (cluster/hub/is_hub) for concepts
        # that have a vault_collision.  The distiller uses this to understand
        # structural importance of the matched note.  Original chunk is not modified.
        vault_ctx = self.context.get("vault_graph_ctx", {})
        enriched_chunk = _inject_graph_ctx(current_chunk, vault_ctx) if vault_ctx else current_chunk

        # Build per-chunk substrate: semantically close vault notes that are not
        # yet directly linked to run notes — candidates for `parent` and wikilinks.
        # Also surface any cleared parent forward-references from earlier chunks.
        substrate: str | None = None
        try:
            from silica.kernel.run_substrate import build_substrate
            substrate = build_substrate(
                enriched_chunk,
                manifest_titles=self.manifest.titles(),
                cleared_parents=self.context.get("run_cleared_parents"),
            )
        except Exception as _sub_e:
            logger.debug("DELEGATE: substrate build failed (non-fatal): %s", _sub_e)

        try:
            chunk_result = run_distiller(
                payload=enriched_chunk,
                target=self.target_dir,
                hub=self.hub,
                ledger_digest=ledger_digest,
                steer_context=steer_context,
                substrate=substrate,
            )
            if "error" in chunk_result:
                self._progress_note(self._chunk_task_id("distill"), "distill", "failed", error=chunk_result["error"])
                raise RuntimeError(f"Distiller error on batch {idx}: {chunk_result['error']}")

            distiller_path = self._make_tmp(chunk_result)
            self._chunk_ctx["distiller_output_path"] = distiller_path
            # Store chunk hash for knowledge-block write at VALIDATE
            self.context[f"chunk_{idx}_hash"] = chunk_hash
            self._progress_note(self._chunk_task_id("distill"), "distill", "done", output_ref=distiller_path)
            self._transition_success()

        except Exception as e:
            raise RuntimeError(f"Critical failure delegating batch {idx}: {e}")

    def _handle_sanitize(self) -> None:
        idx = self._current_chunk_idx
        self._progress_note(self._chunk_task_id("sanitize"), "sanitize", "running")
        res = silica_sanitize(self._chunk_ctx["distiller_output_path"])
        if "error" in res:
            self._progress_note(self._chunk_task_id("sanitize"), "sanitize", "failed", error=res["error"])
            raise RuntimeError(f"Sanitize failed: {res['error']}")
        self._chunk_ctx["sanitized"] = res
        self._progress_note(self._chunk_task_id("sanitize"), "sanitize", "done")
        self._transition_success()

    def _handle_validate(self) -> None:
        idx = self._current_chunk_idx
        self._progress_note(self._chunk_task_id("validate"), "validate", "running")
        sanitized = self._chunk_ctx["sanitized"]["parsed"]
        ops_raw = sanitized.get("updates", sanitized) if isinstance(sanitized, dict) else sanitized
        if not isinstance(ops_raw, list):
            ops_raw = [ops_raw]

        # Merge collision-routed patch ops (Phase 5): prepend so they go through
        # the same validate→snapshot→write path as distiller-generated ops.
        collision_ops = self.context.get(f"chunk_{idx}_collision_ops", [])
        if collision_ops:
            ops_raw = list(collision_ops) + list(ops_raw)

        # Cohesion pass: inject sibling cross-references into write ops' related[]
        # before validation so the links land in the written frontmatter.
        # Scope: same-chunk siblings only (cross-chunk handled by AUTOLINK/BACKLINK).
        from silica.kernel.cohesion import cohesion_pass
        ops_raw = cohesion_pass(ops_raw)

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
            self._chunk_ctx["ops_path"] = ops_path
            self._progress_note(self._chunk_task_id("validate"), "validate", "done")
            self.state = InjectorState.CLEANUP
            return

        # Persist rejected ops to the deferred store so the model can retry them
        # later without re-running the expensive RECON → DELEGATE cycle.
        rejected_raw = res.get("rejected_ops", [])
        if rejected_raw:
            deferred_ops = [
                r.get("op", r) if isinstance(r, dict) and "op" in r else r
                for r in rejected_raw
            ]
            rejection_reasons = {
                (r.get("op", {}).get("path") or r.get("op", {}).get("heading") or "?"): r.get("reason", "")
                for r in rejected_raw if isinstance(r, dict)
            }
            if self._defer_ops(deferred_ops, rejection_reasons, phase="VALIDATE"):
                logger.warning(
                    "VALIDATE: %d op(s) rejected and saved to deferred store (hash=%s…). "
                    "Use silica_deferred_retry to attempt them later.",
                    len(rejected_raw),
                    self._current_content_hash[:8],
                )

        # Accumulate cleared parent references across chunks.
        # These are prospective links (parent notes not yet in vault) that the
        # distiller can anticipate in subsequent chunks or future runs.
        cleared = res.get("cleared_parents", [])
        if cleared:
            self.context.setdefault("run_cleared_parents", []).extend(cleared)
            logger.debug("VALIDATE: %d parent reference(s) cleared to hub fallback (tracked as forward refs)", len(cleared))

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
                        self._chunk_task_id("distill"), "running",
                        error=f"steer attempt {steer_attempts}"
                    )
                except Exception:
                    pass
                self.state = InjectorState.DELEGATE
                return
            # Exhausted steering budget → defer and short-circuit.
            logger.warning("VALIDATE: steer budget exhausted (%d/%d) — deferring chunk %d.", steer_attempts, _max_steer, idx)
            self._chunk_ctx["abort_reason"] = "All ops rejected — nothing to write"
            self.context["final_status"] = "no_ops"
            self._chunk_ctx["ops_path"] = ops_path
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

        self._chunk_ctx["ops_path"] = kb_path
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
        res = silica_snapshot(self._chunk_ctx["ops_path"])
        if "error" in res:
            raise RuntimeError(f"SNAPSHOT failed: {res['error']}")

        self._chunk_ctx["snapshot"] = res
        self._chunk_ctx["txn_id"] = res["txn_id"]
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
            ops = load_ops(self._chunk_ctx["ops_path"])
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
            self._chunk_ctx["snapshot_domain"] = [{"name": r.name, "path": r.path} for r in snapshot_domain_list]
            self._pre_graph = DRIVER.graph_snapshot(snapshot_domain_list)
        except Exception as e:
            logger.error("Failed to take pre-write graph snapshot: %s", e)
            raise RuntimeError(f"Pre-write graph snapshot failed: {e}")

        self._progress_note(self._chunk_task_id("snapshot"), "snapshot", "done")
        self._transition_success()

    def _handle_write(self) -> None:
        from silica.kernel.atomic_write import bulk_write_atomic
        self._progress_note(self._chunk_task_id("write"), "write", "running")

        ops = load_ops(self._chunk_ctx["ops_path"])
        result = bulk_write_atomic(ops, hub=self.hub, lint=True)

        # Accumulate surviving notes' inverses for the undo journal (recorded at
        # CLEANUP after autolink finalises content → correct version-guard hashes).
        for r in result.committed:
            for inv in r.inverses:
                self._run_inverses.append((r.path, inv, None))

        if result.failed:
            try:
                _deferred = [
                    next(o for o in ops if o.touched_ref() == r.path).model_dump()
                    for r in result.failed
                ]
                _errors = {r.path: (r.error or "lint/write failed") for r in result.failed}
                self._defer_ops(_deferred, _errors, phase="WRITE")
                logger.warning(
                    "WRITE: %d op(s) failed lint/write — deferred. "
                    "Continuing with %d committed op(s).",
                    len(result.failed), len(result.committed),
                )
                # Mark failed/deferred ops as skip so subsequent phases (lint, autolink, backlink, cleanup, etc.) skip them
                failed_paths = {r.path for r in result.failed}
                for op in ops:
                    if op.touched_ref() in failed_paths:
                        op.op = OpType.skip
                        op.reason = "Deferred because write/lint failed"
                from silica.kernel.ops_io import dump_ops
                dump_ops(self._chunk_ctx["ops_path"], ops)
            except Exception as _de:
                logger.debug("WRITE: deferred save/update failed (non-fatal): %s", _de)
            self.context["has_partial_failure"] = True

        if not result.committed and result.failed:
            self._progress_note(self._chunk_task_id("write"), "write", "failed",
                                error=f"all {len(result.failed)} ops failed lint/write")

        # Synthesise the legacy `write` result shape that downstream code reads.
        self.context["write"] = {
            "success": True,
            "successful": len(result.committed),
            "failed": [{"path": r.path, "error": r.error} for r in result.failed],
            "total": result.total,
        }

        committed_paths = {r.path for r in result.committed}
        _deferred_stems: frozenset[str] = frozenset(
            os.path.splitext(os.path.basename(r.path))[0].lower()
            for r in result.failed if r.path
        )
        self._chunk_ctx["deferred_stems"] = list(_deferred_stems)

        # Register committed notes in the RunManifest and refresh embedding index.
        try:
            from silica.planner.progress import RunManifestEntry
            vault_ctx = self.context.get("vault_graph_ctx", {})
            for op in ops:
                path = op.touched_ref()
                if op.op not in (OpType.write, OpType.patch) or not path:
                    continue
                if path not in committed_paths:
                    continue
                stem = os.path.splitext(os.path.basename(path))[0]
                path_key = path.removesuffix(".md")
                cluster_id = vault_ctx.get(path_key, {}).get("cluster_id", -1)
                self.manifest.record(RunManifestEntry(
                    title=stem,
                    path=path_key,
                    parent=op.parent,
                    cluster_id=cluster_id,
                    source_basename=op.source_basename or "",
                    op=op.op.value,
                ))
            self.manifest.save()

            # Best-effort incremental embed index refresh
            try:
                from silica.agent.providers import get_embedder
                from silica.kernel.embed import EmbedStore, refresh_note
                embedder = get_embedder(CONFIG)
                store = EmbedStore()
                for op in ops:
                    path = op.touched_ref()
                    if op.op not in (OpType.write, OpType.patch) or not path:
                        continue
                    if path not in committed_paths:
                        continue
                    stem = os.path.splitext(os.path.basename(path))[0]
                    idx_path = path.removesuffix(".md")
                    try:
                        body = DRIVER.read_note(path).content or ""
                        refresh_note(embedder, idx_path, stem, body, store=store)
                    except Exception as _re:
                        logger.debug("WRITE: embed refresh failed for '%s': %s", path, _re)
            except Exception as _ee:
                logger.debug("WRITE: embed refresh skipped (%s)", _ee)

            # Best-effort incremental co-occurrence refresh — the STABLE leg.
            # Separate from the embed block above so it stays fresh even when the
            # embedder is down (no network, pure local compute).
            _refresh_cooccurrence_for_ops(
                ops,
                committed_paths,
                read_body=lambda p: DRIVER.read_note(p).content or "",
                lang=CONFIG.cooccurrence_lang,
            )
        except Exception as _me:
            logger.debug("WRITE: manifest update failed (non-fatal): %s", _me)

        if not result.committed and result.failed:
            pass
        else:
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
            ops = load_ops(self._chunk_ctx["ops_path"])
        except Exception as e:
            raise RuntimeError(f"HUB_UPDATE: failed to read ops: {e}")

        hub_name = self.hub.strip("[]")
        hub_name_lower = hub_name.lower()

        # Collect write ops grouped by effective parent:
        # notes with op.parent set go to that parent note; others fall back to hub.
        hub_notes: list[tuple[str, str]] = []       # (note_name, desc)
        parent_notes: dict[str, list[tuple[str, str]]] = {}  # parent_name → [(note_name, desc)]
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
            effective_parent = (op.parent.strip("[]") if op.parent else None) or hub_name
            if effective_parent.lower() == hub_name_lower:
                hub_notes.append((note_name, desc))
            else:
                parent_notes.setdefault(effective_parent, []).append((note_name, desc))

        # Flatten for backward-compat references below
        new_notes = hub_notes

        if not new_notes and not parent_notes:
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

        # Cross-cluster integrity check: warn when new notes land in a different
        # cluster from the hub.  This is informational only — the MOC link is
        # still written, but the log helps identify structural drift.
        _gctx = self.context.get("vault_graph_ctx", {})
        _hub_key = hub_path.removesuffix(".md")
        _hub_cluster = _gctx.get(_hub_key, {}).get("cluster_id", -1)
        if _gctx and _hub_cluster >= 0:
            for note_name, _ in new_notes:
                _note_key = f"{self.target_dir}/{note_name}".replace("//", "/")
                _note_cluster = _gctx.get(_note_key, {}).get("cluster_id", -1)
                if _note_cluster >= 0 and _note_cluster != _hub_cluster:
                    logger.warning(
                        "HUB_UPDATE: '%s' (cluster %d) linked to hub '%s' (cluster %d) — cross-cluster MOC",
                        note_name, _note_cluster, hub_name, _hub_cluster,
                    )

        # Derive the actual source file for this chunk (self.inbox_file is always
        # the first file and never updates in multi-file runs — use the flat index map).
        _fi, _ci = self._chunk_flat_to_fi_ci.get(self._current_chunk_idx, (0, 0))
        _source_file = (
            self._file_chunks[_fi]["source_file"]
            if self._file_chunks and _fi < len(self._file_chunks)
            else self.inbox_file
        )
        source_name = os.path.splitext(os.path.basename(_source_file))[0]

        # Language-aware heading: "## Da: {name}" (Italian) or "## From: {name}" (English).
        # Sample the hub content + first snippet to detect language.
        _lang_sample = hub_note.content + " ".join(d for _, d in new_notes[:3])
        moc_heading = _moc_heading(source_name, _lang_sample)

        # Build note link lines.
        note_lines = [
            f"- [[{n}]] — {d}" if d else f"- [[{n}]]"
            for n, d in new_notes
        ]

        # Merge: append to existing section if present (same file, multiple chunks),
        # otherwise create a new section.  Use overwrite to avoid the settle race.
        new_hub_content = _merge_moc_section(hub_note.content, moc_heading, note_lines)

        try:
            DRIVER.overwrite(hub_path, new_hub_content)
            # Explicitly wait until the section header is readable.
            _deadline = time.monotonic() + 5.0
            while time.monotonic() < _deadline:
                try:
                    if moc_heading in DRIVER.read_note(hub_ref).content:
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
        existing_paths = {d["path"] for d in self._chunk_ctx.get("snapshot_domain", [])}
        if hub_path not in existing_paths:
            self._chunk_ctx.setdefault("snapshot_domain", []).append({"name": hub_name, "path": hub_path})

        # Write MOC sections to specific parent notes (best-effort — only active when
        # the distiller emits op.parent, which requires the Block 4 prompt update).
        if parent_notes:
            from silica.kernel.ops import InverseOp, InverseOpKind
            for parent_name, p_new_notes in parent_notes.items():
                parent_path = f"{self.target_dir}/{parent_name}.md".replace("//", "/")
                try:
                    from silica.driver.base import NoteRef as _NR
                    p_ref = _NR(name=parent_name, path=parent_path)
                    p_note = DRIVER.read_note(p_ref)
                    # Register rollback inverse for parent note
                    if self._txn is not None:
                        p_inverse = InverseOp(
                            kind=InverseOpKind.restore_version,
                            path=parent_path,
                            prior_content=p_note.content,
                        )
                        self._txn.inverses.append(p_inverse)
                    # Build and write parent MOC block (same language-aware heading,
                    # same deduplication logic as the hub section above).
                    p_heading = _moc_heading(source_name, p_note.content)
                    p_note_lines = [
                        f"- [[{n}]] — {d}" if d else f"- [[{n}]]"
                        for n, d in p_new_notes
                    ]
                    new_p_content = _merge_moc_section(p_note.content, p_heading, p_note_lines)
                    DRIVER.overwrite(parent_path, new_p_content)
                    existing_paths = {d["path"] for d in self._chunk_ctx.get("snapshot_domain", [])}
                    if parent_path not in existing_paths:
                        self._chunk_ctx.setdefault("snapshot_domain", []).append(
                            {"name": parent_name, "path": parent_path}
                        )
                    logger.info(
                        "HUB_UPDATE: updated parent '%s' with %d sub-spoke link(s)",
                        parent_path, len(p_new_notes),
                    )
                except Exception as _pe:
                    logger.warning("HUB_UPDATE: failed to update parent '%s': %s", parent_path, _pe)

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
            from silica.kernel.autolink import build_title_index

            ops = load_ops(self._chunk_ctx["ops_path"])
            touched_paths = [
                ref
                for op in ops
                if (ref := op.touched_ref()) and op.op not in (OpType.delete, OpType.skip)
            ]

            if not touched_paths:
                self._progress_note(self._chunk_task_id("autolink"), "autolink", "done")
                self._transition_success()
                return

            all_refs = DRIVER.list_files()
            title_index = build_title_index(all_refs)

            # Build a reverse map: title (basename, no .md) → cluster_id for fast lookup
            vault_ctx = self.context.get("vault_graph_ctx", {})
            _title_to_cluster: dict[str, int] = {
                k.rsplit("/", 1)[-1]: v["cluster_id"]
                for k, v in vault_ctx.items()
                if v.get("cluster_id", -1) >= 0
            }

            total_added = 0
            for path in touched_paths:
                try:
                    note_title = os.path.splitext(os.path.basename(path))[0]
                    note_cluster = _title_to_cluster.get(note_title, -1)

                    # Narrow candidates to the same cluster when cluster data is available.
                    # This prevents cross-cluster noise links (e.g. Economics ↔ Physics).
                    if vault_ctx and note_cluster >= 0:
                        candidates = [
                            t for t in title_index
                            if _title_to_cluster.get(t, -1) == note_cluster and t != note_title
                        ]
                    else:
                        candidates = None

                    added = DRIVER.autolink_note(
                        path, candidates=candidates if candidates is not None else title_index
                    )
                    if added:
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

    def _handle_backlink(self) -> None:
        """Best-effort reverse link injection into pre-existing neighbouring notes (Phase 4.5).

        For each newly-written note (write ops, excluding the hub auto-creation),
        scans pre-existing notes that textually mention the new title and wraps
        those mentions as wikilinks.  Extends snapshot_domain and registers
        rollback inverses for any modified note so ROLLBACK and LINT graph-diff
        both cover the backlinks.
        """
        idx = self._current_chunk_idx
        self._progress_note(self._chunk_task_id("backlink"), "backlink", "running")

        try:
            from silica.kernel.autolink import backlink_pass, build_title_index

            ops = load_ops(self._chunk_ctx["ops_path"])

            hub_name_lower = (self.hub or "").strip("[]").lower()
            new_titles: list[str] = []
            for op in ops:
                if op.op != OpType.write:
                    continue
                path = op.touched_ref()
                if not path:
                    continue
                stem = os.path.splitext(os.path.basename(path))[0]
                if stem.lower() != hub_name_lower:
                    new_titles.append(stem)

            if not new_titles:
                self._progress_note(self._chunk_task_id("backlink"), "backlink", "done")
                self._transition_success()
                return

            touched_paths_abs: set[str] = {
                os.path.abspath(p)
                for op in ops
                for p in (op.touched_ref(),)
                if p is not None
            }

            neighbourhood: list[str] = []
            seen_norm: set[str] = set()

            # Use the O(1) inverted text index if available; fall back to search_context.
            if hasattr(DRIVER, "mentions_of"):
                for title in new_titles:
                    try:
                        for path in DRIVER.mentions_of(title):
                            norm = os.path.abspath(path)
                            if norm not in seen_norm and norm not in touched_paths_abs:
                                seen_norm.add(norm)
                                neighbourhood.append(path)
                    except Exception as _me:
                        logger.debug("BACKLINK: mentions_of for '%s' failed: %s", title, _me)
            else:
                for title in new_titles:
                    try:
                        for hit in DRIVER.search_context(title):
                            p = hit.ref.path or hit.ref.name
                            norm = os.path.abspath(p)
                            if norm not in seen_norm and norm not in touched_paths_abs:
                                seen_norm.add(norm)
                                neighbourhood.append(p)
                    except Exception as _se:
                        logger.debug("BACKLINK: search_context for '%s': %s", title, _se)


            if not neighbourhood:
                self._progress_note(self._chunk_task_id("backlink"), "backlink", "done")
                self._transition_success()
                return

            # Pre-read prior content before backlink_pass writes, for rollback inverses
            prior_contents: dict[str, str] = {}
            for path in neighbourhood:
                try:
                    prior_contents[path] = DRIVER.read_note(path).content or ""
                except Exception:
                    pass

            all_refs = DRIVER.list_files()
            title_index = build_title_index(all_refs)
            added_map = backlink_pass(new_titles, title_index=title_index, neighbourhood=neighbourhood)

            if added_map and self._txn is not None:
                from silica.kernel.ops import InverseOp, InverseOpKind
                existing_snapshot_paths = {d["path"] for d in self._chunk_ctx.get("snapshot_domain", [])}
                for path_modified in added_map:
                    if path_modified not in existing_snapshot_paths:
                        stem = os.path.splitext(os.path.basename(path_modified))[0]
                        self._chunk_ctx.setdefault("snapshot_domain", []).append(
                            {"name": stem, "path": path_modified}
                        )
                        existing_snapshot_paths.add(path_modified)
                    if path_modified in prior_contents:
                        inverse = InverseOp(
                            kind=InverseOpKind.restore_version,
                            path=path_modified,
                            prior_content=prior_contents[path_modified],
                        )
                        self._txn.inverses.append(inverse)

            total_links = sum(len(v) for v in added_map.values())
            logger.info(
                "BACKLINK: %d link(s) added to %d pre-existing note(s)", total_links, len(added_map)
            )
        except Exception as e:
            logger.warning("BACKLINK: phase failed (non-fatal): %s", e)

        self._progress_note(self._chunk_task_id("backlink"), "backlink", "done")
        self._transition_success()

    def _handle_lint(self) -> None:
        idx = self._current_chunk_idx
        self._progress_note(self._chunk_task_id("lint"), "lint", "running")
        try:
            ops = load_ops(self._chunk_ctx["ops_path"])
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
                self._chunk_ctx["abort_reason"] = f"Lint failed for {path}: {res['errors']}"
                self._progress_note(self._chunk_task_id("lint"), "lint", "failed", error=self._chunk_ctx["abort_reason"])
                self.state = InjectorState.ROLLBACK
                return

        # S3.2: Run graph-diff check
        regression_rule = self._get_recipe_gate("graph_regression", "forbid_new_orphans")
        if regression_rule != "allow":
            if self._pre_graph is None:
                self._chunk_ctx["abort_reason"] = "Graph regression gate failed: pre-write snapshot is missing"
                self._progress_note(self._chunk_task_id("lint"), "lint", "failed", error=self._chunk_ctx["abort_reason"])
                self.state = InjectorState.ROLLBACK
                return
            try:
                from silica.driver.base import NoteRef
                snapshot_domain_dicts = self._chunk_ctx.get("snapshot_domain", [])
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
                deferred_stems = set(self._chunk_ctx.get("deferred_stems", []))
                try:
                    from silica.kernel.templates import slugify
                    for chunk in getattr(self, "_chunks", []):
                        for batch in chunk.get("batches", []):
                            for concept in batch.get("concepts", []):
                                name = concept.get("name")
                                if name:
                                    stem = os.path.splitext(os.path.basename(name))[0].lower()
                                    deferred_stems.add(stem)
                                    deferred_stems.add(slugify(stem))
                except Exception as _ce:
                    logger.debug("Failed to extract run concept stems for graph check: %s", _ce)

                success, errors = check_graph_regression(
                    self._pre_graph, post_graph, created_paths, frozenset(deferred_stems)
                )

                if CONFIG.verbose:
                    logger.info(
                        "[DEBUG Graph Regression Gate]: Pre-write graph size: %d nodes | Post-write graph size: %d nodes | Rule: %s | Result: %s",
                        len(self._pre_graph.link_counts) if self._pre_graph and self._pre_graph.link_counts else 0,
                        len(post_graph.link_counts) if post_graph and post_graph.link_counts else 0,
                        regression_rule,
                        "PASSED" if success else f"FAILED: {errors}"
                    )

                if not success:
                    orphan_errors = [e for e in errors if e.startswith("Unplanned orphans")]
                    blocking_errors = [e for e in errors if not e.startswith("Unplanned orphans")]
                    if orphan_errors:
                        logger.warning(
                            "[Graph Regression Gate]: Orphan warning (non-blocking): %s",
                            "; ".join(orphan_errors),
                        )
                        # Record run-created notes that ended this chunk orphaned.
                        # Acted on (if still orphaned) at end of run, not now —
                        # AUTOLINK/BACKLINK or a later chunk may yet connect them.
                        if self.warning_ledger is not None:
                            try:
                                from silica.kernel.graph_diff import normalize_ref
                                post_orphan_keys = {normalize_ref(r) for r in post_graph.orphans}
                                detail = "; ".join(orphan_errors)
                                for op in ops:
                                    p = op.touched_ref()
                                    if not p or op.op != OpType.write:
                                        continue
                                    name = os.path.splitext(os.path.basename(p))[0]
                                    if normalize_ref(NoteRef(name=name, path=p)) in post_orphan_keys:
                                        self.warning_ledger.add(p, "orphan", detail)
                            except Exception as _we:
                                logger.debug("orphan warning record failed (non-fatal): %s", _we)
                    if blocking_errors:
                        reason = f"Graph regression gate failed: {'; '.join(blocking_errors)}"
                        logger.warning("[Graph Regression Gate]: Blocking errors (triggering rollback): %s", "; ".join(blocking_errors))
                        self._chunk_ctx["abort_reason"] = reason
                        self._progress_note(self._chunk_task_id("lint"), "lint", "failed", error=reason)
                        self.state = InjectorState.ROLLBACK
                        return
            except Exception as e:
                logger.error("Failed to perform graph-diff check: %s", e)
                self._chunk_ctx["abort_reason"] = f"Graph regression gate error during check: {e}"
                self._progress_note(self._chunk_task_id("lint"), "lint", "failed", error=self._chunk_ctx["abort_reason"])
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

        # Persist this run's inverses for /revert, with final content hash.
        if self._undo_run_id and self._run_inverses:
            import hashlib
            from silica.kernel.undo_journal import get_undo_journal
            journal = get_undo_journal()
            for path, inv, _ in self._run_inverses:
                try:
                    post = DRIVER.read_note(path).content
                    post_hash = hashlib.sha256((post or "").encode("utf-8")).hexdigest()
                except Exception:
                    post_hash = None
                journal.record(self._undo_run_id, inv, post_hash)
            self._run_inverses.clear()

        self._progress_note(self._chunk_task_id("cleanup"), "cleanup", "done")
        self._transition_success()

    def _handle_rollback(self) -> None:
        self._progress_note("rollback", "rollback", "running")
        snapshot_res = self._chunk_ctx.get("snapshot", {})
        # self._txn.inverses is the single source of truth for rollback (C3 /
        # ADR-009): SNAPSHOT seeds it and every phase that mutates a pre-existing
        # note appends to it. Fall back to the persisted snapshot dict only when
        # no live transaction exists (defensive — both share the per-chunk lifetime).
        if self._txn is not None:
            txn_id = self._txn.id
            inverses = self._txn.inverses_serialized
        else:
            txn_id = snapshot_res.get("txn_id")
            inverses = snapshot_res.get("inverses", [])

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

        # Clean up the embedding index for notes that were created and then rolled back
        # to prevent stale phantom entries that would bias future candidate searches.
        created_paths: list[str] = []
        if self._txn is not None and self._txn.created_paths:
            created_paths = list(self._txn.created_paths)
        elif snapshot_res.get("created_paths"):
            created_paths = list(snapshot_res["created_paths"])
        if created_paths:
            try:
                from silica.kernel.embed import EmbedStore
                store = EmbedStore()
                for cp in created_paths:
                    store.delete(cp.removesuffix(".md"))
                store.save()
            except Exception as _ee:
                logger.debug("ROLLBACK: embed index cleanup failed (non-fatal): %s", _ee)

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
        # Read abort_reason before clearing the chunk namespace
        abort_reason = self._chunk_ctx.get("abort_reason", "chunk failure")

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

        # Clear the per-chunk namespace atomically (prevents state leakage to next chunk).
        # idx-keyed context keys (chunk_{idx}_*) are already safe — each chunk uses
        # its own idx — so only the chunk namespace dict needs explicit teardown.
        self.context.pop("chunk", None)
        self._txn = None
        self._pre_graph = None

        # Record that at least one chunk failed (used by cleanup to set "partial")
        self.context["has_partial_failure"] = True

        # Advance to next uncommitted chunk, or conclude the run as partial
        self._get_chunks_from_context_if_empty()
        next_idx = self._next_uncommitted_chunk_idx(self._current_chunk_idx + 1)
        if next_idx < len(self._chunks):
            self._current_chunk_idx = next_idx
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
                "Chunk f%d_c%d failed (last uncommitted chunk). Run concludes with partial success.", fi, ci
            )
            self.context["final_status"] = "partial"
            self.state = InjectorState.DONE

    def _next_uncommitted_chunk_idx(self, start: int) -> int:
        """Return the first chunk index >= start whose file is not already committed."""
        idx = start
        committed = getattr(self, "_committed_file_indices", set())
        while idx < len(self._chunks):
            fi, _ = self._chunk_flat_to_fi_ci.get(idx, (0, 0))
            if fi not in committed:
                return idx
            logger.info("Skipping already-committed file %d chunk %d", fi, idx)
            idx += 1
        return idx

    @property
    def _current_source_file(self) -> str:
        """Vault-relative path of the inbox file for the current chunk."""
        fi, _ = self._chunk_flat_to_fi_ci.get(self._current_chunk_idx, (0, 0))
        if self._file_chunks and fi < len(self._file_chunks):
            return self._file_chunks[fi]["source_file"]
        return self.inbox_file

    @property
    def _current_content_hash(self) -> str:
        """Content hash for the inbox file of the current chunk."""
        fi, _ = self._chunk_flat_to_fi_ci.get(self._current_chunk_idx, (0, 0))
        if self._file_content_hashes and fi < len(self._file_content_hashes):
            return self._file_content_hashes[fi]
        return self.context.get("source_content_hash", "")

    # ------------------------------------------------------------------
    # Ledger helpers (C5)
    # ------------------------------------------------------------------

    def _write_ledger_for_file(self, fi: int, status: str) -> None:
        """Record this chunk's ops into the ledger, attributed to file fi."""
        try:
            from silica.kernel.ledger import get_ledger
            ledger = get_ledger()
            txn_id = self._chunk_ctx.get("txn_id", "unknown")

            # Use per-file canonical/hash when available; fall back to context
            if fi < len(self._file_canonicals):
                source_canonical = self._file_canonicals[fi]
                content_hash = self._file_content_hashes[fi] if fi < len(self._file_content_hashes) else None
            else:
                source_canonical = self.context.get("source_canonical", "")
                content_hash = self.context.get("source_content_hash")

            ops = load_ops(self._chunk_ctx["ops_path"])
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
