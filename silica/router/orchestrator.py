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
# Imported for the states modules (and tests), which resolve patchable
# collaborators through this module's namespace — see silica.router.states.
from silica.router import states

logger = logging.getLogger(__name__)


def _count_files_done(flat_map: dict[int, tuple[int, int]], upto_idx: int) -> int:
    """Number of inbox files whose every chunk's flat index is < upto_idx.

    Drives the TUI file-progress bar: a file is "done" once the FSM has advanced
    past its last chunk. Pass upto_idx=len(chunks) to mark all files done.
    """
    last_flat: dict[int, int] = {}
    for flat, (fi, _ci) in flat_map.items():
        last_flat[fi] = max(last_flat.get(fi, -1), flat)
    return sum(1 for last in last_flat.values() if last < upto_idx)


def _refresh_cooccurrence_for_ops(
    ops: list,
    committed_paths: set,
    *,
    read_body: Callable[[str], str],
    lang: str = "english",
    store: Any | None = None,
    save: bool = True,
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
                       concepts_by_path=concepts_by_path or None, save=save)
    except Exception as exc:
        logger.debug("WRITE: cooccur refresh skipped (%s)", exc)
        return 0
    return len(notes)


# Drift above this many notes is treated as a stale/cold index (not crash-drift):
# the reconcile warns and defers to an explicit /embed rather than embedding a
# large batch implicitly at run start.
_RECONCILE_CAP = 500


def _reconcile_embed_index(*, folder: str = "") -> int:
    """Repair embed-index drift from a prior hard crash (Fix A safety net).

    Set-diffs vault note paths against index keys and embeds ONLY the missing
    ones. Bounded best-effort: a cold/empty index is left to an explicit /embed
    (never implicitly embed the whole vault), and drift beyond ``_RECONCILE_CAP``
    is treated as a stale index — warn and skip. Also closes the pre-existing gap
    where a mid-run crash desynced the index until a manual /embed. Returns the
    number of notes re-embedded.
    """
    try:
        from silica.agent.providers import get_embedder
        from silica.kernel.embed import build_index, get_store
        from silica.kernel.media import preprocess_text

        store = get_store()
        if len(store) == 0:
            return 0  # cold index — an explicit /embed owns the full build
        have = set(store.paths())
        missing: list[tuple[str, str]] = []
        for ref in DRIVER.list_files(folder or ""):
            idx_path = (ref.path or ref.name).removesuffix(".md")
            if idx_path and idx_path not in have:
                missing.append((idx_path, ref.name or idx_path))
        if not missing:
            return 0
        if len(missing) > _RECONCILE_CAP:
            logger.warning(
                "embed index drift %d > cap %d — run /embed to rebuild; skipping reconcile",
                len(missing), _RECONCILE_CAP,
            )
            return 0
        embedder = get_embedder(CONFIG)
        notes: list[tuple[str, str, str]] = []
        for idx_path, name in missing:
            try:
                body = preprocess_text(DRIVER.read_note(idx_path + ".md").content or "")
            except Exception:
                continue
            notes.append((idx_path, name, body))
        if notes:
            build_index(embedder, notes, store=store)  # embeds the missing + saves
        return len(notes)
    except Exception as e:
        logger.debug("embed reconcile skipped (%s)", e)
        return 0


def _commit_docs_for_ops(
    ops: list,
    committed_paths: set,
    *,
    vault: str,
    git_commit: str,
) -> str | None:
    """Commit touched vault paths for write/patch ops to git.

    Git safety net behind SILICA_GIT_COMMIT=auto: an additive snapshot on top
    of the undo journal (ADR-0002), never a replacement. Best-effort: no git
    binary, vault outside a repo, nothing staged, or any subprocess failure all
    yield None and never raise. Commits ONLY vault paths — the out-of-vault
    guard lives inside gitstate.commit_docs, so a bug cannot commit source files.
    """
    from silica.kernel import gitstate
    from pathlib import Path as _Path

    if git_commit != "auto" or not vault:
        return None

    seen: set[str] = set()
    abs_paths: list[_Path] = []
    for op in ops:
        path = op.touched_ref()
        if op.op not in (OpType.write, OpType.patch) or not path:
            continue
        if path not in committed_paths or path in seen:
            continue
        seen.add(path)
        abs_paths.append(_Path(vault) / path)

    if not abs_paths:
        return None

    try:
        root = gitstate.find_repo_root(vault)
        if root is None:
            return None
        n = len(abs_paths)
        return gitstate.commit_docs(root, vault, abs_paths, f"silica: write {n} note(s)")
    except Exception as _ge:
        logger.debug("WRITE: git auto-commit skipped (%s)", _ge)
        return None


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

        # Run facade — TaskLedger (immutable plan, built from the recipe) +
        # ProgressLedger (mutable state) + RunManifest (short-term memory)
        # under one run_id; the resume fallback dance lives in Run.resume.
        from silica.kernel.progress import PlanStep, Run
        _checkpoints = [
            PlanStep(
                id=p["id"],
                kind=p.get("kind", "mechanical"),
                objective=p.get("tool", p.get("worker", p["id"])),
            )
            for p in self._recipe.get("phases", [])
        ]
        _run_kwargs: dict[str, Any] = dict(
            mode="inject",
            user_request=f"inject {', '.join(self.inbox_files)} → {target_dir}",
            checkpoints=_checkpoints,
            inputs={
                "inbox_files": self.inbox_files,
                "inbox_file": self.inbox_file,
                "target_dir": target_dir,
                "hub": hub or "",
            },
        )
        # NB: named _run because `run` would shadow the FSM's run() entry point
        run = Run.resume(resume_run_id, **_run_kwargs) if resume_run_id else Run.new(**_run_kwargs)
        self._run = run
        self.progress = run.progress
        self.manifest = run.manifest
        self.task_ledger = run.task_ledger
        if not run.resumed:
            self.progress.add_task("recon",   task_id="recon")
            self.progress.add_task("payload", task_id="payload", depends_on=["recon"])
            self.progress.save()

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
            from silica.ui.renderer import emit_pipeline_phase
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

        # Only open a journal run when the pipeline will actually execute writes.
        from silica.kernel.undo_journal import get_undo_journal
        self._undo_run_id = get_undo_journal().start_run(source=self.inbox_file)

        # Fix A: repair any embed-index drift left by a prior hard crash before
        # this run reads the index (no-op/sub-ms when the index is in sync).
        _reconcile_embed_index()

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
            self._flush_indexes()
        return self.context

    def _flush_indexes(self) -> None:
        """Persist the deferred embed + co-occurrence upserts once per run (Fix A).

        The write path upserts into the shared in-memory singletons with
        save=False and marks the index dirty; this single flush rewrites each
        dirty index file once instead of per note (1.17s/note at 10k). Gated on
        the dirty flags so a run that wrote nothing (or had the embedder down)
        never rewrites the index. Runs in the _run_loop finally so it fires on
        success, error, and Ctrl+C; a hard kill is repaired by the reconcile.
        """
        ctx = getattr(self, "context", {})
        if ctx.get("_embed_dirty"):
            try:
                from silica.kernel.embed import get_store
                get_store().save()
            except Exception as e:
                logger.debug("flush: embed index save skipped (%s)", e)
        if ctx.get("_cooccur_dirty"):
            try:
                from silica.kernel.cooccurrence import get_cooccur_store
                get_cooccur_store().save()
            except Exception as e:
                logger.debug("flush: cooccur index save skipped (%s)", e)

    def _on_sequence_end(self) -> None:
        self._eval_loop_or_done()

    def _on_cleanup_done(self) -> None:
        self._eval_loop_or_done()

    def _emit_files_progress(self, upto_idx: int) -> None:
        """Surface files-processed/total to the TUI bar (no-op without a renderer)."""
        try:
            from silica.ui.renderer import emit_run_progress
            total = len(self.inbox_files)
            done = _count_files_done(self._chunk_flat_to_fi_ci, upto_idx)
            emit_run_progress(min(done, total), total, label=self._current_source_file)
        except Exception:
            pass

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
            self._emit_files_progress(next_idx)
            logger.info(f"✔ Batch completed successfully. Advancing to batch {self._current_chunk_idx + 1}")
            # Restart per-chunk loop from COLLISION (Phase 5) if present, else DELEGATE
            has_collision = any(
                p.get("id") == "collision"
                for p in self._recipe.get("phases", [])
            )
            self.state = InjectorState.COLLISION if has_collision else InjectorState.DELEGATE
        else:
            self._emit_files_progress(len(self._chunks))
            logger.info("🎉 All batched chunks have been successfully injected and verified!")
            self.state = InjectorState.DONE

    # ------------------------------------------------------------------
    # State Handlers
    # ------------------------------------------------------------------

    def _handle_recon(self) -> None:
        states.setup.handle_recon(self)

    def _handle_crossdedup(self) -> None:
        states.setup.handle_crossdedup(self)

    def _handle_payload(self) -> None:
        states.setup.handle_payload(self)

    def _handle_salience(self) -> None:
        states.setup.handle_salience(self)

    def _handle_collision(self) -> None:
        states.collision.handle_collision(self)

    def _handle_delegate(self) -> None:
        states.distill.handle_delegate(self)

    def _handle_sanitize(self) -> None:
        states.distill.handle_sanitize(self)

    def _handle_validate(self) -> None:
        states.distill.handle_validate(self)

    def _handle_snapshot(self) -> None:
        states.write.handle_snapshot(self)

    def _handle_write(self) -> None:
        states.write.handle_write(self)

    def _handle_hub_update(self) -> None:
        states.write.handle_hub_update(self)

    def _handle_autolink(self) -> None:
        states.linking.handle_autolink(self)

    def _handle_backlink(self) -> None:
        states.linking.handle_backlink(self)

    def _handle_lint(self) -> None:
        states.finalize.handle_lint(self)

    def _handle_cleanup(self) -> None:
        states.finalize.handle_cleanup(self)

    def _handle_rollback(self) -> None:
        states.finalize.handle_rollback(self)

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
            self._emit_files_progress(next_idx)
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
            self._emit_files_progress(len(self._chunks))
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
