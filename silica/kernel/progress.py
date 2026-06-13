"""ProgressLedger — persistent run-state (working memory of a run).

Distinct from its sibling kernel/ledger.py (CommitLedger), which records
what was written to the vault.  This module records what a run intends to
do and how far along it is.

Two complementary objects:
  TaskLedger    — immutable run plan (written once at init, never mutated)
  ProgressLedger — mutable execution state (updated each transition)

Both serialised to ~/.silica/runs/<run_id>/ via orjson.
"""
from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import orjson

TaskStatus = Literal["pending", "running", "done", "failed", "skipped", "blocked", "deferred"]
PlanStepKind = Literal["mechanical", "semantic", "gate", "txn"]

_RUNS_DIR = Path.home() / ".silica" / "runs"


# ---------------------------------------------------------------------------
# PlanStep — one entry in the immutable plan
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """A single planned step from the recipe. Written once; never mutated."""
    id: str
    kind: str          # PlanStepKind — str for forward-compat with new kinds
    objective: str     # tool / worker name or free-form description


# ---------------------------------------------------------------------------
# TaskLedger — immutable plan for the full run
# ---------------------------------------------------------------------------

@dataclass
class TaskLedger:
    """Immutable run plan — created at FSM init, written once, never overwritten."""
    run_id: str
    user_request: str
    checkpoints: list[PlanStep]
    facts: dict[str, Any]
    created_at: str

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def new(
        cls,
        run_id: str,
        user_request: str,
        checkpoints: list[PlanStep],
        facts: dict[str, Any] | None = None,
    ) -> TaskLedger:
        return cls(
            run_id=run_id,
            user_request=user_request,
            checkpoints=checkpoints,
            facts=facts or {},
            created_at=_now(),
        )

    # ------------------------------------------------------------------
    # Persistence — write-once
    # ------------------------------------------------------------------

    def save(self) -> Path:
        """Write to disk only if the file does not yet exist (immutable semantics)."""
        run_dir = _RUNS_DIR / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "task_ledger.json"
        if path.exists():
            return path  # already written — do not overwrite
        path.write_bytes(
            orjson.dumps(dataclasses.asdict(self), option=orjson.OPT_INDENT_2)
        )
        return path

    @classmethod
    def load(cls, run_id: str) -> TaskLedger:
        path = _RUNS_DIR / run_id / "task_ledger.json"
        data = orjson.loads(path.read_bytes())
        return _task_ledger_from_dict(data)


# ---------------------------------------------------------------------------
# RunManifest — short-term memory of what was injected in this run
# ---------------------------------------------------------------------------

@dataclass
class RunManifestEntry:
    """One injected note recorded during this run."""
    title: str
    path: str               # vault-relative, without .md extension
    parent: str | None
    cluster_id: int
    source_basename: str
    op: str                 # "write" | "patch"


@dataclass
class RunManifest:
    """Short-term memory: tracks every note created or patched in a run.

    Serialised to ~/.silica/runs/<run_id>/manifest.json (orjson).
    """
    run_id: str
    entries: list[RunManifestEntry] = field(default_factory=list)

    def record(self, e: RunManifestEntry) -> None:
        self.entries.append(e)

    def titles(self) -> list[str]:
        return [e.title for e in self.entries]

    def save(self) -> Path:
        run_dir = _RUNS_DIR / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "manifest.json"
        path.write_bytes(orjson.dumps(dataclasses.asdict(self), option=orjson.OPT_INDENT_2))
        return path

    @classmethod
    def load(cls, run_id: str) -> "RunManifest":
        path = _RUNS_DIR / run_id / "manifest.json"
        data = orjson.loads(path.read_bytes())
        entries = [RunManifestEntry(**e) for e in data.get("entries", [])]
        return cls(run_id=data["run_id"], entries=entries)

    def digest_section(self, max_items: int = 30) -> str:
        """Compact '## Already injected' section for the LLM context (< 500 tokens)."""
        if not self.entries:
            return ""
        shown = self.entries[-max_items:]
        lines = ["## Already injected in this run"]
        for e in shown:
            if e.parent:
                lines.append(f"- [[{e.title}]] (parent: [[{e.parent}]]) [{e.source_basename}]")
            else:
                lines.append(f"- [[{e.title}]] [{e.source_basename}]")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# IssueCard — human-in-the-loop escalation
# ---------------------------------------------------------------------------

@dataclass
class IssueCard:
    """A question the planner cannot resolve without human input."""
    task_id: str
    question: str
    options: list[dict[str, Any]]
    default_option: str | None = None
    free_form_allowed: bool = False


# ---------------------------------------------------------------------------
# Task — one unit of work inside a ProgressLedger
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A single unit of work tracked by the ProgressLedger."""
    id: str
    capability_name: str
    status: TaskStatus = "pending"
    input_ref: str | None = None    # path to input payload on disk
    output_ref: str | None = None   # path to output payload on disk
    depends_on: list[str] = field(default_factory=list)
    attempts: int = 0
    content_hash: str | None = None # for idempotent checkpoint resumption (Phase 2)
    error: str | None = None


# ---------------------------------------------------------------------------
# ProgressLedger — mutable execution state
# ---------------------------------------------------------------------------

@dataclass
class ProgressLedger:
    """Mutable, serialisable run-state for a planner pipeline execution."""
    run_id: str
    mode: str
    started_at: str          # ISO-8601 UTC
    last_updated: str        # ISO-8601 UTC
    inputs: dict[str, Any]
    tasks: list[Task] = field(default_factory=list)
    issues: list[IssueCard] = field(default_factory=list)
    cursor: str | None = None  # task_id currently running

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def new(cls, mode: str, inputs: dict[str, Any] | None = None) -> ProgressLedger:
        now = _now()
        return cls(
            run_id=uuid.uuid4().hex,
            mode=mode,
            started_at=now,
            last_updated=now,
            inputs=inputs or {},
        )

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_task(
        self,
        capability_name: str,
        *,
        task_id: str | None = None,
        input_ref: str | None = None,
        depends_on: list[str] | None = None,
    ) -> Task:
        t = Task(
            id=task_id or uuid.uuid4().hex,
            capability_name=capability_name,
            input_ref=input_ref,
            depends_on=depends_on or [],
        )
        self.tasks.append(t)
        return t

    def set_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        error: str | None = None,
    ) -> None:
        t = self._get(task_id)
        t.status = status
        if error is not None:
            t.error = error
        if status == "running":
            t.attempts += 1
            self.cursor = task_id
        elif status in ("done", "failed", "skipped", "blocked", "deferred"):
            if self.cursor == task_id:
                self.cursor = None
        self._touch()

    def mark_done(
        self,
        task_id: str,
        *,
        output_ref: str | None = None,
        content_hash: str | None = None,
    ) -> None:
        t = self._get(task_id)
        t.status = "done"
        if output_ref is not None:
            t.output_ref = output_ref
        if content_hash is not None:
            t.content_hash = content_hash
        if self.cursor == task_id:
            self.cursor = None
        self._touch()

    def mark_failed(self, task_id: str, error: str = "") -> None:
        t = self._get(task_id)
        t.status = "failed"
        t.error = error
        if self.cursor == task_id:
            self.cursor = None
        self._touch()

    def next_pending(self) -> Task | None:
        """Return first pending task whose dependencies are all done.

        Tasks with status 'blocked' or 'deferred' are never returned —
        they require human resolution via IssueCard before they can proceed.
        """
        done_ids = {t.id for t in self.tasks if t.status == "done"}
        for t in self.tasks:
            if t.status != "pending":
                continue
            if all(dep in done_ids for dep in t.depends_on):
                return t
        return None

    # ------------------------------------------------------------------
    # Idempotency helpers (Phase 2)
    # ------------------------------------------------------------------

    @property
    def run_dir(self) -> Path:
        """Directory that owns all artefacts for this run."""
        return _RUNS_DIR / self.run_id

    def is_checkpoint_done(self, task_id: str, content_hash: str) -> str | None:
        """Return the output_ref if task_id is done with a matching content_hash.

        Used by the FSM to skip a checkpoint that was already successfully
        processed in a prior (or partial) run with the same input data.
        Returns None if the task has not been done or the hash differs.
        """
        for t in self.tasks:
            if t.id == task_id and t.status == "done" and t.content_hash == content_hash:
                return t.output_ref
        return None

    # ------------------------------------------------------------------
    # Digest — compact summary for LLM context injection
    # ------------------------------------------------------------------

    def digest(self, manifest: "RunManifest | None" = None) -> str:
        """Return a compact human-readable run summary, targeting < 500 tokens.

        Loads TaskLedger from disk (if present) to include the immutable plan.
        Pass `manifest` to append the '## Already injected' section so the
        distiller knows what was created in earlier chunks.
        Safe to call at any point during or after a run.
        """
        # Try to load the immutable plan
        plan_lines: list[str] = []
        request_line = ""
        try:
            tl = TaskLedger.load(self.run_id)
            if tl.user_request:
                request_line = f"REQUEST  {tl.user_request}\n"
            if tl.checkpoints:
                spec_strs = "  ".join(f"{s.id}({s.kind})" for s in tl.checkpoints)
                plan_lines = [
                    f"PLAN  [{len(tl.checkpoints)} checkpoints]",
                    f"  {spec_strs}",
                ]
        except Exception:
            pass

        # Build per-task status lines, grouping f{fi}_c{ci}_{cap} tasks by file
        _sym = {
            "done":     "✓",
            "running":  "→",
            "failed":   "✗",
            "blocked":  "⊘",
            "deferred": "⟳",
            "skipped":  "–",
        }
        counts: dict[str, int] = {}
        task_lines: list[str] = []

        import re as _re
        _file_pat = _re.compile(r"^f(\d+)_c(\d+)_(.+)$")

        # Group tasks by file index for the f{fi}_c{ci}_{cap} scheme
        file_groups: dict[int, list[Task]] = {}
        ungrouped: list[Task] = []
        for t in self.tasks:
            m = _file_pat.match(t.id)
            if m:
                fi = int(m.group(1))
                file_groups.setdefault(fi, []).append(t)
            else:
                ungrouped.append(t)

        # Emit ungrouped tasks first (recon, payload, rollback, …)
        for t in ungrouped:
            counts[t.status] = counts.get(t.status, 0) + 1
            sym = _sym.get(t.status, "·")
            line = f"  {sym} {t.id}"
            if t.status == "running":
                line += f" (attempts={t.attempts})"
            elif t.error:
                line += f"  [{t.error[:60]}]"
            task_lines.append(line)

        # Emit per-file groups with summary header
        sources = (self.inputs or {}).get("sources", [])
        for fi in sorted(file_groups.keys()):
            file_tasks = file_groups[fi]
            done_n = sum(1 for t in file_tasks if t.status == "done")
            total_n = len(file_tasks)
            fname = ""
            if fi < len(sources):
                fname = sources[fi].get("inbox_file", "") if isinstance(sources[fi], dict) else ""
            label = fname.rsplit("/", 1)[-1].removesuffix(".md") if fname else f"file{fi}"
            task_lines.append(f"  FILE {label} [done={done_n}/{total_n}]")
            for t in file_tasks:
                counts[t.status] = counts.get(t.status, 0) + 1
                sym = _sym.get(t.status, "·")
                line = f"    {sym} {t.id}"
                if t.status == "running":
                    line += f" (attempts={t.attempts})"
                elif t.error:
                    line += f"  [{t.error[:60]}]"
                task_lines.append(line)

        counts_str = "  ".join(f"{s}={n}" for s, n in counts.items())
        progress_header = f"PROGRESS  [{counts_str}]"

        inputs_str = "  ".join(f"{k}={v}" for k, v in (self.inputs or {}).items())

        sep = "─" * 36
        parts = [f"RUN {self.run_id[:8]} | {self.mode} | {self.started_at[:19]}Z"]
        if request_line:
            parts.append(request_line.rstrip())
        if plan_lines:
            parts.append(sep)
            parts.extend(plan_lines)
        parts.append(sep)
        parts.append(progress_header)
        parts.extend(task_lines)
        if self.cursor:
            parts.append(f"CURSOR: {self.cursor}")
        if inputs_str:
            parts.append(f"INPUTS: {inputs_str}")
        if manifest is not None:
            section = manifest.digest_section()
            if section:
                parts.append(sep)
                parts.append(section)

        try:
            from silica.kernel.deferred import get_deferred_store
            depth = get_deferred_store().queue_depth()
            if depth > 0:
                parts.append(f"REVIEW QUEUE: {depth} bundle(s) pending — run /review to inspect")
        except Exception:
            pass

        try:
            from silica.config import CONFIG
            from silica.kernel import codedocs
            if CONFIG.vault_path:
                from pathlib import Path
                n = codedocs.stale_count(Path(CONFIG.vault_path))
                if n > 0:
                    parts.append(f"STALE DOCS: {n} note/path pair(s) — run /stale to inspect")
        except Exception:
            pass

        try:
            from silica.config import CONFIG
            from silica.kernel import plans
            if CONFIG.vault_path:
                from pathlib import Path
                counts = plans.status_counts(Path(CONFIG.vault_path))
                if counts:
                    summary = ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
                    parts.append(f"PLANS: {summary} — run /plans to inspect")
        except Exception:
            pass

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> Path:
        run_dir = _RUNS_DIR / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "ledger.json"
        path.write_bytes(orjson.dumps(dataclasses.asdict(self), option=orjson.OPT_INDENT_2))
        return path

    @classmethod
    def load(cls, run_id: str) -> ProgressLedger:
        path = _RUNS_DIR / run_id / "ledger.json"
        data = orjson.loads(path.read_bytes())
        return _from_dict(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, task_id: str) -> Task:
        for t in self.tasks:
            if t.id == task_id:
                return t
        raise KeyError(f"Task {task_id!r} not found in run {self.run_id!r}")

    def _touch(self) -> None:
        self.last_updated = _now()


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _from_dict(data: dict[str, Any]) -> ProgressLedger:
    tasks = [Task(**t) for t in data.get("tasks", [])]
    issues = [IssueCard(**i) for i in data.get("issues", [])]
    return ProgressLedger(
        run_id=data["run_id"],
        mode=data["mode"],
        started_at=data["started_at"],
        last_updated=data["last_updated"],
        inputs=data.get("inputs", {}),
        tasks=tasks,
        issues=issues,
        cursor=data.get("cursor"),
    )


def _task_ledger_from_dict(data: dict[str, Any]) -> TaskLedger:
    checkpoints = [PlanStep(**c) for c in data.get("checkpoints", [])]
    return TaskLedger(
        run_id=data["run_id"],
        user_request=data.get("user_request", ""),
        checkpoints=checkpoints,
        facts=data.get("facts", {}),
        created_at=data.get("created_at", ""),
    )
