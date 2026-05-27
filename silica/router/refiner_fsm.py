"""L3 Router / Orchestrator for Silica — Refiner FSM.

Deterministic state machine for the Refiner pipeline.
"""
from __future__ import annotations

import datetime
import hashlib
import logging
import os
import tempfile
from enum import Enum, auto
from typing import Any
import orjson

from silica.driver import DRIVER
from silica.driver.base import NoteRef, Txn, GraphSnapshot
from silica.tools.composed import (
    silica_bulk_write,
    silica_lint,
    silica_validate_ops,
)
from silica.kernel.ops import OpType
from silica.kernel.ops_io import load_ops, parse_ops
from silica.kernel.ledger import get_ledger
from silica.kernel import frontmatter, ofm, templates

logger = logging.getLogger(__name__)


class RefinerState(Enum):
    INIT = auto()
    TRIAGE = auto()
    DELEGATE = auto()      # semantic enrichment worker
    VALIDATE = auto()      # Gate: check rejection rate
    SNAPSHOT = auto()      # build inverses
    WRITE = auto()         # bulk write ops
    LINT = auto()          # Gate: lint + graph diff
    CLEANUP = auto()       # mark committed
    ROLLBACK = auto()      # rollback txn if gate fails
    DONE = auto()
    ERROR = auto()


class RefinerFSM:
    """Deterministic state machine for the Refiner pipeline."""

    def __init__(self, folder: str, hub_override: str | None = None):
        self.folder = folder
        self.hub_override = hub_override

        self.state = RefinerState.INIT
        self.context: dict[str, Any] = {
            "det_ops": [],
            "enrich_queue": [],
            "enrich_ops": [],
            "ops": [],
        }
        self._tmp_files: list[str] = []
        self._txn: Txn | None = None  # holds the live Txn object for ROLLBACK
        self._pre_graph: GraphSnapshot | None = None  # pre-write graph snapshot

        # Load the recipe
        from silica.router.recipe_parser import load_recipe
        try:
            self._recipe = load_recipe("refiner")
        except Exception as e:
            logger.warning("Failed to load recipe 'refiner', using defaults: %s", e)
            self._recipe = {}

        if not self._recipe or "phases" not in self._recipe:
            self._recipe = {
                "name": "refiner",
                "gates": {
                    "rejection_rate_max": 0.10,
                    "graph_regression": "forbid_new_orphans"
                },
                "phases": [
                    { "id": "triage",     "kind": "mechanical", "tool": "silica_triage" },
                    { "id": "enrich",     "kind": "semantic",   "worker": "enricher", "fanout": True, "max_workers": 7 },
                    { "id": "validate",   "kind": "gate",       "tool": "silica_validate_ops", "abort_code": 2 },
                    { "id": "snapshot",   "kind": "txn",        "tool": "silica_snapshot" },
                    { "id": "write",      "kind": "mechanical", "tool": "silica_bulk_write" },
                    { "id": "lint",       "kind": "gate",       "tool": "silica_lint" },
                    { "id": "cleanup",    "kind": "mechanical", "tool": "silica_cleanup", "on_success_only": True },
                    { "id": "rollback",   "kind": "txn",        "tool": "silica_restore", "on_gate_fail": True }
                ]
            }

        self._HANDLERS = {
            RefinerState.TRIAGE: self._handle_triage,
            RefinerState.DELEGATE: self._handle_delegate,
            RefinerState.VALIDATE: self._handle_validate,
            RefinerState.SNAPSHOT: self._handle_snapshot,
            RefinerState.WRITE: self._handle_write,
            RefinerState.LINT: self._handle_lint,
            RefinerState.CLEANUP: self._handle_cleanup,
            RefinerState.ROLLBACK: self._handle_rollback,
        }

        self._ON_ERROR = {
            RefinerState.TRIAGE: RefinerState.ERROR,
            RefinerState.DELEGATE: RefinerState.ERROR,
            RefinerState.VALIDATE: RefinerState.ERROR,
            RefinerState.SNAPSHOT: RefinerState.ERROR,
            RefinerState.WRITE: RefinerState.ROLLBACK,
            RefinerState.LINT: RefinerState.ROLLBACK,
        }

    def _get_recipe_gate(self, name: str, default: Any) -> Any:
        return self._recipe.get("gates", {}).get(name, default)

    def _get_recipe_phase(self, phase_id: str) -> dict:
        for phase in self._recipe.get("phases", []):
            if phase.get("id") == phase_id:
                return phase
        return {}

    def _make_tmp(self, content: Any, suffix: str = ".json") -> str:
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
        return path

    def _cleanup_tmp(self) -> None:
        for path in self._tmp_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._tmp_files.clear()

    def run(self) -> dict[str, Any]:
        self.state = RefinerState.TRIAGE

        try:
            while self.state not in (RefinerState.DONE, RefinerState.ERROR):
                try:
                    self.step()
                except Exception as e:
                    logger.error("FSM Error in state %s: %s", self.state, e)
                    self.context["error"] = str(e)
                    
                    next_state = self._ON_ERROR.get(self.state, RefinerState.ERROR)
                    if next_state == RefinerState.ROLLBACK and self._txn:
                        self.context["abort_reason"] = str(e)
                        self.state = RefinerState.ROLLBACK
                    else:
                        self.state = RefinerState.ERROR
        finally:
            self._cleanup_tmp()

        if self.state == RefinerState.DONE:
            if "final_status" not in self.context:
                self.context["final_status"] = "Success"
            self._write_ledger("committed")
        elif self.state == RefinerState.ERROR:
            if "final_status" not in self.context:
                self.context["final_status"] = f"Failed: {self.context.get('error', 'unknown error')}"
            # C2.5 — materialise 'failed' so the note isn't stuck as stale
            self._write_ledger_failed()

        return self.context

    def step(self) -> None:
        logger.info("Refiner phase: %s", self.state.name)
        handler = self._HANDLERS.get(self.state)
        if handler:
            handler()
        else:
            raise RuntimeError(f"No handler defined for state {self.state}")

    def _transition_success(self) -> None:
        phases = self._recipe.get("phases", [])
        
        PHASE_TO_STATE = {
            "triage": RefinerState.TRIAGE,
            "enrich": RefinerState.DELEGATE,
            "validate": RefinerState.VALIDATE,
            "snapshot": RefinerState.SNAPSHOT,
            "write": RefinerState.WRITE,
            "lint": RefinerState.LINT,
            "cleanup": RefinerState.CLEANUP,
            "rollback": RefinerState.ROLLBACK,
        }

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
                if "cleanup" in [p["id"] for p in phases]:
                    self.state = RefinerState.CLEANUP
                else:
                    self.state = RefinerState.DONE
        elif self.state == RefinerState.CLEANUP:
            self.state = RefinerState.DONE
        elif self.state == RefinerState.ROLLBACK:
            self.state = RefinerState.ERROR

    # ------------------------------------------------------------------
    # State Handlers
    # ------------------------------------------------------------------

    def _handle_triage(self) -> None:
        import glob
        md_files = sorted(glob.glob(os.path.join(self.folder, "**", "*.md"), recursive=True))
        
        det_ops = []
        enrich_queue = []
        summary: dict[str, Any] = {"total": len(md_files), "decouple": 0, "reformat": 0, "enrich": 0, "ok": 0, "errors": []}
        ledger = get_ledger()

        for path in md_files:
            # Compute canonical key for this note (path relative to self.folder, no .md, lowercase)
            try:
                rel = os.path.relpath(path, self.folder).replace("\\", "/")
            except ValueError:
                rel = os.path.basename(path)
            source_canonical = rel.removesuffix(".md").lower()
            basename = os.path.basename(path)  # still used in op dicts below

            # Compute content hash for content-aware skip (C2.2)
            try:
                with open(path, "rb") as _fh:
                    content_hash = hashlib.sha256(_fh.read()).hexdigest()
            except OSError:
                content_hash = ""

            # Store for _write_ledger (keyed by canonical)
            self.context.setdefault("content_hashes", {})[source_canonical] = content_hash

            if ledger.is_committed(source_canonical, content_hash=content_hash):
                logger.info("Skipping already processed note: %s", source_canonical)
                summary["ok"] += 1
                continue

            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                data, _, body = frontmatter.split(content)
                m = ofm.metrics(content)
                heads = ofm.parse_headings(body)
                h2 = [h for h in heads if h["level"] == 2]
                
                over_limit = m["char_count"] > ofm.LIMITS["max_chars"] or m["line_count"] > ofm.LIMITS["max_lines"]
                is_empty = len(body.strip()) == 0
                is_lean = ofm.is_lean(body)
                
                # Determine Category
                if over_limit and len(h2) >= 2:
                    category = "decouple"
                elif is_lean or is_empty:
                    category = "enrich"
                elif data is not None and frontmatter.lint_tags(data):
                    category = "reformat"
                elif data is None:
                    category = "reformat"
                else:
                    category = "ok"

                summary[category] += 1

                # Generate Ops
                if category == "decouple":
                    parent = os.path.dirname(path)
                    hub = self.hub_override or os.path.splitext(basename)[0]
                    
                    preamble = body[:h2[0]["pos"]].strip()
                    if preamble:
                        lines = preamble.splitlines()
                        if lines and lines[0].startswith("# "):
                            preamble = "\n".join(lines[1:]).strip()

                    sections = ofm.sections_by_h2(body)
                    seen: dict[str, int] = {}
                    titles = []
                    for s in sections:
                        slug = templates.slugify(s["title"])
                        seen[slug] = seen.get(slug, 0) + 1
                        fname = slug if seen[slug] == 1 else f"{slug} ({seen[slug]})"
                        titles.append(s["title"])
                        spoke_path = os.path.join(parent, f"{fname}.md")
                        
                        spoke_content = templates.template_spoke(
                            heading=s["title"],
                            snippet=s["content"],
                            hub=hub,
                            tags=[hub]
                        )
                        det_ops.append({
                            "op": "write",
                            "path": spoke_path,
                            "heading": s["title"],
                            "snippet": s["content"],
                            "content": spoke_content,
                            "hub": hub,
                            "source_basename": basename
                        })
                    
                    hub_fm = {
                        "related": [],
                        "tags": [frontmatter.clean_tag(hub)],
                        "last modified": datetime.date.today().strftime("%Y, %m, %d"),
                        "AI": True,
                    }
                    links = "\n".join(f"- [[{t}]]" for t in titles)
                    index_body = f"# {hub}\n\n" + (f"{preamble}\n\n" if preamble else "") + links + "\n"
                    det_ops.append({
                        "op": "overwrite",
                        "path": path,
                        "heading": hub,
                        "source_basename": basename,
                        "content": frontmatter.dump(hub_fm, index_body),
                        "hub": hub
                    })

                elif category == "reformat":
                    if data is not None:
                        norm = frontmatter.normalize_tags(data)
                        new_content = frontmatter.dump(norm, body)
                        det_ops.append({
                            "op": "overwrite",
                            "path": path,
                            "heading": os.path.splitext(basename)[0],
                            "source_basename": basename,
                            "content": new_content,
                            "hub": self.hub_override or os.path.splitext(basename)[0]
                        })

                elif category == "enrich":
                    # Normalize tags deterministically first
                    if data is not None:
                        norm = frontmatter.normalize_tags(data)
                        new_content = frontmatter.dump(norm, body)
                        det_ops.append({
                            "op": "overwrite",
                            "path": path,
                            "heading": os.path.splitext(basename)[0],
                            "source_basename": basename,
                            "content": new_content,
                            "hub": self.hub_override or os.path.splitext(basename)[0]
                        })
                    enrich_queue.append({
                        "path": path,
                        "title": os.path.splitext(basename)[0],
                        "char_count": m["char_count"],
                        "is_empty": is_empty,
                    })

            except Exception as e:
                summary["errors"].append({"path": path, "error": str(e)})

        self.context["det_ops"] = det_ops
        self.context["enrich_queue"] = enrich_queue
        self.context["triage_summary"] = summary
        self._transition_success()

    def _handle_delegate(self) -> None:
        queue = self.context["enrich_queue"]
        if not queue:
            self.context["ops"] = self.context["det_ops"]
            self._transition_success()
            return

        from silica.agent.delegate import delegate
        from silica.agent.llm import call_llm
        from silica.config import CONFIG
        from silica.kernel.sanitize import parse_json

        def enrich_one(task: dict) -> dict:
            path = task["path"]
            title = task["title"]
            hub = self.hub_override or title

            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                return {"error": f"Failed to read file for enrichment: {e}"}

            system_prompt = (
                "You are an academic assistant expert in writing and structuring notes in Obsidian Flavored Markdown (OFM) in English.\n"
                "Your task is to enrich the note specified by the target.\n"
                "Fundamental rules:\n"
                "1. Produce a rigorous, complete, and exhaustive academic text in English.\n"
                "2. Preserve all factual information and concepts already present in the note (anti-deletion policy). Do not remove pre-existing information, but expand upon it.\n"
                "3. Perform structuring in Obsidian Flavored Markdown: use callouts (> [!tip], > [!note]), LaTeX equation blocks ($$ ... $$) if appropriate, lists, and bold text.\n"
                f"4. You must include a wikilink [[{hub}]] to the hub/parent note (for example in a final section called '# Relations' or '# Connections').\n"
                "5. Return the result structured in JSON format containing a single key 'content' with the full body of the note (including normalized and updated YAML frontmatter tags, and the enriched body)."
            )

            user_message = (
                f"Enrich the following note.\n"
                f"Title: {title}\n"
                f"Path: {path}\n"
                f"Current content of the note:\n"
                f"{content}"
            )

            try:
                response = call_llm(
                    model=CONFIG.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    tools=None,
                )
                raw_output = response.text or ""
                parsed, _ = parse_json(raw_output, strict=False)
                if not isinstance(parsed, dict) or "content" not in parsed:
                    return {"error": "Enricher output missing 'content' key", "raw": raw_output[:500]}
                return {"path": path, "content": parsed["content"]}
            except Exception as e:
                return {"error": str(e)}

        phase_conf = self._get_recipe_phase("enrich")
        max_workers = phase_conf.get("max_workers", 7)

        results = delegate(queue, enrich_one, max_workers=max_workers)

        enrich_ops = []
        for idx, r in enumerate(results):
            if "error" in r:
                logger.error("Enricher failed for task %d: %s", idx, r["error"])
                # Degrade gracefully by skipping this enrichment (the det_op tag normalization still stands)
                continue
            enrich_ops.append({
                "op": "overwrite",
                "path": r["path"],
                "heading": os.path.splitext(os.path.basename(r["path"]))[0],
                "source_basename": os.path.basename(r["path"]),
                "content": r["content"],
                "hub": self.hub_override or os.path.splitext(os.path.basename(r["path"]))[0]
            })

        # Merge det_ops and enrich_ops
        merged = []
        enrich_paths = {op["path"] for op in enrich_ops}
        for op in self.context["det_ops"]:
            if op["path"] not in enrich_paths:
                merged.append(op)
        merged.extend(enrich_ops)

        self.context["enrich_ops"] = enrich_ops
        self.context["ops"] = merged
        self._transition_success()

    def _handle_validate(self) -> None:
        ops = self.context["ops"]
        ops_path = self._make_tmp(ops)

        res = silica_validate_ops(
            ops_path,
            payload_paths=[],
            target_dir=self.folder,
        )

        if "error" in res:
            raise RuntimeError(f"Validate failed: {res['error']}")

        self.context["validate"] = res
        max_rate = self._get_recipe_gate("rejection_rate_max", 0.10)
        if not res["success"] or res.get("rejection_rate", 0) >= max_rate:
            self.context["abort_reason"] = (
                f"Rejection rate {res.get('rejection_rate', 0):.1%} >= {max_rate:.1%}"
            )
            self.state = RefinerState.ERROR
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

        # Graph-diff check
        try:
            ops_data = load_ops(self.context["ops_path"])
            touched_refs = []
            snapshot_domain = set()
            
            for op in ops_data:
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
                f"Write partially failed: {failed}/{total} operations failed."
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
            if not res["success"]:
                self.context["abort_reason"] = (
                    f"Lint failed for {path}: {res['errors']}"
                )
                self.state = RefinerState.ROLLBACK
                return

        # Run graph-diff check
        regression_rule = self._get_recipe_gate("graph_regression", "forbid_new_orphans")
        if regression_rule != "allow":
            if self._pre_graph is None:
                self.context["abort_reason"] = "Graph regression gate failed: pre-write snapshot is missing"
                self.state = RefinerState.ROLLBACK
                return
            try:
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
                if not success:
                    self.context["abort_reason"] = (
                        f"Graph regression gate failed: {'; '.join(errors)}"
                    )
                    self.state = RefinerState.ROLLBACK
                    return
            except Exception as e:
                logger.error("Failed to perform graph-diff check: %s", e)
                self.context["abort_reason"] = f"Graph regression gate error: {e}"
                self.state = RefinerState.ROLLBACK
                return

        self._transition_success()

    def _handle_cleanup(self) -> None:
        # Mark committed in the ledger for all ops
        self._write_ledger("committed")
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
            except Exception as e:
                logger.error("Rollback failed: %s", e)
                self.context["rollback_error"] = str(e)
            self._write_ledger_rollback(txn_id)

        self.context["final_status"] = (
            f"Rolled Back: {self.context.get('abort_reason', 'unknown reason')}"
        )
        self._transition_success()

    # ------------------------------------------------------------------
    # Ledger helpers
    # ------------------------------------------------------------------

    def _write_ledger(self, status: str) -> None:
        try:
            txn_id = self.context.get("txn_id", "unknown")
            ops = parse_ops(self.context["ops"])

            for op in ops:
                if op.op == OpType.skip:
                    continue
                # Build source_canonical for this op: path relative to folder, no .md, lowercase
                touched = op.touched_ref() or ""
                try:
                    rel = os.path.relpath(touched, self.folder).replace("\\", "/")
                except ValueError:
                    rel = os.path.basename(touched)
                source_canonical = rel.removesuffix(".md").lower()

                # Retrieve content_hash for this source from context if available
                # (set during triage; keyed by canonical path)
                content_hash = self.context.get("content_hashes", {}).get(source_canonical)

                get_ledger().record(
                    txn_id=txn_id,
                    source_canonical=source_canonical,
                    path=touched,
                    op=op.op.value if op.op else "",
                    status=status,
                    content_hash=content_hash,
                )
        except Exception as e:
            logger.warning("Failed to write ledger: %s", e)

    def _write_ledger_failed(self) -> None:
        """Write 'failed' rows for all ops that were staged (C2.5)."""
        try:
            txn_id = self.context.get("txn_id", "unknown")
            if txn_id == "unknown" or not self.context.get("ops"):
                return
            ops = parse_ops(self.context["ops"])
            for op in ops:
                if op.op == OpType.skip:
                    continue
                touched = op.touched_ref() or ""
                try:
                    rel = os.path.relpath(touched, self.folder).replace("\\", "/")
                except ValueError:
                    rel = os.path.basename(touched)
                source_canonical = rel.removesuffix(".md").lower()
                get_ledger().record(
                    txn_id=txn_id,
                    source_canonical=source_canonical,
                    path=touched,
                    op=op.op.value if op.op else "",
                    status="failed",
                    content_hash=self.context.get("content_hashes", {}).get(source_canonical),
                )
        except Exception as e:
            logger.warning("Failed to write failed ledger: %s", e)

    def _write_ledger_rollback(self, txn_id: str) -> None:
        try:
            get_ledger().mark_rolled_back(txn_id)
        except Exception as e:
            logger.warning("Failed to mark rollback in ledger: %s", e)
