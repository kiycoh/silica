"""L3 Router / Orchestrator for Silica — Refiner FSM.

Deterministic state machine for the Refiner pipeline.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import tempfile
from enum import Enum, auto
from typing import Any

from silica.driver import DRIVER
from silica.driver.base import NoteRef
from silica.tools.composed import (
    silica_bulk_write,
    silica_lint,
    silica_validate_ops,
)
from silica.tools.wrapped import build_txn
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
        self._txn = None  # holds the live Txn object for ROLLBACK
        self._pre_graph = None  # pre-write graph snapshot

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
        summary = {"total": len(md_files), "decouple": 0, "reformat": 0, "enrich": 0, "ok": 0, "errors": []}
        ledger = get_ledger()

        for path in md_files:
            basename = os.path.basename(path)
            if ledger.is_committed(basename):
                logger.info("Skipping already processed note: %s", basename)
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
                    seen = {}
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
                            "hub": hub
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
                "Sei un assistente accademico esperto nella scrittura e strutturazione di note in stile Obsidian Flavored Markdown (OFM) in lingua italiana.\n"
                "Il tuo compito è arricchire la nota specificata dal target.\n"
                "Regole fondamentali:\n"
                "1. Produci un testo accademico rigoroso, completo ed esaustivo in lingua italiana.\n"
                "2. Conserva tutte le informazioni fattuali e i concetti già presenti nella nota (anti-deletion policy). Non rimuovere informazioni preesistenti, ma espandile.\n"
                "3. Esegui la strutturazione in Obsidian Flavored Markdown: usa callout (> [!tip], > [!note]), blocchi di equazioni LaTeX ($$ ... $$) se appropriato, elenchi e grassetti.\n"
                f"4. Includi obbligatoriamente un link wikilink [[{hub}]] verso la nota hub/parent (ad esempio in una sezione finale chiamata '# Relazioni' o '# Collegamenti').\n"
                "5. Restituisci il risultato strutturato in formato JSON contenente una sola chiave 'content' con il corpo completo della nota (inclusi i tag YAML frontmatter normalizzati e aggiornati, e il corpo arricchito)."
            )

            user_message = (
                f"Arricchisci la seguente nota.\n"
                f"Titolo: {title}\n"
                f"Path: {path}\n"
                f"Contenuto attuale della nota:\n"
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
            with open(self.context["ops_path"], "r", encoding="utf-8") as f:
                ops_data = json.load(f)
            self._txn = build_txn(ops_data)
        except Exception as e:
            raise RuntimeError(f"SNAPSHOT rebuild failed: {e}")

        # Graph-diff check
        try:
            touched_refs = []
            for op in ops_data:
                path = op.get("path")
                if path:
                    name = os.path.splitext(os.path.basename(path))[0]
                    touched_refs.append(NoteRef(name=name, path=path))
            self._pre_graph = DRIVER.graph_snapshot(touched_refs)
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
            with open(self.context["ops_path"], "r", encoding="utf-8") as f:
                ops = json.load(f)
        except Exception as e:
            raise RuntimeError(f"LINT: failed to read ops: {e}")

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
                touched_refs = []
                for op in ops:
                    path = op.get("path")
                    if path:
                        name = os.path.splitext(os.path.basename(path))[0]
                        touched_refs.append(NoteRef(name=name, path=path))
                post_graph = DRIVER.graph_snapshot(touched_refs)
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
            ops = self.context["ops"]

            for op in ops:
                if op.get("op") == "skip":
                    continue
                get_ledger().record(
                    txn_id=txn_id,
                    source_basename=os.path.basename(op.get("path", "")),
                    path=op.get("path"),
                    op=op.get("op", ""),
                    status=status,
                )
        except Exception as e:
            logger.warning("Failed to write ledger: %s", e)

    def _write_ledger_rollback(self, txn_id: str) -> None:
        try:
            get_ledger().mark_rolled_back(txn_id)
        except Exception as e:
            logger.warning("Failed to mark rollback in ledger: %s", e)
