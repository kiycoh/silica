"""L3 Router / Orchestrator for Silica.

From SILICA.md §3 L3 & §7.3:
  State machine for running pipelines. Hardcoded Injector for Phase 2.
  Gates: >= 10% rejection rate -> abort + rollback.
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
from silica.tools.wrapped import silica_snapshot, silica_move

logger = logging.getLogger(__name__)


class InjectorState(Enum):
    INIT = auto()
    RECON = auto()         # Phase 1
    PAYLOAD = auto()       # Phase 2.0
    DELEGATE = auto()      # Phase 2.1
    SANITIZE = auto()      # Phase 2.2
    VALIDATE = auto()      # Phase 2.3 (Gate)
    SNAPSHOT = auto()      # Phase 2.5
    WRITE = auto()         # Phase 3
    LINT = auto()          # Phase 4 (Gate)
    CLEANUP = auto()       # Phase 5
    ROLLBACK = auto()      # On gate fail
    DONE = auto()
    ERROR = auto()


class InjectorFSM:
    """Hardcoded State Machine for the Injector Pipeline (Walking Skeleton)."""

    def __init__(self, inbox_file: str, target_dir: str, hub: str | None = None):
        self.inbox_file = inbox_file
        self.target_dir = target_dir
        self.hub = hub
        
        self.state = InjectorState.INIT
        self.context: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        """Execute the pipeline end-to-end."""
        self.state = InjectorState.RECON
        
        while self.state not in (InjectorState.DONE, InjectorState.ERROR):
            try:
                self.step()
            except Exception as e:
                logger.error("FSM Error in state %s: %s", self.state, e)
                self.context["error"] = str(e)
                # If we've already written or are about to, and have a snapshot, we should rollback
                if self.state in (InjectorState.WRITE, InjectorState.LINT) and "txn" in self.context:
                    self.state = InjectorState.ROLLBACK
                else:
                    self.state = InjectorState.ERROR
                    
        return self.context

    def step(self):
        """Execute the current state and transition."""
        logger.info("Executing Injector phase: %s", self.state.name)
        
        if self.state == InjectorState.RECON:
            res = silica_recon.fn(self.inbox_file)
            if "error" in res:
                raise RuntimeError(f"Recon failed: {res['error']}")
            self.context["recon"] = res
            self.state = InjectorState.PAYLOAD
            
        elif self.state == InjectorState.PAYLOAD:
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as f:
                json.dump(self.context["recon"], f)
                recon_path = f.name
                
            res = silica_payload.fn(recon_path, max_concepts=7)
            if "error" in res:
                raise RuntimeError(f"Payload failed: {res['error']}")
                
            self.context["payload"] = res
            self.state = InjectorState.DELEGATE
            
        elif self.state == InjectorState.DELEGATE:
            # Phase 2.1: Semantic delegation to Sub-Agent
            # For the walking skeleton (S2.2/S2.3), if we don't have the actual distiller 
            # sub-agent running parallel tasks yet, we could either mock it or just pass 
            # the payload through as if the LLM provided ops.
            # In S2.2 we just need the FSM logic. We'll simulate a dummy output.
            dummy_ops = {
                "updates": [
                    {
                        "operation": "create",
                        "name": "Concept From Inbox",
                        "content": "Simulated distilled content from inbox."
                    }
                ]
            }
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as f:
                json.dump(dummy_ops, f)
                self.context["distiller_output_path"] = f.name
                
            self.state = InjectorState.SANITIZE
            
        elif self.state == InjectorState.SANITIZE:
            res = silica_sanitize.fn(self.context["distiller_output_path"])
            if "error" in res:
                raise RuntimeError(f"Sanitize failed: {res['error']}")
                
            self.context["sanitized"] = res
            self.state = InjectorState.VALIDATE
            
        elif self.state == InjectorState.VALIDATE:
            # GATE: rejection rate
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as f:
                json.dump(self.context["sanitized"]["parsed"], f)
                ops_path = f.name
                
            self.context["ops_path"] = ops_path
            res = silica_validate_ops.fn(ops_path, target_dir=self.target_dir)
            
            if "error" in res:
                raise RuntimeError(f"Validate failed: {res['error']}")
                
            self.context["validate"] = res
            if not res["success"]:
                # Abort
                self.context["abort_reason"] = f"Rejection rate {res['rejection_rate']:.2f} >= 10%"
                self.state = InjectorState.ERROR
            else:
                self.state = InjectorState.SNAPSHOT
                
        elif self.state == InjectorState.SNAPSHOT:
            res = silica_snapshot.fn(self.context["ops_path"])
            if "error" in res:
                # If snapshot fails, we shouldn't write. Proceed to error without writing.
                raise RuntimeError(res["error"])
                
            self.context["txn"] = res["_txn_obj"]
            self.state = InjectorState.WRITE
            
        elif self.state == InjectorState.WRITE:
            res = silica_bulk_write.fn(self.context["ops_path"])
            if "error" in res:
                raise RuntimeError(f"Write failed: {res['error']}")
                
            self.context["write"] = res
            self.state = InjectorState.LINT
            
        elif self.state == InjectorState.LINT:
            # GATE: OFM/Atomicity
            ops = self.context["sanitized"]["parsed"]
            if isinstance(ops, dict) and "updates" in ops:
                ops = ops["updates"]
            if not isinstance(ops, list):
                ops = [ops]
                
            touched = {op["name"] for op in ops if op.get("name")}
            for note in touched:
                res = silica_lint.fn(note)
                if not res["success"]:
                    self.context["abort_reason"] = f"Lint failed for {note}: {res['errors']}"
                    self.state = InjectorState.ROLLBACK
                    return
                    
            self.state = InjectorState.CLEANUP
            
        elif self.state == InjectorState.CLEANUP:
            done_dir = "done"
            base_name = os.path.basename(self.inbox_file)
            target = f"{done_dir}/{base_name}"
            
            res = silica_move.fn(self.inbox_file, target)
            if "error" in res:
                # Cleanup failed, but pipeline succeeded.
                self.context["cleanup_warning"] = res["error"]
                
            self.context["final_status"] = "Success"
            self.state = InjectorState.DONE
            
        elif self.state == InjectorState.ROLLBACK:
            txn = self.context.get("txn")
            if txn:
                DRIVER.restore(txn)
                self.context["final_status"] = f"Rolled Back: {self.context.get('abort_reason')}"
            self.state = InjectorState.ERROR
