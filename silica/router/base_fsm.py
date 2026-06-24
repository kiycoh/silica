"""Shared FSM mechanics for InjectorFSM."""
from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

import orjson

from silica.kernel.paths import silica_tmp_dir

logger = logging.getLogger(__name__)

StateT = TypeVar("StateT", bound=Enum)


class BaseFSM(Generic[StateT]):
    """Abstract base for deterministic pipeline state machines.

    Each subclass must set in __init__:
        _phase_label      — log prefix ("Injector" / "Refiner")
        _done_state       — terminal-success state
        _error_state      — terminal-failure state
        _rollback_state   — rollback state (triggers txn revert on error)
        _phase_to_state   — dict mapping recipe phase-id → state enum member
        _HANDLERS         — dict mapping state → handler callable
        _ON_ERROR         — dict mapping state → error-fallback state
        state, context, _tmp_files, _txn, _recipe
    """

    # Annotated here for static analysis; values set by subclass __init__.
    _phase_label: str
    _done_state: StateT
    _error_state: StateT
    _rollback_state: StateT
    _phase_to_state: dict[str, StateT]
    _HANDLERS: dict[StateT, Callable[[], None]]
    _ON_ERROR: dict[StateT, StateT]
    state: StateT
    context: dict[str, Any]
    _tmp_files: list[str]
    _txn: Any  # Txn | None at runtime
    _recipe: dict

    # ------------------------------------------------------------------
    # Shared execution mechanics
    # ------------------------------------------------------------------

    def step(self) -> None:
        logger.info("%s phase: %s", self._phase_label, self.state.name)
        handler = self._HANDLERS.get(self.state)
        if handler:
            handler()
        else:
            raise RuntimeError(f"No handler defined for state {self.state}")

    def _transition_success(self) -> None:
        phases = self._recipe.get("phases", [])
        sequence = [
            p["id"]
            for p in phases
            if not p.get("on_gate_fail") and p.get("id") != "rollback" and p.get("id") != "cleanup"
        ]
        current_phase_id: str | None = None
        for k, v in self._phase_to_state.items():
            if v == self.state:
                current_phase_id = k
                break
        if current_phase_id in sequence:
            idx = sequence.index(current_phase_id)
            if idx + 1 < len(sequence):
                self.state = self._phase_to_state[sequence[idx + 1]]
            else:
                if "cleanup" in [p["id"] for p in phases]:
                    self.state = self._phase_to_state["cleanup"]
                else:
                    self._on_sequence_end()
        elif self.state == self._phase_to_state.get("cleanup"):
            self._on_cleanup_done()
        elif self.state == self._rollback_state:
            self.state = self._error_state

    def _run_loop(self) -> dict[str, Any]:
        """Common while-loop: dispatch steps, route errors, clean up tmp files."""
        try:
            while self.state not in (self._done_state, self._error_state):
                try:
                    logger.debug("FSM Transition: %s -> executing handler", self.state.name)
                    self.step()
                except Exception as e:
                    logger.error("FSM Error in state %s: %s", self.state, e)
                    self.context["error"] = str(e)
                    next_state = self._ON_ERROR.get(self.state, self._error_state)
                    if next_state == self._rollback_state and self._txn:
                        self.context["abort_reason"] = str(e)
                        self.state = self._rollback_state
                    else:
                        self.state = self._error_state
        finally:
            self._cleanup_tmp()
        return self.context

    # ------------------------------------------------------------------
    # Hooks — override in subclasses to change terminal behaviour
    # ------------------------------------------------------------------

    def _on_sequence_end(self) -> None:
        """Called when the recipe sequence is exhausted and no cleanup phase exists."""
        self.state = self._done_state

    def _on_cleanup_done(self) -> None:
        """Called after the cleanup phase handler succeeds."""
        self.state = self._done_state

    # ------------------------------------------------------------------
    # Shared recipe helpers
    # ------------------------------------------------------------------

    def _get_recipe_gate(self, name: str, default: Any) -> Any:
        return self._recipe.get("gates", {}).get(name, default)

    def _get_recipe_phase(self, phase_id: str) -> dict:
        for phase in self._recipe.get("phases", []):
            if phase.get("id") == phase_id:
                return phase
        return {}

    # ------------------------------------------------------------------
    # Shared tmp-file helpers
    # ------------------------------------------------------------------

    def _make_tmp(self, content: Any, suffix: str = ".json") -> str:
        """Write content as JSON to ~/.silica/tmp/ and track for cleanup."""
        import uuid
        path = str(silica_tmp_dir() / f"{uuid.uuid4().hex}{suffix}")
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
