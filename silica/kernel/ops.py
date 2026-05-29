"""Canonical Op schema for the Silica pipeline (ADR-007 / Addendum C1).

This is the single source of truth imported by sanitize, validate, snapshot,
bulk, and lint. No module defines its own op structure locally.

Key invariant: touched_ref() returns op.path — NEVER a field named 'name'
(which does not exist). This closes B1 at the root.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class OpType(str, Enum):
    write = "write"          # create new note (path MUST NOT exist)
    patch = "patch"          # enrich existing note (path MUST exist)
    overwrite = "overwrite"  # rewrite whole note, preserve identity/history
    delete = "delete"        # only via wrapped tool + confirm
    skip = "skip"            # explicit no-op (excluded from gate denominator)


class Op(BaseModel):
    op: OpType
    heading: str                        # concept name; provenance key in payload
    source_basename: str                # inbox filename (basename) this op derives from
    path: str | None = None             # vault-relative path; required for write/patch/overwrite/delete
    title: str | None = None            # clean note filename stem when heading is structurally compound
    snippet: str = ""                   # distilled body (write / patch)
    hub: str | None = None              # [[Hub]] link required for write ops
    content: str | None = None          # full body (overwrite only)
    tags: list[str] | None = None
    related: list[str] | None = None
    reason: str | None = None           # skip reason / rejection note
    linked_axis: str | None = None      # thematic axis this concept belongs to (Layer 2)

    @model_validator(mode="after")
    def validate_path_required(self) -> Op:
        if self.op in (OpType.write, OpType.patch, OpType.overwrite, OpType.delete):
            if not self.path:
                raise ValueError(f"path required for op '{self.op.value}'")
        return self

    def touched_ref(self) -> str | None:
        """The vault path touched by this op.

        This is the ONLY authorised way for lint/snapshot to derive the
        note reference from an op. Using any other field (e.g. 'name') is a
        violation — 'name' does not exist on Op (closes B1).
        """
        if self.op in (OpType.write, OpType.patch, OpType.overwrite, OpType.delete):
            return self.path
        return None

    def __getitem__(self, item: str) -> Any:
        try:
            val = getattr(self, item)
            if isinstance(val, Enum):
                return val.value
            return val
        except AttributeError:
            raise KeyError(item)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, item: str, default: Any = None) -> Any:
        try:
            val = getattr(self, item)
            if isinstance(val, Enum):
                return val.value
            return val
        except AttributeError:
            return default


# ---------------------------------------------------------------------------
# Rollback inverse ops (ADR-009 / Addendum C3)
# ---------------------------------------------------------------------------

class InverseOpKind(str, Enum):
    delete_created = "delete_created"       # undo a write: delete the note that was created
    restore_version = "restore_version"     # undo a patch: history:restore to prior version
    recreate_deleted = "recreate_deleted"   # undo a delete: recreate with prior content


class InverseOp(BaseModel):
    kind: InverseOpKind
    path: str
    version: int | None = None            # for restore_version
    prior_content: str | None = None      # for recreate_deleted


class FailedOp(BaseModel):
    index: int
    path: str
    op: str | None = None
    error: str

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


class BulkResult(BaseModel):
    ok: bool
    failed: list[FailedOp]
    results: list[dict]
    total: int
    successful: int

    @property
    def success(self) -> bool:
        return self.ok

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        d = super().model_dump(*args, **kwargs)
        d["success"] = self.ok
        return d

    def __getitem__(self, item: str) -> Any:
        if item == "success":
            return self.ok
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        if item == "success":
            return self.ok
        return getattr(self, item, default)


class DistillerOutput(BaseModel):
    main_thematic_axes: list[str] = Field(default_factory=list)
    updates: list[Op]


