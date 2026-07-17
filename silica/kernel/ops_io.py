# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

import logging

import orjson
from pydantic import ValidationError
from typing import Any
from silica.kernel.ops import Op, OpType

logger = logging.getLogger(__name__)


def parse_ops(raw: list | dict | Any) -> list[Op]:
    """Parse list or updates dict into a list of Op models.

    An item that fails validation is salvaged as a skip op (or dropped when
    not even that parses) instead of failing the whole list: the
    non-structured distiller fallback can emit one op with an invalid type,
    and that single item must not kill a multi-file run (real incident:
    2026-07-17, FSM died at Lezione 4 chunk 20 on one bad enum).
    """
    if isinstance(raw, dict) and "updates" in raw:
        items = raw["updates"]
    elif isinstance(raw, list):
        items = raw
    else:
        items = [raw]

    ops = []
    for item in items:
        if isinstance(item, Op):
            ops.append(item)
            continue
        try:
            ops.append(Op.model_validate(item))
        except ValidationError as e:
            if isinstance(item, dict):
                ops.append(Op(
                    op=OpType.skip,
                    heading=str(item.get("heading") or "?"),
                    source_basename=str(item.get("source_basename") or "?"),
                    path=item["path"] if isinstance(item.get("path"), str) else None,
                    reason=f"unparseable op salvaged as skip ({e.error_count()} validation error(s))",
                ))
            logger.warning("parse_ops: invalid op salvaged as skip/dropped: %s", str(e)[:200])
    return ops

def load_ops(path: str) -> list[Op]:
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
    return parse_ops(data)

def dump_ops(path: str, ops: list[Op]) -> None:
    data = [op.model_dump() for op in ops]
    serialized = orjson.dumps(data, option=orjson.OPT_INDENT_2)
    with open(path, "wb") as f:
        f.write(serialized)
