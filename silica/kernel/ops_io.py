# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

import orjson
from typing import Any
from silica.kernel.ops import Op

def parse_ops(raw: list | dict | Any) -> list[Op]:
    """Parse list or updates dict into a list of Op models."""
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
        else:
            ops.append(Op.model_validate(item))
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
