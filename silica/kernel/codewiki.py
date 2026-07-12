# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""codewiki — deterministic partition + digest for the behavioral code wiki.

Zero LLM in this module. Reads the codegraph (structure, imports, call edges)
and produces one SubsystemDigest per subsystem: the grounding the capability
layer renders into prompts. Subsystems are directories, the human mental model
of the repo (spec: community-based partition rejected).
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import orjson

from silica.kernel.codegraph import CodeGraph

_EXCLUDED_TOP = {"tests", "test", "docs"}
# ponytail: single source root in v1; multi-package monorepo deferred, seam here


@dataclass(frozen=True)
class Subsystem:
    key: str
    path: str          # repo-relative dir of the subsystem
    members: list[str]  # repo-relative files, sorted


def source_root(graph: CodeGraph) -> str:
    """Top-level dir with the most supported files; "" when loose files at
    the repo root outnumber every dir (the repo root is the source root)."""
    counts: Counter[str] = Counter()
    loose = 0
    for path in graph.files:
        top, _, rest = path.partition("/")
        if rest:
            counts[top] += 1
        else:
            loose += 1
    # >= not >: a tie between loose files and the densest dir means the repo
    # root itself is the source root (flat repo, tests/docs then excluded).
    if not counts or loose >= max(counts.values()):
        return ""
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def partition(graph: CodeGraph) -> list[Subsystem]:
    root_prefix = source_root(graph)
    groups: dict[str, list[str]] = {}
    for path in sorted(graph.files):
        if root_prefix:
            if not path.startswith(root_prefix + "/"):
                continue
            rest = path[len(root_prefix) + 1:]
        else:
            rest = path
        head, _, tail = rest.partition("/")
        if not root_prefix and head in _EXCLUDED_TOP:
            continue
        key = head if tail else "core"
        groups.setdefault(key, []).append(path)
    out: list[Subsystem] = []
    for key, members in sorted(groups.items()):
        if key == "core":
            sub_path = root_prefix
        elif root_prefix:
            sub_path = f"{root_prefix}/{key}"
        else:
            sub_path = key
        out.append(Subsystem(key=key, path=sub_path, members=members))
    return out
