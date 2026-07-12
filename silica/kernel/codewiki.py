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


# ---------------------------------------------------------------------------
# SubsystemDigest — deterministic grounding, one per subsystem
# ---------------------------------------------------------------------------

_HUB_CAP = 5
_REG_DECORATORS = {"command", "route", "tool", "get", "post", "websocket"}
# ponytail: minimal decorator shortlist; extend when a real framework is missing
_ORCHESTRATOR_MIN_CALLS = 3


@dataclass(frozen=True)
class SubsystemDigest:
    key: str
    path: str
    members: list[str]
    struct_sig: str                                # sha256[:16] of members + call set
    public_symbols: dict[str, list[dict]]          # file -> symbol dicts
    module_docs: dict[str, str]
    module_comments: dict[str, list[str]]
    external_deps: list[str]
    collaborators_out: list[tuple[str, int, int]]  # (key, import_w, call_w)
    collaborators_in: list[tuple[str, int, int]]
    fan_in_hubs: list[tuple[str, int]]
    entry_points: list[tuple[str, str]]            # (path, heuristic label)
    flow_sketches: list[list[str]]                 # filled by flow layer
    parse_errors: int


def _file_to_key(subsystems: list[Subsystem]) -> dict[str, str]:
    return {m: s.key for s in subsystems for m in s.members}


def _public_symbols(entry: dict) -> list[dict]:
    allow = entry.get("dunder_all")
    out: list[dict] = []
    for s in entry.get("symbols", []):
        name = s.get("name", "")
        if s.get("parent"):
            if name.startswith("_"):
                continue
            if allow is not None and s["parent"] not in allow:
                continue
        elif allow is not None:
            if name not in allow:
                continue
        elif name.startswith("_"):
            continue
        out.append(s)
    return out


def _struct_sig(members: list[str], call_set: list[tuple[str, str, str]]) -> str:
    payload = orjson.dumps({"members": members, "calls": sorted(call_set)})
    return hashlib.sha256(payload).hexdigest()[:16]


def _pyproject_script_files(root: Path, files: set[str]) -> set[str]:
    pp = root / "pyproject.toml"
    if not pp.is_file():
        return set()
    try:
        import tomllib
        data = tomllib.loads(pp.read_text(encoding="utf-8"))
    except Exception:
        return set()
    out: set[str] = set()
    for target in (data.get("project", {}).get("scripts") or {}).values():
        stem = str(target).split(":", 1)[0].replace(".", "/")
        for cand in (f"{stem}.py", f"{stem}/__init__.py"):
            if cand in files:
                out.add(cand)
    return out


def _entry_points(graph: CodeGraph, sub: Subsystem, scripts: set[str],
                  call_in: Counter, call_out: Counter) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for path in sub.members:
        entry = graph.files[path]
        labels: list[str] = []
        if path in scripts:
            labels.append("project.scripts")
        if entry.get("has_main_guard"):
            labels.append("__main__ guard")
        if path.endswith("__main__.py"):
            labels.append("python -m")
        if any(d.rsplit(".", 1)[-1] in _REG_DECORATORS
               for s in entry.get("symbols", []) for d in s.get("decorators", [])):
            labels.append("registration decorator")
        if call_in[path] == 0 and call_out[path] >= _ORCHESTRATOR_MIN_CALLS:
            labels.append("caller-free orchestrator")
        if labels:
            out.append((path, ", ".join(labels)))
    return out


def build_digests(graph: CodeGraph, subsystems: list[Subsystem],
                  root: Path) -> list[SubsystemDigest]:
    key_of = _file_to_key(subsystems)
    all_calls = graph.call_edges()
    call_in: Counter = Counter()
    call_out: Counter = Counter()
    for src, tgt, _, _ in all_calls:
        if src != tgt:
            call_out[src] += 1
            call_in[tgt] += 1
    scripts = _pyproject_script_files(root, set(graph.files))

    digests: list[SubsystemDigest] = []
    for sub in subsystems:
        member_set = set(sub.members)
        imp_out: Counter = Counter()
        imp_in: Counter = Counter()
        c_out: Counter = Counter()
        c_in: Counter = Counter()
        externals: set[str] = set()
        parse_errors = 0
        for path in sub.members:
            entry = graph.files[path]
            if entry.get("parse_error"):
                parse_errors += 1
            externals.update(entry.get("external", []))
            for tgt in entry.get("imports", []):
                other = key_of.get(tgt)
                if other and other != sub.key:
                    imp_out[other] += 1
        for path, entry in graph.files.items():
            if path in member_set:
                continue
            for tgt in entry.get("imports", []):
                if tgt in member_set:
                    other = key_of.get(path)
                    if other and other != sub.key:
                        imp_in[other] += 1
        sub_calls: list[tuple[str, str, str]] = []
        for src, tgt, callee, _caller in all_calls:
            src_in, tgt_in = src in member_set, tgt in member_set
            if not (src_in or tgt_in):
                continue
            sub_calls.append((src, tgt, callee))
            if src_in and not tgt_in and key_of.get(tgt):
                c_out[key_of[tgt]] += 1
            if tgt_in and not src_in and key_of.get(src):
                c_in[key_of[src]] += 1

        def _merge(imp: Counter, cal: Counter) -> list[tuple[str, int, int]]:
            keys = sorted(set(imp) | set(cal))
            return [(k, imp[k], cal[k]) for k in keys]

        hubs = sorted(((p, graph.fan_in(p)) for p in sub.members),
                      key=lambda t: (-t[1], t[0]))[:_HUB_CAP]
        digests.append(SubsystemDigest(
            key=sub.key, path=sub.path, members=sub.members,
            struct_sig=_struct_sig(sub.members, sub_calls),
            public_symbols={p: _public_symbols(graph.files[p]) for p in sub.members},
            module_docs={p: graph.files[p].get("module_doc", "") for p in sub.members},
            module_comments={p: graph.files[p].get("module_comments", []) for p in sub.members},
            external_deps=sorted(externals),
            collaborators_out=_merge(imp_out, c_out),
            collaborators_in=_merge(imp_in, c_in),
            fan_in_hubs=hubs,
            entry_points=_entry_points(graph, sub, scripts, call_in, call_out),
            flow_sketches=[],
            parse_errors=parse_errors,
        ))
    return digests


def cross_edges(graph: CodeGraph, subsystems: list[Subsystem]) -> list[tuple[str, str, int, int]]:
    key_of = _file_to_key(subsystems)
    imp: Counter = Counter()
    cal: Counter = Counter()
    for path, entry in graph.files.items():
        a = key_of.get(path)
        if not a:
            continue
        for tgt in entry.get("imports", []):
            b = key_of.get(tgt)
            if b and b != a:
                imp[(a, b)] += 1
        for e in entry.get("calls", []):
            b = key_of.get(e["target"])
            if b and b != a:
                cal[(a, b)] += 1
    pairs = sorted(set(imp) | set(cal))
    return [(a, b, imp[(a, b)], cal[(a, b)]) for a, b in pairs]


def edges_ref(edges: list[tuple[str, str, int, int]]) -> str:
    pairs = sorted({(a, b) for a, b, _, _ in edges})
    return hashlib.sha256(orjson.dumps(pairs)).hexdigest()[:16]
