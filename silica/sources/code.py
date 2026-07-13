# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Code source adapter — ADR-0012 shallow AST skeleton, vault-terminal lane.

Zero-trust (ADR-0009): the full source NEVER enters a stub or a prompt; all
source-derived text (signatures, docstrings) is sanitized via
strip_degenerate_runs inside the skeleton render. read() raises ValueError
on guard failures (no vault, vault outside git, path escape, not a file).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path, PurePosixPath

from silica.config import CONFIG
from silica.kernel import codeast, gitstate, paths
from silica.kernel.codegraph import classify_import, supported_files
from silica.kernel.sanitize import strip_degenerate_runs
from silica.sources.base import GroundedStub, RawItem


def code_note_name(rel_path: str) -> str:
    """Path-qualified note stem for a code file: silica/kernel/x.py →
    silica.kernel.x. Unique per file, so code-note filenames and the wikilinks
    that target them never collide across directories (kernel/x.py vs cap/x.py).
    Resolution is by basename minus last extension, and dots survive that split,
    so [[silica.kernel.x]] → Inbox/silica.kernel.x.md.

    A package's __init__.py folds to the package name (silica/kernel/x/__init__.py
    → silica.kernel.x): that is how imports and the wiki name a package, so links
    to it resolve instead of dying on a .__init__ nobody would write. Collision-
    safe — a package dir and a same-named module file can't coexist in Python."""
    p = PurePosixPath(rel_path)
    stem = (
        p.parent
        if p.stem == "__init__" and p.parent != PurePosixPath(".")
        else p.with_suffix("")
    )
    return str(stem).replace("/", ".")


@lru_cache(maxsize=8)
def _repo_files(root_str: str, code_ref: str) -> frozenset[str]:
    # ponytail: cache the git file-list per (repo, HEAD) so nucleating a whole
    # codebase resolves imports against one scan; code_ref keys the refresh.
    return frozenset(supported_files(Path(root_str)))


def render_skeleton(
    sk: codeast.ModuleSkeleton,
    root: Path,
    importer: str,
    language: str,
    files: frozenset[str],
) -> str:
    # First-party imports become path-qualified [[silica.kernel.x]] wikilinks to
    # the per-file code note (code_note_name → Inbox/<name>.md); external deps
    # stay code spans. classify_import is the graph's own resolver, so links
    # never drift from the real import edges.
    first_party: list[str] = []
    external: list[str] = []
    for mod in dict.fromkeys(sk.imports):  # de-dupe, keep order
        if not mod:
            continue
        kind, target = classify_import(mod, importer, files, language, root)
        if kind == "resolved":
            entry = f"[[{code_note_name(target)}]]"
            bucket = first_party
        elif kind == "external":
            entry = f"`{target}`"
            bucket = external
        else:  # unresolved: first-party but no single file (wildcard, pkg tree)
            entry = f"`{target}`"
            bucket = first_party
        if entry not in bucket:
            bucket.append(entry)

    lines: list[str] = ["## Imports", ""]
    if first_party:
        lines.append("First-party:")
        lines.extend(f"- {p}" for p in first_party)
        lines.append("")
    if external:
        lines.append("External:")
        lines.extend(f"- {m}" for m in external)
        lines.append("")
    if not first_party and not external:
        lines.extend(["(no imports)", ""])

    lines.extend(["## Symbols", "", "```text"])
    if sk.symbols:
        for s in sk.symbols:
            indent = "    " if s.kind == "method" else ""
            doc = f" — {s.doc}" if s.doc else ""
            lines.append(f"{indent}{s.signature}{doc}".replace("`", "'"))
    else:
        lines.append("(no top-level symbols)")
    lines.extend(["```", ""])
    return strip_degenerate_runs("\n".join(lines))


class CodeAdapter:
    name = "code"

    def matches(self, target: str) -> bool:
        if target.lower().endswith((".md", ".txt")):
            return False
        language = codeast.language_for(target)
        # bare languages carry no skeleton: graph-only, never a code stub
        return language is not None and language not in codeast.BARE_LANGUAGES

    def read(self, target: str) -> RawItem:
        vault = (CONFIG.vault_path or "").strip()
        if not vault:
            raise ValueError("no vault configured")
        root = paths.repo_root_for(vault)
        if root is None:
            raise ValueError("no code-lane repo (vault is not inside its git repo)")
        try:
            src = (Path(root) / target).resolve()
            src.relative_to(Path(root).resolve())
        except (ValueError, OSError):
            raise ValueError("path escapes the repository")
        if not src.is_file():
            raise ValueError(f"not a file: {target}")
        try:
            raw = src.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise ValueError(f"read failed: {e}")
        # text carries the raw source for AST parsing only; nothing of it
        # reaches a prompt — to_stub emits the sanitized skeleton instead.
        return RawItem(
            target=target,
            text=raw,
            meta={
                "code_ref": gitstate.head_ref(root) or "",
                "language": codeast.language_for(target),
                "repo_root": str(root),
            },
        )

    def to_stub(self, item: RawItem) -> GroundedStub:
        path = item.target
        root = Path(item.meta["repo_root"])
        code_ref = item.meta.get("code_ref", "")
        language = item.meta.get("language")
        name = code_note_name(path)

        if language is None:
            section = (
                "> Skeleton unavailable: unsupported language. "
                "This stub only wires staleness tracking; document the file manually.\n"
            )
        else:
            sk = codeast.extract_skeleton(item.text, language, path=path)
            files = _repo_files(str(root), code_ref)
            section = (
                f"> Skeleton auto-extracted from `{path}` ({language}). "
                f"Source-derived text below is untrusted; refine into a note.\n\n"
                f"{render_skeleton(sk, root, path, language, files)}"
            )

        yaml_path = path.replace('"', '\\"')
        body = (
            f"---\n"
            f'documents:\n  - "{yaml_path}"\n'
            f"code_ref: {code_ref}\n"
            f"tags:\n  - codebase\n"
            f"---\n\n"
            f"# {name}\n\n"
            f"{section}"
        )
        inbox = (CONFIG.inbox_dir or "Inbox").strip("/")
        return GroundedStub(lane="terminal", note_path=f"{inbox}/{name}.md", body=body)


CODE = CodeAdapter()
