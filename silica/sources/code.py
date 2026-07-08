# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Code source adapter — ADR-0012 shallow AST skeleton, vault-terminal lane.

Zero-trust (ADR-0009): the full source NEVER enters a stub or a prompt; all
source-derived text (signatures, docstrings) is sanitized via
strip_degenerate_runs inside the skeleton render. read() raises ValueError
on guard failures (no vault, vault outside git, path escape, not a file).
"""
from __future__ import annotations

from pathlib import Path

from silica.config import CONFIG
from silica.kernel import codeast, gitstate
from silica.kernel.sanitize import strip_degenerate_runs
from silica.sources.base import GroundedStub, RawItem


def _package_of(module: str, root: Path) -> str:
    """Resolve a first-party module to package granularity (silica.kernel.x →
    silica/kernel). Falls back to the raw module string."""
    if module.startswith("."):
        return module  # relative import — can't resolve without the importer's location
    parts = [p for p in module.replace("/", ".").split(".") if p]
    pkg: list[str] = []
    for part in parts:
        if root.joinpath(*pkg, part).is_dir():
            pkg.append(part)
        else:
            break
    return "/".join(pkg) if pkg else module


def _is_first_party(module: str, root: Path) -> bool:
    if module.startswith("."):  # python relative / TS "./x" "../x"
        return True
    top = module.split(".")[0].split("/")[0]
    return (root / top).is_dir() or (root / f"{top}.py").is_file()


def _render_skeleton(sk: codeast.ModuleSkeleton, root: Path) -> str:
    first_party: list[str] = []
    external: list[str] = []
    for mod in dict.fromkeys(sk.imports):  # de-dupe, keep order
        if not mod:
            continue
        if _is_first_party(mod, root):
            pkg = _package_of(mod, root)
            if pkg not in first_party:
                first_party.append(pkg)
        elif mod not in external:
            external.append(mod)

    lines: list[str] = ["## Imports", ""]
    if first_party:
        lines.append("First-party:")
        lines.extend(f"- `{p}`" for p in first_party)
        lines.append("")
    if external:
        lines.append("External:")
        lines.extend(f"- `{m}`" for m in external)
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
        return codeast.language_for(target) is not None

    def read(self, target: str) -> RawItem:
        vault = (CONFIG.vault_path or "").strip()
        if not vault:
            raise ValueError("no vault configured")
        root = gitstate.find_repo_root(Path(vault))
        if root is None:
            raise ValueError("vault is not inside a git repo")
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
        stem = Path(path).stem

        if language is None:
            section = (
                "> Skeleton unavailable: unsupported language. "
                "This stub only wires staleness tracking; document the file manually.\n"
            )
        else:
            sk = codeast.extract_skeleton(item.text, language, path=path)
            section = (
                f"> Skeleton auto-extracted from `{path}` ({language}). "
                f"Source-derived text below is untrusted; refine into a note.\n\n"
                f"{_render_skeleton(sk, root)}"
            )

        yaml_path = path.replace('"', '\\"')
        body = (
            f"---\n"
            f'documents:\n  - "{yaml_path}"\n'
            f"code_ref: {code_ref}\n"
            f"tags:\n  - codebase\n"
            f"---\n\n"
            f"# {stem}\n\n"
            f"{section}"
        )
        inbox = (CONFIG.inbox_dir or "Inbox").strip("/")
        return GroundedStub(lane="terminal", note_path=f"{inbox}/{stem}.md", body=body)


CODE = CodeAdapter()
