"""Non-`.md` → `.md` conversion — ingress frontier (ADR-0009).

A plain function, not a `SourceAdapter`: `/convert` exposes it and `/ingest`
calls it as the fallback when no source adapter claims a file. Dispatch is by
extension; PDF is the only converter today, provider-selectable
(`pymupdf4llm` default, MinerU opt-in via `CONFIG.pdf_provider` — ADR-0011).

Both PDF providers return `(markdown, images_dir)`; the rest of the pipeline
(sanitize → copy images flat into the vault → rewrite image links to Obsidian
embeds → write the note to the inbox) is shared and provider-agnostic.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from glob import glob
from pathlib import Path

from silica.config import CONFIG
from silica.kernel.sanitize import strip_degenerate_runs

logger = logging.getLogger(__name__)

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp")
_MD_IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

# MinerU knobs — ponytail: module constants. First run downloads models, so the
# timeout is generous; switch to a VLM/hybrid backend or raise the timeout here.
_MINERU_BACKEND = "pipeline"
_MINERU_TIMEOUT_S = 600


def convert(target: str, dest_dir: str = "") -> str:
    """Convert a non-`.md` file into a `.md` note in the inbox; return its path.

    Dispatch by extension. Unknown extension → ``ValueError``. Side artifacts
    (PDF figures) go to ``<dest_dir>/Images`` when given, else
    ``<inbox>/Images``. The note itself always lands in the inbox (distill
    reads from there).
    """
    if Path(target).suffix.lower() != ".pdf":
        raise ValueError(f"no converter for {Path(target).suffix.lower() or 'this file type'}")
    return _pdf_to_md(target, dest_dir)


def _pdf_to_md(target: str, dest_dir: str) -> str:
    src = _resolve_input(target)
    provider = _PDF_PROVIDERS.get(CONFIG.pdf_provider)
    if provider is None:
        raise ValueError(
            f"unknown pdf_provider {CONFIG.pdf_provider!r} "
            f"(known: {', '.join(_PDF_PROVIDERS)})"
        )
    with tempfile.TemporaryDirectory() as tmp:
        md_text, images_src = provider(src, Path(tmp))
        _copy_images(images_src, _images_dest(dest_dir))  # before tmp is cleaned
    body = _rewrite_image_links(strip_degenerate_runs(md_text))
    note_rel = f"{CONFIG.inbox_dir}/{src.stem}.md"
    from silica.driver import DRIVER

    DRIVER.create(note_rel, body)
    return note_rel


# --- providers (each: src pdf, workdir → markdown text, images dir) ---------

def _pdf_via_pymupdf4llm(src: Path, workdir: Path) -> tuple[str, Path]:
    try:
        import pymupdf4llm
    except ImportError:
        raise ValueError(
            "pymupdf4llm not installed (AGPL extra) — `pip install silica[pdf]`, "
            "or set SILICA_PDF_PROVIDER=mineru"
        ) from None

    images = workdir / "images"
    md = pymupdf4llm.to_markdown(str(src), write_images=True, image_path=str(images))
    return md, images


def _pdf_via_mineru(src: Path, workdir: Path) -> tuple[str, Path]:
    out = workdir / "out"
    try:
        proc = subprocess.run(
            ["mineru", "-p", str(src), "-o", str(out), "-b", _MINERU_BACKEND],
            capture_output=True, text=True, timeout=_MINERU_TIMEOUT_S,
        )
    except FileNotFoundError:
        raise ValueError("mineru not installed") from None
    if proc.returncode != 0:
        raise ValueError(f"mineru failed: {proc.stderr.strip()[-300:]}")
    hits = glob(str(out / src.stem / "**" / f"{src.stem}.md"), recursive=True)
    if not hits:
        raise ValueError("mineru produced no markdown")
    md_path = Path(hits[0])
    return md_path.read_text(encoding="utf-8", errors="replace"), md_path.parent / "images"


_PDF_PROVIDERS = {"pymupdf4llm": _pdf_via_pymupdf4llm, "mineru": _pdf_via_mineru}


# --- shared helpers ---------------------------------------------------------

def _resolve_input(target: str) -> Path:
    """Mirror ProseAdapter.read resolution; raise if the file is missing."""
    p = Path(target)
    if not p.is_absolute():
        vault = (CONFIG.vault_path or "").strip()
        p = (Path(vault) / target) if vault else (Path.cwd() / target)
    if not p.exists():
        raise ValueError(f"file not found: {target}")
    return p


def _images_dest(dest_dir: str) -> Path:
    base = dest_dir.strip() or CONFIG.inbox_dir
    return Path(CONFIG.vault_path) / base / "Images"


def _copy_images(src_dir: Path, dest_dir: Path) -> None:
    if not src_dir.is_dir():
        return
    files = [f for f in src_dir.iterdir() if f.is_file()]
    if not files:
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        # ponytail: basenames are unique by construction (content hash / page-index);
        # two PDFs with a same-named figure would clash — namespace then if it bites.
        shutil.copy2(f, dest_dir / f.name)


def _rewrite_image_links(md: str) -> str:
    """`![alt](any/path/x.png)` → `![[x.png]]` (basename, Obsidian embed)."""
    def repl(m: "re.Match[str]") -> str:
        base = os.path.basename(m.group(1))
        return f"![[{base}]]" if base.lower().endswith(_IMG_EXTS) else m.group(0)

    return _MD_IMG_RE.sub(repl, md)
