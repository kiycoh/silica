"""Non-`.md` → `.md` conversion — ingress frontier (ADR-0009).

A plain function, not a `SourceAdapter`: `/convert` exposes it and `/ingest`
calls it as the fallback when no source adapter claims a file. Dispatch is by
extension; PDF is the only converter today, provider-selectable via
`CONFIG.pdf_provider` (ADR-0011): `markitdown` default (permissive, text-only),
`docling` (permissive, keeps figures/tables), `mineru` (heavyweight CLI). All
open-source under permissive licences; the user installs the chosen one.

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
#
# TODO(real-api): each provider's third-party call surface is only exercised by
# hand-faked modules in tests/test_convert.py — a library rename would drift the
# fakes and pass silently. Add a real-install smoke test to catch API drift.

def _pdf_via_markitdown(src: Path, workdir: Path) -> tuple[str, Path]:
    try:
        from markitdown import MarkItDown
    except ImportError:
        raise ValueError(
            "markitdown not installed — `pip install 'markitdown[pdf]'`, "
            "or set SILICA_PDF_PROVIDER to docling/mineru"
        ) from None

    md = MarkItDown().convert(str(src)).text_content
    # ponytail: markitdown is text-only for PDF — no figures extracted. The empty
    # images dir is honest; _copy_images no-ops on a missing dir. Use docling/mineru
    # if you need the figures carried into the vault.
    return md, workdir / "images"


def _pdf_via_docling(src: Path, workdir: Path) -> tuple[str, Path]:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.types.doc import ImageRefMode
    except ImportError:
        raise ValueError(
            "docling not installed — `pip install docling`, "
            "or set SILICA_PDF_PROVIDER to markitdown/mineru"
        ) from None

    opts = PdfPipelineOptions()
    opts.generate_picture_images = True  # else REFERENCED export emits placeholders
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    doc = converter.convert(str(src)).document
    images = workdir / "images"
    md_path = workdir / f"{src.stem}.md"
    doc.save_as_markdown(md_path, image_mode=ImageRefMode.REFERENCED, artifacts_dir=images)
    return md_path.read_text(encoding="utf-8", errors="replace"), images


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


_PDF_PROVIDERS = {
    "markitdown": _pdf_via_markitdown,
    "docling": _pdf_via_docling,
    "mineru": _pdf_via_mineru,
}


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
