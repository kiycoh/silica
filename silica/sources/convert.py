# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Non-`.md` → `.md` conversion — ingress frontier (ADR-0009).

A plain function, not a `SourceAdapter`: `/convert` exposes it and `/nucleate`
calls it as the fallback when no source adapter claims a file. Dispatch is by
extension; PDF is the only converter today, provider-selectable via
`CONFIG.pdf_provider` (ADR-0011): `mineru` default (heavyweight CLI, best
fidelity, downloads models on first run), `docling` (permissive, keeps
figures/tables and heading structure), `opendataloader` (Java-backed, strong on
complex tables and multi-column reading order, needs a JVM). All open-source
under permissive licences; `mineru` installs via the `silica[pdf]` extra, the
alternatives are installed manually. The default preserves heading structure so
book segmentation (below) has headings to split on instead of falling back to
blind size-cutting.

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

# Book segmentation — a converted book is one giant markdown, but RECON caps
# concepts PER FILE (keyphrase.MAX_CONCEPTS=40), so a whole book in one inbox
# note loses almost everything. Split on chapter headings, then size-cap each
# section so RECON sees book-sized units. ~40k chars ≈ 10k tokens ≈ ~15 pages:
# raise for fewer/larger files, lower for more granular notes.
_MAX_SEGMENT_CHARS = 40_000
_HEADING_RE = re.compile(r"^#{1,2} \S")
_SLUG_RE = re.compile(r"[^a-z0-9]+")

# MinerU knobs — ponytail: module constants. First run downloads models, so the
# timeout is generous; switch to a VLM/hybrid backend or raise the timeout here.
# Measured ~0.9 s/page on CPU (80-page probe): 600s died on an 800-page book.
_MINERU_BACKEND = "pipeline"
_MINERU_TIMEOUT_S = 3600


def convert(target: str, dest_dir: str = "") -> list[str]:
    """Convert a non-`.md` file into one or more `.md` notes in the inbox.

    Returns the list of created note paths. A small PDF is a single note; a
    book-sized PDF is split into chapter/size-bounded segments (see
    ``split_markdown``) so RECON — which caps concepts PER FILE — sees book
    units, not the whole book collapsed into one note. Dispatch by extension;
    unknown extension → ``ValueError``. Side artifacts (PDF figures) go to
    ``<dest_dir>/Images`` when given, else ``<inbox>/Images``.
    """
    if Path(target).suffix.lower() != ".pdf":
        raise ValueError(f"no converter for {Path(target).suffix.lower() or 'this file type'}")
    return _pdf_to_md(target, dest_dir)


def _split_on_headings(md: str) -> list[str]:
    """Split markdown at level-1/2 headings (fence-aware). Always ≥1 segment.

    Content before the first heading stays attached to it (no empty lead
    segment). A ``#``/``##`` inside a fenced code block is not a boundary.
    """
    segs: list[str] = []
    cur: list[str] = []
    in_fence = False
    for line in md.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and _HEADING_RE.match(line) and "".join(cur).strip():
            segs.append("".join(cur))
            cur = []
        cur.append(line)
    if "".join(cur).strip():
        segs.append("".join(cur))
    return segs or [md]


def _split_by_size(text: str, max_chars: int) -> list[str]:
    """Greedy split on blank-line (paragraph) boundaries, ≤ max_chars per part.

    A single paragraph larger than max_chars is left whole (its own oversized
    part) rather than cut mid-sentence — vanishingly rare in prose.
    """
    segs: list[str] = []
    cur = ""
    for part in re.split(r"(\n[ \t]*\n)", text):
        if cur and len(cur) + len(part) > max_chars:
            segs.append(cur)
            cur = ""
        cur += part
    if cur.strip():
        segs.append(cur)
    return segs or [text]


def split_markdown(md: str, max_chars: int = _MAX_SEGMENT_CHARS) -> list[str]:
    """Book-sized markdown → RECON-sized segments: heading-split, packed to size.

    Headings (``#``/``##``) are the cut points; any section still over
    ``max_chars`` is further split on paragraph boundaries — the same
    dimensional fallback that carries a heading-less scan. Adjacent pieces are
    then greedily packed up to ``max_chars``: real converters flatten every
    section to ``##`` and emit lone ``## Chapter N`` lines (verified on an
    80-page docling probe: 53 raw segments, some 14 chars), so raw sections
    over-fragment — packing restores chapter-sized units and absorbs the
    micro-segments. A document smaller than ``max_chars`` packs to a single
    segment. Always returns ≥1 segment.
    """
    pieces: list[str] = []
    for section in _split_on_headings(md):
        if len(section) <= max_chars:
            pieces.append(section)
        else:
            pieces.extend(_split_by_size(section, max_chars))

    out: list[str] = []
    cur = ""
    for p in pieces:
        if cur and len(cur) + len(p) > max_chars:
            out.append(cur)
            cur = ""
        cur += p
    if cur.strip():
        out.append(cur)
    return out or [md]


def _segment_slug(segment: str, fallback: str) -> str:
    """Filename slug from the segment's first heading; ``fallback`` if none."""
    for line in segment.splitlines():
        if _HEADING_RE.match(line):
            slug = _SLUG_RE.sub("-", line.lstrip("#").strip().lower()).strip("-")
            if slug:
                return slug[:50]
    return fallback


# mineru drops the space after , ; : between letters ("symmetric,and positive")
# and the glitch flows into RECON concepts and note titles. Letters-only guard
# keeps digits ("10,000") and LaTeX macros ("\alpha,\beta") untouched.
_TIGHT_PUNCT_RE = re.compile(r"(?<=[A-Za-zà-ÿ])([,;:])(?=[A-Za-zà-ÿ])")


def _respace_prose(md: str) -> str:
    """Re-insert the missing space after ,;: in prose — not in code or math.

    ponytail: inline $…$ spans are skipped per line; display-math interiors are
    not tracked ("x,y" → "x, y" renders identically in LaTeX). Glued words with
    no punctuation ("overthe") need a dictionary — out of scope.
    """
    out: list[str] = []
    in_fence = False
    for line in md.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            parts = line.split("$")
            for i in range(0, len(parts), 2):  # even = outside $…$
                parts[i] = _TIGHT_PUNCT_RE.sub(r"\1 ", parts[i])
            line = "$".join(parts)
        out.append(line)
    return "".join(out)


def _pdf_to_md(target: str, dest_dir: str) -> list[str]:
    src = _resolve_input(target)
    provider = _PDF_PROVIDERS.get(CONFIG.pdf_provider)
    if provider is None:
        raise ValueError(
            f"unknown pdf_provider {CONFIG.pdf_provider!r} "
            f"(known: {', '.join(_PDF_PROVIDERS)})"
        )
    with tempfile.TemporaryDirectory() as tmp:
        md_text, images_src = provider(src, Path(tmp))
        # Copy only images the markdown references: mineru dumps every crop it
        # detects (477 files for a 200-page book, 19 referenced) — the rest
        # would land in the vault as orphans.
        referenced = {os.path.basename(m.group(1)) for m in _MD_IMG_RE.finditer(md_text)}
        _copy_images(images_src, _images_dest(dest_dir), only=referenced)  # before tmp is cleaned
    body = _rewrite_image_links(_respace_prose(strip_degenerate_runs(md_text)))
    from silica.driver import DRIVER

    segments = split_markdown(body)
    # Single segment (a paper, an article) keeps the flat inbox path — no change
    # in behaviour, no subdir for the common case. Image links are basename
    # embeds (![[fig.png]]) so they resolve from any segment regardless of dir.
    if len(segments) == 1:
        note_rel = f"{CONFIG.inbox_dir}/{src.stem}.md"
        DRIVER.create(note_rel, body)
        return [note_rel]

    width = len(str(len(segments)))
    paths: list[str] = []
    for i, seg in enumerate(segments, 1):
        slug = _segment_slug(seg, "part")
        note_rel = f"{CONFIG.inbox_dir}/{src.stem}/{i:0{width}d}-{slug}.md"
        DRIVER.create(note_rel, seg)
        paths.append(note_rel)
    logger.info("PDF %s split into %d inbox segment(s)", src.name, len(segments))
    return paths


# --- providers (each: src pdf, workdir → markdown text, images dir) ---------
#
# TODO(real-api): each provider's third-party call surface is only exercised by
# hand-faked modules in tests/test_convert.py — a library rename would drift the
# fakes and pass silently. Add a real-install smoke test to catch API drift.

def _pdf_via_docling(src: Path, workdir: Path) -> tuple[str, Path]:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.types.doc import ImageRefMode
    except ImportError:
        raise ValueError(
            "docling not installed — `pip install docling`, "
            "or set SILICA_PDF_PROVIDER to mineru/opendataloader"
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


def _pdf_via_opendataloader(src: Path, workdir: Path) -> tuple[str, Path]:
    # Java-backed (JVM per convert), Apache-2.0. Strong on complex tables and
    # multi-column reading order; the wheel bundles the CLI but needs Java 11+.
    try:
        import opendataloader_pdf
    except ImportError:
        raise ValueError(
            "opendataloader-pdf not installed — `pip install opendataloader-pdf` "
            "(needs Java 11+), or set SILICA_PDF_PROVIDER to docling/mineru"
        ) from None

    out = workdir / "out"
    images = workdir / "images"
    opendataloader_pdf.convert(
        input_path=str(src), output_dir=str(out),
        format="markdown", image_output="external", image_dir=str(images),
    )
    hits = glob(str(out / "**" / "*.md"), recursive=True)
    if not hits:
        raise ValueError("opendataloader produced no markdown")
    return Path(hits[0]).read_text(encoding="utf-8", errors="replace"), images


def _pdf_via_mineru(src: Path, workdir: Path) -> tuple[str, Path]:
    out = workdir / "out"
    try:
        proc = subprocess.run(
            ["mineru", "-p", str(src), "-o", str(out), "-b", _MINERU_BACKEND],
            capture_output=True, text=True, timeout=_MINERU_TIMEOUT_S,
        )
    except FileNotFoundError:
        raise ValueError(
            "mineru not installed — `pip install 'silica[pdf]'` (or `pip install "
            "'mineru[pipeline]'`), or set SILICA_PDF_PROVIDER to docling/opendataloader"
        ) from None
    if proc.returncode != 0:
        raise ValueError(f"mineru failed: {proc.stderr.strip()[-300:]}")
    hits = glob(str(out / src.stem / "**" / f"{src.stem}.md"), recursive=True)
    if not hits:
        raise ValueError("mineru produced no markdown")
    md_path = Path(hits[0])
    return md_path.read_text(encoding="utf-8", errors="replace"), md_path.parent / "images"


_PDF_PROVIDERS = {
    "docling": _pdf_via_docling,
    "mineru": _pdf_via_mineru,
    "opendataloader": _pdf_via_opendataloader,
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


def _copy_images(src_dir: Path, dest_dir: Path, only: set[str] | None = None) -> None:
    if not src_dir.is_dir():
        return
    files = [
        f for f in src_dir.iterdir()
        if f.is_file() and (only is None or f.name in only)
    ]
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
