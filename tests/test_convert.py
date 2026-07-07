"""convert() — non-.md → .md ingress frontier (PDF, provider-selectable).

Providers are mocked: markitdown/docling are injected as fake modules and the
mineru subprocess is patched, so no ML models / real PDFs / installs are needed.
"""
import sys
import types
from pathlib import Path

import pytest

from silica.config import CONFIG
from silica.sources import convert as conv


def _inbox_note(note_rel: str) -> Path:
    return Path(CONFIG.vault_path) / note_rel


# --- dispatch ---------------------------------------------------------------

@pytest.mark.parametrize("target", ["notes.xyz", "noext", "data.csv"])
def test_unknown_extension_raises(target):
    with pytest.raises(ValueError, match="no converter"):
        conv.convert(target)


# --- markitdown provider (default, text-only) -------------------------------
#
# TODO(real-api): the fakes below hand-mirror the markitdown/docling APIs
# (`MarkItDown().convert(p).text_content`, `doc.save_as_markdown(path,
# image_mode=, artifacts_dir=)`). They prove the SHARED pipeline, not the
# provider wiring — if a library renames those, the fakes drift with the bug and
# stay green. Add a real-install smoke test (`@pytest.mark.skipif` on import,
# one tiny bundled PDF) to catch API drift against the actual packages.

def _fake_markitdown(monkeypatch, md="# Title\n\nbody"):
    """Inject a fake `markitdown` whose MarkItDown().convert(p).text_content == md."""
    mod = types.ModuleType("markitdown")

    class MarkItDown:
        def convert(self, path):
            return types.SimpleNamespace(text_content=md)

    mod.MarkItDown = MarkItDown
    monkeypatch.setitem(sys.modules, "markitdown", mod)


def test_pdf_default_provider_writes_inbox_note_text_only(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "markitdown")
    tmp_vault.note("paper.pdf", "%PDF-1.4 fake")
    _fake_markitdown(monkeypatch)

    note_rel = conv.convert("paper.pdf", dest_dir="Concepts/X")

    assert note_rel == f"{CONFIG.inbox_dir}/paper.md"
    body = _inbox_note(note_rel).read_text(encoding="utf-8")
    assert "# Title" in body
    # text-only: no figures extracted, so no Images dir is created
    assert not (Path(CONFIG.vault_path) / "Concepts/X/Images").exists()


def test_pdf_markitdown_rewrites_any_image_link(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "markitdown")
    tmp_vault.note("paper.pdf", "x")
    _fake_markitdown(monkeypatch, md="see [ref](https://x.test/a) and ![](a/b/fig.png)")

    body = _inbox_note(conv.convert("paper.pdf")).read_text(encoding="utf-8")
    assert "[ref](https://x.test/a)" in body          # ordinary link survives
    assert "![[fig.png]]" in body                      # image link → Obsidian embed


def test_markitdown_missing_raises(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "markitdown")
    tmp_vault.note("paper.pdf", "x")
    monkeypatch.setitem(sys.modules, "markitdown", None)  # simulate absent package
    with pytest.raises(ValueError, match="markitdown not installed"):
        conv.convert("paper.pdf")


def test_missing_file_raises(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "markitdown")
    with pytest.raises(ValueError, match="file not found"):
        conv.convert("ghost.pdf")


# --- docling provider (keeps figures) ---------------------------------------

def _fake_docling(monkeypatch, md="# Title\n\n![](images/fig.png)\n\nbody"):
    """Inject a fake docling whose save_as_markdown writes one image + references it."""

    class _Doc:
        def save_as_markdown(self, path, *, image_mode, artifacts_dir):
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "fig.png").write_bytes(b"\x89PNG fake")
            Path(path).write_text(md, encoding="utf-8")

    class DocumentConverter:
        def __init__(self, **kw):
            pass

        def convert(self, path):
            return types.SimpleNamespace(document=_Doc())

    dc = types.ModuleType("docling.document_converter")
    dc.DocumentConverter = DocumentConverter
    dc.PdfFormatOption = lambda **kw: None
    base = types.ModuleType("docling.datamodel.base_models")
    base.InputFormat = types.SimpleNamespace(PDF="pdf")
    popts = types.ModuleType("docling.datamodel.pipeline_options")
    popts.PdfPipelineOptions = type("PdfPipelineOptions", (), {})
    core = types.ModuleType("docling_core.types.doc")
    core.ImageRefMode = types.SimpleNamespace(REFERENCED="referenced")
    fakes = {
        "docling": types.ModuleType("docling"),
        "docling.datamodel": types.ModuleType("docling.datamodel"),
        "docling.datamodel.base_models": base,
        "docling.datamodel.pipeline_options": popts,
        "docling.document_converter": dc,
        "docling_core": types.ModuleType("docling_core"),
        "docling_core.types": types.ModuleType("docling_core.types"),
        "docling_core.types.doc": core,
    }
    for name, mod in fakes.items():
        monkeypatch.setitem(sys.modules, name, mod)


def test_pdf_docling_provider_embeds_extracted_image(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "docling")
    tmp_vault.note("paper.pdf", "x")
    _fake_docling(monkeypatch)

    body = _inbox_note(conv.convert("paper.pdf", dest_dir="Concepts/X")).read_text(encoding="utf-8")
    assert "![[fig.png]]" in body
    assert (Path(CONFIG.vault_path) / "Concepts/X/Images/fig.png").is_file()


def test_docling_missing_raises(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "docling")
    tmp_vault.note("paper.pdf", "x")
    monkeypatch.setitem(sys.modules, "docling.document_converter", None)
    with pytest.raises(ValueError, match="docling not installed"):
        conv.convert("paper.pdf")


# --- mineru provider --------------------------------------------------------

def _fake_mineru_run(returncode=0, stderr="", write_md=True):
    def run(cmd, **kw):
        if write_md:
            out = Path(cmd[cmd.index("-o") + 1])
            stem = Path(cmd[cmd.index("-p") + 1]).stem
            d = out / stem / "txt"
            (d / "images").mkdir(parents=True)
            (d / f"{stem}.md").write_text("# M\n\n![](images/h.jpg)\n", encoding="utf-8")
            (d / "images" / "h.jpg").write_bytes(b"img")

        class R:
            pass

        R.returncode, R.stderr, R.stdout = returncode, stderr, ""
        return R()

    return run


def test_pdf_mineru_provider_success(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "mineru")
    tmp_vault.note("paper.pdf", "x")
    monkeypatch.setattr(conv.subprocess, "run", _fake_mineru_run())

    body = _inbox_note(conv.convert("paper.pdf")).read_text(encoding="utf-8")
    assert "![[h.jpg]]" in body
    assert (Path(CONFIG.vault_path) / "Inbox/Images/h.jpg").is_file()


def test_mineru_missing_raises(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "mineru")
    tmp_vault.note("paper.pdf", "x")

    def boom(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(conv.subprocess, "run", boom)
    with pytest.raises(ValueError, match="mineru not installed"):
        conv.convert("paper.pdf")


def test_mineru_nonzero_exit_raises(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "mineru")
    tmp_vault.note("paper.pdf", "x")
    monkeypatch.setattr(
        conv.subprocess, "run", _fake_mineru_run(returncode=1, stderr="kaboom", write_md=False)
    )
    with pytest.raises(ValueError, match="mineru failed"):
        conv.convert("paper.pdf")


def test_unknown_provider_raises(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "bogus")
    tmp_vault.note("paper.pdf", "x")
    with pytest.raises(ValueError, match="unknown pdf_provider"):
        conv.convert("paper.pdf")
