"""convert() — non-.md → .md ingress frontier (PDF, provider-selectable).

Providers are mocked: pymupdf4llm.to_markdown and the mineru subprocess are
patched so no ML models / real PDFs are needed.
"""
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


# --- pymupdf4llm provider (default) -----------------------------------------

def _fake_pymupdf(monkeypatch, md_template="# Title\n\n![]({img})\n\nbody"):
    """Patch pymupdf4llm.to_markdown to write one image and reference it."""
    pymupdf4llm = pytest.importorskip("pymupdf4llm")  # AGPL extra; skip if absent

    def fake(doc, *, write_images=False, image_path="", **kw):
        Path(image_path).mkdir(parents=True, exist_ok=True)
        img = Path(image_path) / "paper.pdf-0-0.png"
        img.write_bytes(b"\x89PNG fake")
        return md_template.format(img=img)

    monkeypatch.setattr(pymupdf4llm, "to_markdown", fake)


def test_pdf_default_provider_writes_inbox_note_and_embeds_image(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "pymupdf4llm")
    tmp_vault.note("paper.pdf", "%PDF-1.4 fake")
    _fake_pymupdf(monkeypatch)

    note_rel = conv.convert("paper.pdf", dest_dir="Concepts/X")

    assert note_rel == f"{CONFIG.inbox_dir}/paper.md"
    body = _inbox_note(note_rel).read_text(encoding="utf-8")
    assert "# Title" in body
    assert "![[paper.pdf-0-0.png]]" in body          # rewritten to Obsidian embed
    # image copied flat into the --target subfolder's Images dir
    assert (Path(CONFIG.vault_path) / "Concepts/X/Images/paper.pdf-0-0.png").is_file()


def test_pdf_images_fall_back_to_inbox_without_target(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "pymupdf4llm")
    tmp_vault.note("paper.pdf", "x")
    _fake_pymupdf(monkeypatch)

    conv.convert("paper.pdf")  # no dest_dir

    assert (Path(CONFIG.vault_path) / "Inbox/Images/paper.pdf-0-0.png").is_file()


def test_pdf_non_image_link_left_untouched(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "pymupdf4llm")
    tmp_vault.note("paper.pdf", "x")
    _fake_pymupdf(monkeypatch, md_template="see [ref](https://x.test/a) and ![]({img})")

    body = _inbox_note(conv.convert("paper.pdf")).read_text(encoding="utf-8")
    assert "[ref](https://x.test/a)" in body          # ordinary link survives
    assert "![[paper.pdf-0-0.png]]" in body


def test_missing_file_raises(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "pdf_provider", "pymupdf4llm")
    with pytest.raises(ValueError, match="file not found"):
        conv.convert("ghost.pdf")


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
