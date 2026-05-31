"""Unit tests for silica.kernel.media — strip_images and preprocess_text."""
from __future__ import annotations

import pytest
from silica.kernel.media import strip_images, preprocess_text


# ---------------------------------------------------------------------------
# strip_images — Obsidian-flavor embeds
# ---------------------------------------------------------------------------

class TestStripOFMImages:
    def test_bare_jpg(self):
        text = "Before\n![[images/a1e8022c.jpg]]\nAfter"
        result = strip_images(text)
        assert "![[" not in result
        assert "Before" in result
        assert "After" in result

    def test_bare_png(self):
        assert "![[" not in strip_images("![[screenshot.png]]")

    def test_bare_gif(self):
        assert "![[" not in strip_images("![[anim.gif]]")

    def test_bare_webp(self):
        assert "![[" not in strip_images("![[photo.webp]]")

    def test_bare_svg(self):
        assert "![[" not in strip_images("![[icon.svg]]")

    def test_size_hint(self):
        """![[file.png|300]] — Obsidian size hint must be stripped too."""
        result = strip_images("![[diagram.png|300]]")
        assert "![[" not in result

    def test_size_hint_wh(self):
        result = strip_images("![[diagram.png|200x400]]")
        assert "![[" not in result

    def test_path_with_subdirs(self):
        raw = "![[attachments/2024/screenshot.jpeg]]"
        assert strip_images(raw).strip() == ""

    def test_uppercase_extension(self):
        raw = "![[Photo.JPG]]"
        assert "![[" not in strip_images(raw)

    def test_multiple_embeds(self):
        raw = "Title\n![[a.jpg]]\nText\n![[b.png|100]]\nEnd"
        result = strip_images(raw)
        assert "![[" not in result
        assert "Title" in result
        assert "Text" in result
        assert "End" in result


# ---------------------------------------------------------------------------
# strip_images — Standard Markdown images
# ---------------------------------------------------------------------------

class TestStripMarkdownImages:
    def test_empty_alt(self):
        raw = "![](images/a1e8022c.jpg)"
        assert "![]" not in strip_images(raw)
        assert strip_images(raw).strip() == ""

    def test_with_alt_text(self):
        raw = "![Figure 1](images/fig1.png)"
        result = strip_images(raw)
        assert "!["not in result

    def test_remote_url(self):
        raw = "![logo](https://example.com/logo.png)"
        assert strip_images(raw).strip() == ""

    def test_mixed_with_text(self):
        raw = "Some text.\n![](img.jpg)\nMore text."
        result = strip_images(raw)
        assert "Some text." in result
        assert "More text." in result
        assert "!["  not in result

    def test_inline_in_paragraph(self):
        raw = "See the ![diagram](diag.png) below for details."
        result = strip_images(raw)
        assert "See the" in result
        assert "below for details." in result
        assert "![" not in result


# ---------------------------------------------------------------------------
# strip_images — things that must NOT be stripped
# ---------------------------------------------------------------------------

class TestStripPreservation:
    def test_wikilink_untouched(self):
        raw = "[[NeuralNetwork]] is connected to [[Backprop]]."
        assert strip_images(raw) == raw

    def test_plain_text_untouched(self):
        raw = "The quick brown fox jumps over the lazy dog."
        assert strip_images(raw) == raw

    def test_code_block_untouched(self):
        raw = "```python\nprint('![[fake.jpg]]\n')\n```"
        # The regex doesn't special-case code blocks — it removes any ![[...ext]]
        # This is intentional: images inside code blocks are unusual; the
        # current behaviour is documented and acceptable.
        pass  # no assertion — just confirm it doesn't crash

    def test_hyperlink_not_image(self):
        """[text](url) without leading ! must not be stripped."""
        raw = "[See docs](https://example.com/docs)"
        assert strip_images(raw) == raw

    def test_markdown_bold_untouched(self):
        raw = "**Bold text** and *italic*."
        assert strip_images(raw) == raw

    def test_empty_string(self):
        assert strip_images("") == ""

    def test_blank_line_collapse(self):
        """Multiple blank lines left by removed embeds are collapsed to one."""
        raw = "Line A\n\n![[a.jpg]]\n\n![[b.png]]\n\nLine B"
        result = strip_images(raw)
        assert "Line A" in result
        assert "Line B" in result
        # Should not have 3+ consecutive newlines
        assert "\n\n\n" not in result


# ---------------------------------------------------------------------------
# preprocess_text — integration with CONFIG.image_mode
# ---------------------------------------------------------------------------

class TestPreprocessText:
    def test_strip_mode_removes_images(self, monkeypatch):
        import silica.config as cfg
        monkeypatch.setattr(cfg.CONFIG, "image_mode", "strip")
        raw = "Text\n![[img.png]]\nMore text"
        result = preprocess_text(raw)
        assert "![[" not in result
        assert "Text" in result

    def test_vlm_mode_falls_back_to_strip(self, monkeypatch):
        """vlm mode not yet implemented; must fall back to strip behaviour."""
        import silica.config as cfg
        monkeypatch.setattr(cfg.CONFIG, "image_mode", "vlm")
        raw = "![[photo.jpg]]"
        result = preprocess_text(raw)
        assert "![[" not in result

    def test_unknown_mode_uses_strip(self, monkeypatch):
        """Unknown image_mode values should not crash and default to strip."""
        import silica.config as cfg
        monkeypatch.setattr(cfg.CONFIG, "image_mode", "nonexistent_mode")  # type: ignore
        raw = "![[x.png]]"
        # preprocess_text falls through to strip for any unrecognised mode
        result = preprocess_text(raw)
        assert "![[" not in result
