"""silica_write_note new contract: body-only input, mechanical frontmatter.

The tool takes structured fields, never raw YAML — the fix for the recurring
"frontmatter 'AI' missing" deferrals on interactive/MCP writes.
"""
from __future__ import annotations

import pytest

import silica.kernel.checkpoints as checkpoints
from silica.tools.notes import silica_write_note


@pytest.fixture
def vault(tmp_path, monkeypatch):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    monkeypatch.setenv("SILICA_BACKEND", "fs")
    monkeypatch.setattr("silica.config.CONFIG.backend", "fs")
    monkeypatch.setattr("silica.config.CONFIG.vault_path", str(vault_dir))
    monkeypatch.setattr("silica.driver._driver", None)
    monkeypatch.setattr("silica.kernel.checkpoints._store", None)
    checkpoints.get_checkpoint_store(tmp_path / "checkpoints.db")
    yield vault_dir
    monkeypatch.setattr("silica.driver._driver", None)
    monkeypatch.setattr("silica.kernel.checkpoints._store", None)


def test_body_only_write_gets_mechanical_frontmatter(vault):
    res = silica_write_note(path="CS/Vision.md", body="Computer vision notes.",
                            tags=["Computer Vision"], related=["Deep Learning"])
    assert res.get("success"), res
    landed = (vault / "CS" / "Vision.md").read_text(encoding="utf-8")
    head = landed.split("\n---\n")[0]
    assert "AI: true" in head
    assert "tags:\n  - computer-vision" in head
    assert 'related:\n  - "[[Deep Learning]]"' in head
    assert "# Vision" in landed              # title defaults to filename stem
    assert "Computer vision notes." in landed


def test_named_template_selection(vault):
    (vault / "templates").mkdir()
    (vault / "templates" / "paper.md").write_text(
        "---\nkind: paper\nAI: true\n---\n\n# {{title}}\n\n{{body}}\n",
        encoding="utf-8",
    )
    res = silica_write_note(path="Papers/P.md", body="Abstract.",
                            title="A Paper", template="paper")
    assert res.get("success"), res
    landed = (vault / "Papers" / "P.md").read_text(encoding="utf-8")
    assert "kind: paper" in landed and "# A Paper" in landed


def test_unknown_template_name_errors_listing_available(vault):
    (vault / "templates").mkdir()
    (vault / "templates" / "paper.md").write_text(
        "---\nAI: true\n---\n\n{{body}}\n", encoding="utf-8")
    res = silica_write_note(path="X.md", body="b", template="nope")
    assert "error" in res
    assert "nope" in res["error"] and "paper" in res["error"]
    assert not (vault / "X.md").exists()


def test_template_none_skips_skeleton_but_floors(vault):
    res = silica_write_note(path="Raw.md", body="# Raw\n\nas-is body\n",
                            template="none")
    assert res.get("success"), res
    landed = (vault / "Raw.md").read_text(encoding="utf-8")
    assert landed.startswith("---\nAI: true\nlast modified: ")
    assert "# Raw\n\nas-is body" in landed


def test_leading_yaml_in_body_is_stripped(vault):
    res = silica_write_note(path="Drift.md",
                            body="---\ntags: [rogue]\n---\n\nActual body.")
    assert res.get("success"), res
    landed = (vault / "Drift.md").read_text(encoding="utf-8")
    assert landed.count("---\n") == 2        # exactly one frontmatter block
    assert "rogue" not in landed
    assert "Actual body." in landed


def test_existing_note_still_refused(vault):
    (vault / "Dup.md").write_text("x", encoding="utf-8")
    res = silica_write_note(path="Dup.md", body="y")
    assert "already exists" in res.get("error", "")


def test_template_none_keeps_body_frontmatter(vault):
    """The opt-out contract: template='none' writes the body's own frontmatter
    through untouched (AI ensured, nothing stripped, no second block)."""
    res = silica_write_note(
        path="Own.md",
        body="---\ntags:\n  - mine\n---\n\n# Own\n\nbody\n",
        template="none",
    )
    assert res.get("success"), res
    landed = (vault / "Own.md").read_text(encoding="utf-8")
    head = landed.split("\n---\n")[0]
    assert "tags:\n  - mine" in head and "AI: true" in head
    assert landed.count("---\n") == 2
