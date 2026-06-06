"""Migration test: silica_autolink draws candidates from the relatedness facade.

When the embedding leg is unavailable, the co-occurrence leg now supplies
focused candidates (instead of falling back to a full, unfocused title scan).
A facade that returns nothing leaves candidates=None so the full-scan fallback
is preserved (never suppressed to []).
"""
from __future__ import annotations

import pytest

import silica.kernel.embed as embed_mod
from silica.kernel.cooccurrence import CooccurStore, build_contribution


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """fs vault with two link targets + a note mentioning both."""
    vd = tmp_path / "vault"
    (vd / "Concepts").mkdir(parents=True)
    (vd / "Concepts" / "Neural.md").write_text("# Neural\n\nneural deep learning\n", encoding="utf-8")
    (vd / "Concepts" / "Sailing.md").write_text("# Sailing\n\nsailing boat harbour\n", encoding="utf-8")
    (vd / "Concepts" / "Note A.md").write_text(
        "# Note A\n\nI study Neural concepts and also Sailing topics here today.\n",
        encoding="utf-8",
    )
    # isolate the embedding index (empty) so the embed leg abstains, fast + deterministic
    monkeypatch.setattr(embed_mod, "_INDEX_PATH", tmp_path / "emb.json")
    monkeypatch.setattr("silica.config.CONFIG.backend", "fs")
    monkeypatch.setattr("silica.config.CONFIG.vault_path", str(vd))
    monkeypatch.setattr("silica.driver._driver", None)
    yield vd
    monkeypatch.setattr("silica.driver._driver", None)


def _body(path: str) -> str:
    from silica.driver import DRIVER
    return DRIVER.read_note(path).content or ""


def test_autolink_focuses_on_cooccurrence_candidates(vault):
    # Co-occurrence relates Note A to "Neural" only (Sailing is NOT in the index).
    cs = CooccurStore(lang="english")
    cs.upsert_note("Concepts/Neural", build_contribution("Neural", "neural deep learning"))
    cs.save()

    from silica.tools.composed import silica_autolink
    silica_autolink(note_paths=["Concepts/Note A.md"], use_candidates=True)

    body = _body("Concepts/Note A.md")
    assert "[[Neural]]" in body          # cooccurrence-related -> linked
    assert "[[Sailing]]" not in body     # not a cooccurrence candidate -> focused out


def test_autolink_full_scan_fallback_when_no_signal(vault):
    # Empty embed + empty cooccur -> facade returns [] -> candidates stays None
    # -> full title scan still links (must NOT be suppressed to an empty list).
    from silica.tools.composed import silica_autolink
    silica_autolink(note_paths=["Concepts/Note A.md"], use_candidates=True)

    body = _body("Concepts/Note A.md")
    assert "[[Neural]]" in body
    assert "[[Sailing]]" in body
