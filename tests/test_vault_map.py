from pathlib import Path

from silica.kernel.cooccurrence import CooccurStore, build_index
from silica.kernel.vault_map import build_vault_map


def test_empty_store_returns_none(tmp_path):
    store = CooccurStore(path=tmp_path / "cooccur.json")
    assert build_vault_map(store=store) is None


def test_populated_store_yields_map(tmp_path):
    store = CooccurStore(path=tmp_path / "cooccur.json")
    notes = [
        ("ml/embeddings.md", "Embeddings",
         "Embeddings map tokens to vectors. Vector search over embeddings."),
        ("ml/vectors.md", "Vectors",
         "Vector databases index embeddings for similarity search."),
    ]
    build_index(notes, store=store)

    out = build_vault_map(store=store)

    assert out is not None
    assert out.startswith("## Vault map")
    # almeno un termine di dominio emerge nella riga vocabolario
    assert "embed" in out.lower() or "vector" in out.lower()
    # il blocco cluster produce una riga (regressione: non deve marcire in silenzio)
    assert "Top clusters:" in out


def test_inject_appends_system_message(monkeypatch):
    import silica.cli as cli
    import silica.kernel.vault_map as vm

    monkeypatch.setattr(vm, "build_vault_map", lambda **k: "## Vault map\n- Note: 3")
    messages = [{"role": "system", "content": "SYSTEM_PROMPT"}]

    cli._inject_vault_map(messages)

    assert len(messages) == 2
    assert messages[1]["role"] == "system"
    assert "Vault map" in messages[1]["content"]


def test_inject_noop_when_map_is_none(monkeypatch):
    import silica.cli as cli
    import silica.kernel.vault_map as vm

    monkeypatch.setattr(vm, "build_vault_map", lambda **k: None)
    messages = [{"role": "system", "content": "SYSTEM_PROMPT"}]

    cli._inject_vault_map(messages)

    assert len(messages) == 1


def test_inject_swallows_errors(monkeypatch):
    import silica.cli as cli
    import silica.kernel.vault_map as vm

    def _boom(**k):
        raise RuntimeError("index corrotto")

    monkeypatch.setattr(vm, "build_vault_map", _boom)
    messages = [{"role": "system", "content": "SYSTEM_PROMPT"}]

    cli._inject_vault_map(messages)  # non deve sollevare

    assert len(messages) == 1


# ---------------------------------------------------------------------------
# log.md tail (Task 2)
# ---------------------------------------------------------------------------

def _populated_store(tmp_path):
    store = CooccurStore(path=tmp_path / "cooccur.json")
    notes = [
        ("ml/embeddings.md", "Embeddings",
         "Embeddings map tokens to vectors. Vector search over embeddings."),
    ]
    build_index(notes, store=store)
    return store


def test_log_tail_appears_when_log_exists(tmp_vault, tmp_path):
    from silica.config import CONFIG
    from silica.kernel.run_log import append_log_line

    store = _populated_store(tmp_path)
    append_log_line(
        "nucleate `a.md` → 1 new, 0 patch, 0 deferred",
        "runidabc1234",
        vault_path=CONFIG.vault_path,
    )

    out = build_vault_map(store=store)

    assert out is not None
    assert "Recent log" in out
    assert "a.md" in out


def test_log_tail_absent_when_no_log_file(tmp_vault, tmp_path):
    store = _populated_store(tmp_path)

    out = build_vault_map(store=store)

    assert out is not None
    assert "Recent log" not in out


# ---------------------------------------------------------------------------
# ⚠ N note contestate (spec 1 residual, same seam)
# ---------------------------------------------------------------------------

def test_contested_line_present_when_notes_contested(tmp_vault, tmp_path):
    tmp_vault.note(
        "Dir/Contested1.md",
        "---\ncontested: true\ncontradictions:\n  - src.md\n---\nBody\n",
    )
    tmp_vault.note("Dir/Clean.md", "Body without frontmatter\n")

    store = _populated_store(tmp_path)

    out = build_vault_map(store=store)

    assert out is not None
    assert "⚠ 1 contested notes" in out
    assert "[[Contested1]]" in out


def test_contested_line_absent_when_no_contested_notes(tmp_vault, tmp_path):
    tmp_vault.note("Dir/Clean.md", "Body without frontmatter\n")

    store = _populated_store(tmp_path)

    out = build_vault_map(store=store)

    assert out is not None
    assert "contested notes" not in out
