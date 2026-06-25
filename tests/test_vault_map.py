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
    assert "Cluster principali:" in out


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
