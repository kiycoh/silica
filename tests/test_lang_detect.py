from silica.kernel.cooccurrence import CooccurStore, build_index, detect_lang


def test_detect_italian():
    text = "Il sistema degli agenti autonomi che cooperano tra loro nella rete."
    assert detect_lang(text) == "italian"


def test_detect_english():
    text = "The system of autonomous agents that cooperate with each other on the network."
    assert detect_lang(text) == "english"


def test_detect_empty_defaults_english():
    assert detect_lang("") == "english"
    assert detect_lang("   \n  ") == "english"


def test_build_index_auto_freezes_detected_lang(tmp_path):
    store = CooccurStore(path=tmp_path / "cooccur.json", lang="auto")
    notes = [
        ("reti/intro.md", "Intro",
         "La rete trasporta i dati tra i nodi che compongono il sistema distribuito."),
        ("reti/tcp.md", "TCP",
         "Il protocollo garantisce che i pacchetti arrivino senza perdite nella rete."),
    ]
    build_index(notes, store=store, lang="auto")

    # lingua rilevata e congelata nello store
    assert store.lang == "italian"
    # le stopword italiane sono state filtrate al build (no junk nei nodi)
    stems = store.top_stems(20)
    assert "che" not in stems
    assert "rete" in [s.lower() for s in stems]
