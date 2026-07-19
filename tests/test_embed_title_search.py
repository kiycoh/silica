"""Title-vector search primitive backing the novelty gate's order parameter."""
from silica.kernel.embed import EmbedStore


def _store(tmp_path):
    s = EmbedStore(path=tmp_path / "embeddings.json")
    # Body vecs are deliberately orthogonal to the title vecs so the two signals
    # cannot be confused: title_cosine_top_k must read title vectors only.
    s.upsert("A.md", "Robot Operating System", [1.0, 0.0, 0.0], title_vec=[0.0, 1.0, 0.0])
    s.upsert("B.md", "Banana Bread", [0.0, 1.0, 0.0], title_vec=[0.0, 0.0, 1.0])
    return s


def test_ranks_by_title_vector_not_body_vector(tmp_path):
    s = _store(tmp_path)
    # Query aligned with A's TITLE vec [0,1,0]; it is also aligned with B's BODY
    # vec, so a body search would rank B first. A must win.
    hits = s.title_cosine_top_k([0.0, 1.0, 0.0], k=2)
    assert hits[0]["path"] == "A.md"
    assert hits[0]["score"] > hits[1]["score"]


def test_note_without_title_vec_scores_zero(tmp_path):
    s = EmbedStore(path=tmp_path / "embeddings.json")
    s.upsert("A.md", "Alpha", [1.0, 0.0], title_vec=[0.0, 1.0])
    s.upsert("B.md", "Beta", [0.0, 1.0])  # legacy entry, no title_vec
    hits = s.title_cosine_top_k([0.0, 1.0], k=2)
    assert hits[0]["path"] == "A.md" and hits[0]["score"] > 0.99
    b = next(h for h in hits if h["path"] == "B.md")
    assert b["score"] == 0.0


def test_empty_store_and_exclude(tmp_path):
    assert EmbedStore(path=tmp_path / "empty.json").title_cosine_top_k([1.0, 0.0], k=3) == []
    hits = _store(tmp_path).title_cosine_top_k([0.0, 1.0, 0.0], k=5, exclude={"A.md"})
    assert all(h["path"] != "A.md" for h in hits)
