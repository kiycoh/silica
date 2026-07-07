"""Cross-encoder rerank pass over a fused candidate pool.

The relatedness facade fuses embeddings + co-occurrence by RANK (RRF); neither leg
ever reads the query and a candidate *together*. A cross-encoder does exactly that,
scoring query x document jointly — the strongest precision lever after first-stage
recall. This module applies that pass to an already-retrieved pool: it reorders,
never retrieves, and abstains (leaves the pool's order untouched) whenever the
reranker is absent or errors, mirroring a down leg in the facade.

The reranker CLIENT lives in agent/providers.py (Reranker/get_reranker); this module
holds only the note-aware reorder, so the client stays a plain HTTP provider.
"""
from __future__ import annotations

from typing import Any, Callable


def note_document(path: str, *, max_chars: int = 800) -> str:
    """Title + body excerpt for one note, as the reranker's document text.

    Returns '' when the note is unreadable (the reranker scores '' as irrelevant,
    which is the right default for a candidate we cannot open).
    """
    try:
        from silica.driver import DRIVER

        content = DRIVER.read_note(path).content or ""
    except Exception:
        return ""
    from silica.kernel import frontmatter

    _data, _raw, body = frontmatter.split(content)
    name = path.rsplit("/", 1)[-1].removesuffix(".md")
    return f"{name}\n{(body or content)[:max_chars]}".strip()


def _path_of(item: Any) -> str:
    if isinstance(item, dict):
        return item.get("path", "")
    return getattr(item, "path", "")


def rerank_related(
    reranker: Any,
    query_text: str,
    results: list,
    *,
    k: int,
    document_of: Callable[[Any], str] | None = None,
) -> list:
    """Reorder `results` by cross-encoder relevance to `query_text`, return top-k.

    Each result is any object/dict exposing a note path (`.path` or `["path"]`).
    Abstention — no reranker, empty query, or the reranker erroring — falls back to
    the pool's existing order truncated to k, so a disabled or down reranker is a
    pure no-op. `document_of(item) -> str` supplies each candidate's text; it
    defaults to reading the note by its path.
    """
    if reranker is None or not results or not query_text:
        return results[:k]
    get_doc = document_of or (lambda it: note_document(_path_of(it)))
    docs = [get_doc(it) for it in results]
    scores = reranker.scores(query_text, docs)
    if scores is None or len(scores) != len(results):
        return results[:k]
    order = sorted(range(len(results)), key=lambda i: scores[i], reverse=True)
    return [results[i] for i in order[:k]]


def demo() -> None:
    """Self-check: reorder by score, truncate to k, abstain keeps order."""

    class _Fake:
        def __init__(self, s):
            self._s = s

        def scores(self, query, documents):
            return self._s

    items = [{"path": "a"}, {"path": "b"}, {"path": "c"}]
    doc = lambda it: it["path"]

    # scores rank c > a > b; k=2 -> [c, a]
    out = rerank_related(_Fake([0.2, 0.1, 0.9]), "q", items, k=2, document_of=doc)
    assert [i["path"] for i in out] == ["c", "a"], out

    # reranker abstains (None) -> pool order, truncated
    out = rerank_related(_Fake(None), "q", items, k=2, document_of=doc)
    assert [i["path"] for i in out] == ["a", "b"], out

    # no reranker -> no-op passthrough, truncated
    out = rerank_related(None, "q", items, k=1, document_of=doc)
    assert [i["path"] for i in out] == ["a"], out

    # empty query -> no-op
    out = rerank_related(_Fake([0.9, 0.0, 0.0]), "", items, k=3, document_of=doc)
    assert [i["path"] for i in out] == ["a", "b", "c"], out

    print("rerank demo ok")


if __name__ == "__main__":
    demo()
