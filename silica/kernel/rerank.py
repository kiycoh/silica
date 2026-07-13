# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

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
import re
from typing import Any, Callable


def _best_window(text: str, query: str, width: int) -> str:
    """The `width`-char slice of `text` densest in query terms.

    A cross-encoder sees ~512 tokens (~2k chars); on a long note the naive
    head slice `text[:width]` can miss the passage the query is actually about
    entirely, so the reranker scores irrelevant opening text and demotes a true
    match (measured: on LongMemEval's multi-turn chat sessions the head slice
    evicts gold sessions whose relevant turn sits past char 800). Anchoring the
    window on query-term density fixes that with no extra model call.
    """
    if len(text) <= width:
        return text
    terms = {t for t in re.findall(r"\w+", query.lower()) if len(t) > 3}
    if not terms:
        return text[:width]
    low = text.lower()
    step = max(1, width // 4)
    best_pos, best_hits = 0, -1
    for pos in range(0, max(1, len(text) - width) + step, step):
        hits = sum(low.count(t, pos, pos + width) for t in terms)
        if hits > best_hits:
            best_hits, best_pos = hits, pos
    return text[best_pos:best_pos + width]


def note_document(path: str, *, query: str = "", max_chars: int = 800) -> str:
    """Title + body excerpt for one note, as the reranker's document text.

    With `query`, the excerpt is the query-densest ``max_chars`` window of the
    body (see `_best_window`) rather than the head slice, so a long note's
    relevant passage reaches the cross-encoder. Returns '' when the note is
    unreadable (the reranker scores '' as irrelevant, the right default for a
    candidate we cannot open).
    """
    try:
        from silica.driver import DRIVER

        content = DRIVER.read_note(path).content or ""
    except Exception:
        return ""
    from silica.kernel import frontmatter

    _data, _raw, body = frontmatter.split(content)
    name = path.rsplit("/", 1)[-1].removesuffix(".md")
    text = body or content
    excerpt = _best_window(text, query, max_chars) if query else text[:max_chars]
    return f"{name}\n{excerpt}".strip()


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
    get_doc = document_of or (lambda it: note_document(_path_of(it), query=query_text))
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

    # query-aware window: the relevant passage sits past a naive head slice, and
    # the window must surface it so the cross-encoder scores the right text.
    body = ("intro chatter " * 60) + "the user practices yoga for anxiety " + ("filler " * 60)
    assert len(body) > 800
    win = _best_window(body, "how often yoga for anxiety?", 200)
    assert "yoga for anxiety" in win, win
    assert _best_window("short body", "anything", 800) == "short body"  # no-op under width

    print("rerank demo ok")


if __name__ == "__main__":
    demo()
