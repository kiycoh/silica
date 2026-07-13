"""Cross-encoder reranker: reorder logic (kernel/rerank) + client abstention
(agent/providers.Reranker). The reranker is a precision pass over an already-fused
pool; when absent or erroring it must be a pure no-op that preserves the pool order.
"""
from __future__ import annotations

import httpx

from silica.kernel.rerank import _best_window, rerank_related
from silica.agent.providers import Reranker, get_reranker


class _Fake:
    def __init__(self, scores):
        self._scores = scores
        self.seen = None

    def scores(self, query, documents):
        self.seen = (query, list(documents))
        return self._scores


_ITEMS = [{"path": "a"}, {"path": "b"}, {"path": "c"}]
_DOC = lambda it: it["path"]


def test_reorders_by_score_and_truncates():
    out = rerank_related(_Fake([0.2, 0.1, 0.9]), "q", _ITEMS, k=2, document_of=_DOC)
    assert [i["path"] for i in out] == ["c", "a"]


def test_abstains_when_reranker_returns_none_keeps_order():
    out = rerank_related(_Fake(None), "q", _ITEMS, k=3, document_of=_DOC)
    assert [i["path"] for i in out] == ["a", "b", "c"]


def test_no_reranker_is_passthrough_truncated():
    out = rerank_related(None, "q", _ITEMS, k=1, document_of=_DOC)
    assert [i["path"] for i in out] == ["a"]


def test_empty_query_is_noop():
    out = rerank_related(_Fake([0.9, 0.0, 0.0]), "", _ITEMS, k=3, document_of=_DOC)
    assert [i["path"] for i in out] == ["a", "b", "c"]


def test_mismatched_score_length_abstains():
    # A malformed reranker reply must not silently drop/misalign candidates.
    out = rerank_related(_Fake([0.9]), "q", _ITEMS, k=3, document_of=_DOC)
    assert [i["path"] for i in out] == ["a", "b", "c"]


# --- query-aware document window (long-note blind-spot fix) ----------------

def test_window_surfaces_query_passage_past_head_slice():
    # The relevant turn sits far past a naive head slice; the window must find it
    # so the cross-encoder scores the right text instead of opening chatter.
    body = ("weather smalltalk " * 80) + "I attend yoga classes for my anxiety " \
           + ("unrelated filler " * 80)
    assert len(body) > 800
    win = _best_window(body, "how often do I attend yoga for anxiety?", 200)
    assert "yoga classes for my anxiety" in win


def test_window_is_noop_when_body_fits():
    assert _best_window("short body", "anything", 800) == "short body"


def test_window_falls_back_to_head_without_query_terms():
    long = "x" * 2000
    assert _best_window(long, "a an of", 500) == long[:500]  # no >3-char terms


# --- client ---------------------------------------------------------------

def test_client_parses_results_into_input_order(monkeypatch):
    def fake_post(url, **kwargs):
        return httpx.Response(200, request=httpx.Request("POST", url), json={"results": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.5},
            {"index": 1, "relevance_score": 0.1},
        ]})

    rr = Reranker(base_url="http://x/v1", model="m")
    monkeypatch.setattr(httpx, "post", fake_post)
    scores = rr.scores("q", ["d0", "d1", "d2"])
    assert scores == [0.5, 0.1, 0.9]   # remapped to input order


def test_client_abstains_on_transport_error(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("down")

    rr = Reranker(base_url="http://x/v1", model="m")
    monkeypatch.setattr(httpx, "post", boom)
    assert rr.scores("q", ["d0"]) is None   # abstain, never raise


def test_get_reranker_disabled_without_config():
    class _Cfg:
        rerank_base_url = ""
        rerank_model = ""
    assert get_reranker(_Cfg()) is None


def test_get_reranker_enabled_with_config():
    class _Cfg:
        rerank_base_url = "http://x/v1"
        rerank_model = "bge-reranker"
        rerank_api_key = "k"
    rr = get_reranker(_Cfg())
    assert isinstance(rr, Reranker) and rr.url == "http://x/v1/rerank"
