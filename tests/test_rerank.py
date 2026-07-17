"""Cross-encoder reranker: reorder logic (kernel/rerank) + client abstention
(agent/providers.Reranker). The reranker is a precision pass over an already-fused
pool; when absent or erroring it must be a pure no-op that preserves the pool order.
"""
from __future__ import annotations

import httpx

from silica.kernel.rerank import _best_window, rerank_related
from silica.agent.providers import LocalReranker, Reranker, get_reranker


class _Fake:
    def __init__(self, scores):
        self._scores = scores
        self.seen = None

    def scores(self, query, documents):
        self.seen = (query, list(documents))
        return self._scores


_ITEMS = [{"path": "a"}, {"path": "b"}, {"path": "c"}]
_DOC = lambda it: it["path"]


def test_reorders_by_score_within_pool():
    out = rerank_related(_Fake([0.2, 0.1, 0.9]), "q", _ITEMS, k=3, document_of=_DOC)
    assert [i["path"] for i in out] == ["c", "a", "b"]


def test_membership_belongs_to_first_stage():
    # Reorder-only (gate 2a): the pool is truncated to k BEFORE scoring, so the
    # reranker can neither evict a first-stage top-k member nor pull one in.
    fake = _Fake([0.1, 0.9])
    out = rerank_related(fake, "q", _ITEMS, k=2, document_of=_DOC)
    assert [i["path"] for i in out] == ["b", "a"]   # reordered within top-k
    assert fake.seen[1] == ["a", "b"]               # c was never scored


def test_granularity_gate_skips_reranker_on_long_docs():
    # Gate 2b: when the median document dwarfs the cross-encoder window, its
    # ordering is noise — skip the call, keep first-stage order.
    from silica.kernel import rerank as rr_mod

    long_doc = "x" * (rr_mod._RERANK_WINDOW_FACTOR * rr_mod._WINDOW_CHARS + 1)
    fake = _Fake([0.9, 0.1])
    out = rerank_related(fake, "q", _ITEMS, k=2, document_of=lambda it: long_doc)
    assert [i["path"] for i in out] == ["a", "b"]
    assert fake.seen is None                        # cross-encoder never called


def test_granularity_gate_lets_short_docs_rerank():
    fake = _Fake([0.1, 0.9])
    out = rerank_related(fake, "q", _ITEMS, k=2, document_of=_DOC)
    assert [i["path"] for i in out] == ["b", "a"]
    assert fake.seen is not None


def test_rerank_gate_probe_receives_median_and_fired(monkeypatch):
    from silica.kernel import rerank as rr_mod

    seen = []
    monkeypatch.setattr(rr_mod, "RERANK_GATE_PROBE", seen.append)
    rerank_related(_Fake([0.1, 0.9]), "q", _ITEMS, k=2, document_of=_DOC)
    assert seen and seen[0]["fired"] is False
    assert seen[0]["median_len"] == 1               # docs are 1-char paths


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


# --- multi-window selection (multi-window spec 2026-07-15) -----------------

def test_best_windows_do_not_overlap():
    from silica.kernel.rerank import best_windows

    body = ("zebra zebra ") + ("filler " * 100) + ("zebra " * 5) + ("filler " * 100)
    wins = best_windows(body, "zebra", 60, 2)
    assert len(wins) == 2
    spans = sorted((body.find(w), body.find(w) + len(w)) for w in wins)
    assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:]))


def test_best_windows_output_in_document_order():
    from silica.kernel.rerank import best_windows

    # The denser region sits LAST in the document, so greedy picks it first;
    # the output must still follow document order (chat chronology).
    body = ("zebra zebra ") + ("filler " * 100) + ("zebra " * 5) + ("filler " * 100)
    wins = best_windows(body, "zebra", 60, 2)
    assert [w.count("zebra") for w in wins] == [2, 5]


def test_best_windows_whole_text_when_it_fits_the_budget():
    from silica.kernel.rerank import best_windows

    text = "zebra " * 30  # 180 chars <= 2 * 100
    assert best_windows(text, "zebra", 100, 2) == [text]


def test_best_windows_head_slice_without_usable_terms():
    from silica.kernel.rerank import best_windows

    text = "y" * 1000
    assert best_windows(text, "a an of it", 100, 2) == [text[:100]]


def test_best_windows_drops_second_window_without_hits():
    from silica.kernel.rerank import best_windows

    # All hits in the head: never pad with irrelevant text, fewer windows is normal.
    body = ("zebra " * 4) + ("filler " * 200)
    wins = best_windows(body, "zebra", 60, 2)
    assert len(wins) == 1
    assert wins[0].count("zebra") == 4


def test_best_windows_n1_equals_best_window():
    from silica.kernel.rerank import best_window, best_windows

    body = ("weather smalltalk " * 80) + "I attend yoga classes for my anxiety " \
           + ("unrelated filler " * 80)
    q = "how often do I attend yoga for anxiety?"
    assert best_windows(body, q, 200, 1) == [best_window(body, q, 200)]
    assert best_windows("short body", q, 800, 1) == [best_window("short body", q, 800)]
    long = "x" * 2000
    assert best_windows(long, "a an of", 500, 1) == [best_window(long, "a an of", 500)]


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


class _NoCfg:
    rerank_base_url = ""
    rerank_model = ""
    rerank_api_key = ""


def _extra(monkeypatch, present: bool):
    from silica.agent import providers as p
    monkeypatch.setattr(p, "has_local_rerank", lambda: present)


def test_get_reranker_disabled_without_config_or_extra(monkeypatch):
    _extra(monkeypatch, False)
    assert get_reranker(_NoCfg()) is None


def test_get_reranker_enabled_with_config():
    class _Cfg:
        rerank_base_url = "http://x/v1"
        rerank_model = "bge-reranker"
        rerank_api_key = "k"
    rr = get_reranker(_Cfg())
    assert isinstance(rr, Reranker) and rr.url == "http://x/v1/rerank"


def test_get_reranker_falls_back_to_local_when_extra_installed(monkeypatch):
    """Neither LM Studio nor Ollama serves cross-encoders: with the [rerank]
    extra present and no endpoint configured, rerank works with no user setup."""
    _extra(monkeypatch, True)
    rr = get_reranker(_NoCfg())
    assert isinstance(rr, LocalReranker)


def test_served_endpoint_wins_over_local(monkeypatch):
    """A configured endpoint keeps priority (the eval harness pins llama-server)."""
    _extra(monkeypatch, True)

    class _Cfg:
        rerank_base_url = "http://x/v1"
        rerank_model = "bge-reranker"
        rerank_api_key = "k"
    assert isinstance(get_reranker(_Cfg()), Reranker)


def test_local_reranker_scores_in_input_order(monkeypatch):
    from silica.agent import providers as p

    class _Encoder:
        def predict(self, pairs):
            assert pairs == [["q", "d0"], ["q", "d1"]]  # [query, doc] cross-encoder pairs
            return [0.25, 0.75]
    monkeypatch.setattr(p, "_load_cross_encoder", lambda model: _Encoder())
    assert LocalReranker(model="m").scores("q", ["d0", "d1"]) == [0.25, 0.75]


def test_local_reranker_abstains_on_error(monkeypatch):
    """Same contract as the served client: abstain, never raise, never drop."""
    from silica.agent import providers as p

    def boom(model):
        raise RuntimeError("no weights")
    monkeypatch.setattr(p, "_load_cross_encoder", boom)
    assert LocalReranker(model="m").scores("q", ["d0"]) is None


def test_local_reranker_empty_input():
    assert LocalReranker(model="m").scores("", ["d"]) is None
    assert LocalReranker(model="m").scores("q", []) is None
