# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Answer-time perception — the one assembly of recalled memory into context.

Validated on the LongMemEval perception grid (frozen corpus A, 2026-07-14):
facts-first episodic block + per-note query-densest window + rank/evidence/date
headers. The LME harness consumes perceive() directly, so the eval and the
product cannot diverge on this seam — the measured number belongs to Silica.

Kernel rule: no ``datetime.now()`` here — ``now`` is supplied by the caller
(the tool layer passes today, the eval adapter passes the simulated question
date).

Failure behavior: the episodic lane is additive and best-effort (a broken
store never blocks answering); retrieval errors propagate — a silently empty
context would score as a memory miss with no signal.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ponytail: perception-grid winners as plain defaults; promote to CONFIG only
# when a real vault needs different values.
DEFAULT_K = 15
WINDOW_CHARS = 3000
DEFAULT_WINDOWS = 1  # multi-window spec 2026-07-15: moves only after the grid decides
FACTS_K = 10


@dataclass
class NoteBlock:
    """One recalled note, ready for the prompt."""
    path: str       # store-keyspace rel path (no .md)
    date: str       # frontmatter `date`, '' when absent
    evidence: str   # joined per-leg provenance ("embed:0.83 cooccur:w9"), '' in --stuff
    body: str       # full body, frontmatter stripped
    excerpt: str    # query-densest window of the body


@dataclass
class Perception:
    """perceive()'s result: render() is the prompt string, the rest is telemetry."""
    query: str
    facts_block: str = ""
    fact_hits: list = field(default_factory=list)    # episodic.FactHit
    fact_chains: list = field(default_factory=list)  # per-hit supersede chain (episodic.Fact)
    blocks: list[NoteBlock] = field(default_factory=list)

    def render(self, *, facts_first: bool = True, windowed: bool = True) -> str:
        """The context string. Defaults are the validated perception; the flags
        exist as A/B arms for the eval harness (legacy layouts)."""
        parts: list[str] = []
        for rank, b in enumerate(self.blocks, 1):
            if windowed:
                head = f"[#{rank}" + (f" | {b.evidence}" if b.evidence else "")
                head += (f" | dated {b.date}" if b.date else "") + "]"
                parts.append(f"{head}\n{b.excerpt}")
            else:
                head = f"[dated {b.date}]\n" if b.date else ""
                parts.append(f"{head}{b.body}")
        ctx = "\n\n---\n\n".join(parts)
        if not self.facts_block or not ctx:
            return self.facts_block or ctx
        return (f"{self.facts_block}\n\n---\n\n{ctx}" if facts_first
                else f"{ctx}\n\n---\n\n{self.facts_block}")


def facade_retrieve(query: str, *, k: int, use_embedder: bool = True,
                    use_rerank: bool = True, use_recall_weights: bool = False):
    """Fused first-stage retrieval + cross-encoder rerank for a fresh text query.

    The single retrieval path shared by the chat tools
    (silica_semantic_search) and perceive() — and therefore by
    the eval adapter. Both lanes (active vault + personal memory, ADR-0019) are
    queried; a down leg abstains to the survivor.

    Returns ``(results, query_vec)``: results is the RelatedNote list ([] for
    no hits), or None when no leg is available at all (no query embedding AND
    no co-occurrence index in either lane). query_vec is surfaced for reuse —
    episodic fact recall scores against the same vector.

    ``use_recall_weights`` (phase 1 of `improve`, LoCoMo eval-only): when True,
    folds the vault's recall-outcome weights in as an extra fusion leg. False
    (the default) leaves the retrieval path byte-identical for every other
    caller.
    """
    from silica.agent.providers import get_embedder, get_reranker
    from silica.config import CONFIG
    from silica.kernel.cooccurrence import get_cooccur_store
    from silica.kernel.embed import get_store
    from silica.kernel.memory_lane import memory_stores
    from silica.kernel.relatedness import related_notes_for_query
    from silica.kernel.rerank import rerank_related

    embed_store = get_store()
    try:
        cooccur_store = get_cooccur_store(lang=CONFIG.cooccurrence_lang)
        if len(cooccur_store) == 0:
            cooccur_store = None
    except Exception:
        cooccur_store = None
    mem_embed, mem_cooccur = memory_stores()

    query_vec = None
    if use_embedder and (len(embed_store) > 0 or mem_embed is not None):
        try:
            query_vec = get_embedder(CONFIG).embed([query])[0]
        except Exception:
            query_vec = None  # embed leg abstains; co-occurrence may still carry

    if query_vec is None and cooccur_store is None and mem_cooccur is None:
        return None, None

    recall_rank = None
    if use_recall_weights:
        from silica.kernel.recall_weights import ranking

        recall_rank = ranking()

    results = related_notes_for_query(
        query_vec=query_vec,
        query_text=query,
        embed_store=embed_store,
        cooccur_store=cooccur_store,
        memory_embed_store=mem_embed,
        memory_cooccur_store=mem_cooccur,
        k=k,
        recall_rank=recall_rank,
    ) or []
    reranker = get_reranker(CONFIG) if use_rerank else None
    if reranker:
        # Default document path: gate 2b sees full body lengths, the scored
        # docs are query-densest windows, memory-lane bodies resolve by origin.
        results = rerank_related(reranker, query, results, k=k)
    return results, query_vec


def _read_dated_body(path: str, origin: str = "vault") -> tuple[str, str | None]:
    """(frontmatter date, body) for one note; ('', None) when unreadable.
    origin='memory' resolves in the personal-memory vault (ADR-0019)."""
    if origin == "memory":
        from silica.kernel.memory_lane import memory_vault

        mv = memory_vault()
        if mv is None:
            return "", None
        p = mv / (path if path.endswith(".md") else path + ".md")
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "", None
    else:
        from silica.driver import DRIVER

        try:
            content = DRIVER.read_note(
                path if path.endswith(".md") else path + ".md").content or ""
        except Exception:
            return "", None
    from silica.kernel import frontmatter

    data, _raw, body = frontmatter.split(content)
    # data is None for a body-only note (no frontmatter) or a YAML error —
    # product notes from the FSM write path can lack frontmatter entirely.
    date = str((data or {}).get("date") or "").strip()
    return date, (body or content)


def _recall_facts(perception: Perception, query: str, query_vec, *, now: str,
                  facts_k: int, episodic_ttl_days: int | None,
                  use_embedder: bool) -> None:
    """Fill the Personal-memory side of `perception`. Best-effort: additive
    evidence must never block answering (mirror of capture_from_distill)."""
    try:
        from silica.kernel.episodic import EpisodicStore, render as render_facts

        store = EpisodicStore()
        if not store.live_facts():
            return
        if query_vec is None and use_embedder:
            try:
                from silica.agent.providers import get_embedder
                from silica.config import CONFIG

                query_vec = get_embedder(CONFIG).embed([query])[0]
            except Exception:
                query_vec = None  # lexical fact recall
        hits = store.recall(query, query_vec, k=facts_k, now=now,
                            ttl_days=episodic_ttl_days)
        if not hits:
            return
        perception.fact_hits = hits
        perception.fact_chains = [store.chain(h.fact) for h in hits]
        perception.facts_block = "Personal memory:\n" + render_facts(hits, store=store)
    except Exception as e:
        logger.warning("perceive: episodic recall failed (context continues): %s", e)


def perceive(query: str, *, now: str, k: int = DEFAULT_K,
             window_chars: int = WINDOW_CHARS, windows: int = DEFAULT_WINDOWS,
             facts_k: int = FACTS_K,
             episodic_ttl_days: int | None = None, with_facts: bool = True,
             use_embedder: bool = True, use_rerank: bool = True,
             paths: list[str] | None = None,
             use_recall_weights: bool = False) -> Perception:
    """Retrieve + assemble the answer-time context for `query`.

    ``paths`` skips retrieval and assembles the given notes in order (the eval
    adapter's --stuff arm, or a caller that already holds a shortlist);
    unreadable paths are skipped and ranks stay dense. ``episodic_ttl_days``:
    None = CONFIG default, 0 = never expire. ``use_recall_weights`` (phase 1 of
    `improve`, eval-only, default off) is forwarded to `facade_retrieve`; it
    has no effect when ``paths`` is set, since that bypasses retrieval.
    """
    from silica.kernel.rerank import best_windows

    query_vec = None
    if paths is not None:
        hits = [(p, "", "vault") for p in paths]
    else:
        results, query_vec = facade_retrieve(
            query, k=k, use_embedder=use_embedder, use_rerank=use_rerank,
            use_recall_weights=use_recall_weights)
        hits = [(r.path, " ".join(r.evidence), getattr(r, "origin", "vault"))
                for r in (results or [])]

    blocks: list[NoteBlock] = []
    for path, evidence, origin in hits:
        date, body = _read_dated_body(path, origin)
        if body is None:
            continue
        excerpt = ("\n[…]\n".join(best_windows(body, query, window_chars, windows))
                   if query else body[:window_chars])
        blocks.append(NoteBlock(path=path, date=date, evidence=evidence,
                                body=body, excerpt=excerpt))

    perception = Perception(query=query, blocks=blocks)
    if with_facts:
        _recall_facts(perception, query, query_vec, now=now, facts_k=facts_k,
                      episodic_ttl_days=episodic_ttl_days, use_embedder=use_embedder)
    return perception
