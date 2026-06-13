"""Embedding-based PROPOSED signals: missing links and duplicate pairs.

Both functions degrade to [] when the embedding index is empty or the
embedder is unavailable. Lazy imports into silica.agent.providers and
silica.config are a known, pre-existing kernel impurity (see the kernel
import-linter contract note in pyproject.toml).
"""
from __future__ import annotations

import logging
import time
from typing import Any

from silica.kernel.graph_report.models import DuplicatePair, MissingLink, VaultReport

logger = logging.getLogger(__name__)

# Temporal decay half-life: a note updated 30 days ago gets ~50% boost; at
# 90 days the boost is negligible.  The constant is chosen so the recency
# factor degrades smoothly without overwhelming the cosine signal.
_RECENCY_HALFLIFE_DAYS = 30.0


def _compute_missing_links(
    report: VaultReport,
    G_und: Any,
    *,
    tau: float = 0.82,
    k: int = 10,
) -> list[MissingLink]:
    """Propose missing links via embedding similarity (PROPOSED — not authoritative).

    Paper-inspired refinements (Marwitz et al. 2026):
      - common_neighbors: a structural boost from the 2-length path count, the
        paper's Baseline core feature — likelier links rank above structurally
        isolated but equally-similar pairs.
      - d_prev annotation: each result carries its shortest-path distance before
        prediction. Only direct neighbours (d<=1) are hard-gated; d=2 candidates
        (likely links) and d>=3 (novel, high creative value) both surface.
      - Temporal decay: recent note pairs receive a modest cosine boost based on
        EmbedStore timestamps, capturing the paper's velocity-of-growth signal.
    """
    try:
        from silica.agent.providers import get_embedder
        from silica.config import CONFIG
        from silica.kernel.embed import EmbedStore
        import networkx as nx

        store = EmbedStore()
        if len(store) == 0:
            return []
        embedder = get_embedder(CONFIG)
    except Exception as exc:
        logger.debug("graph_report: embeddings unavailable (%s)", exc)
        return []

    now = time.time()
    god_paths = [n.id for n in report.god_nodes]
    results: list[MissingLink] = []
    seen: set[tuple[str, str]] = set()

    for source in god_paths:
        vec = store.get_vec(source)
        if vec is None:
            vec = store.get_vec(source.removesuffix(".md"))
        if vec is None:
            continue
        try:
            candidates = store.cosine_top_k(vec, k=k, exclude={source})
        except Exception:
            continue

        for cand in candidates:
            tgt = cand["path"]
            score = cand.get("score", 0.0)
            if score < tau:
                break  # results are sorted desc
            if tgt not in G_und or source not in G_und:
                continue

            # --- #7 d_prev: annotate instead of hard-gating at d<=2 ----------
            try:
                d_prev = nx.shortest_path_length(G_und, source, tgt)
            except nx.NetworkXNoPath:
                d_prev = 0  # unreachable → highest novelty
            if d_prev <= 1:
                continue  # only skip direct neighbours

            # --- #2 common_neighbors: structural-likelihood boost ------------
            # Paper Baseline uses sum_i A^2_u,i (2-length path count) as a core
            # feature; more shared neighbours → likelier real link. Mapped to
            # [0, 1) for diminishing returns so it nudges the ranking without
            # overwhelming the cosine signal.
            cn = len(list(nx.common_neighbors(G_und, source, tgt)))
            structural = cn / (1.0 + cn)

            # --- #5 temporal decay: boost recent note pairs ------------------
            ts_src = store._notes.get(source, {}).get("ts", 0)
            ts_tgt = store._notes.get(tgt, {}).get("ts", 0)
            age_days = max(0.0, (now - max(ts_src, ts_tgt)) / 86400.0)
            recency = 2.0 ** (-age_days / _RECENCY_HALFLIFE_DAYS)  # [0, 1]
            adjusted = score * (1.0 + 0.3 * structural) * (1.0 + 0.1 * recency)

            key = (min(source, tgt), max(source, tgt))
            if key not in seen:
                seen.add(key)
                results.append(MissingLink(
                    source=source, target=tgt,
                    cosine=round(adjusted, 4),
                    d_prev=d_prev,
                ))

    results.sort(key=lambda m: (-m.cosine, m.source, m.target))
    return results[:k]


def _compute_duplicate_pairs(report: VaultReport) -> list[DuplicatePair]:
    """Find near-duplicate note pairs using embedding index (PROPOSED — not authoritative)."""
    try:
        from silica.config import CONFIG
        from silica.kernel.embed import EmbedStore

        store = EmbedStore()
        if len(store) == 0:
            return []
    except Exception as exc:
        logger.debug("graph_report: embeddings unavailable for dedup (%s)", exc)
        return []

    tau_high = getattr(CONFIG, "sim_threshold_high", 0.85)
    tau_low = getattr(CONFIG, "sim_threshold_low", 0.65)

    results: list[DuplicatePair] = []
    seen: set[tuple[str, str]] = set()

    def _in_folder(path: str, folder: str) -> bool:
        if not folder:
            return True
        f = folder.replace("\\", "/").strip("/").lower()
        p = path.replace("\\", "/").removesuffix(".md").lower()
        return p == f or p.startswith(f + "/")

    scope = [p for p in store.paths() if _in_folder(p, report.scope)]

    for p in scope:
        vec = store.get_vec(p)
        if not vec:
            continue
        try:
            candidates = store.cosine_top_k(vec, k=1, exclude={p})
        except Exception:
            continue

        if not candidates:
            continue

        cand = candidates[0]
        tgt = cand["path"]
        score = cand.get("score", 0.0)

        if tau_low < score < tau_high:
            key = (min(p, tgt), max(p, tgt))
            if key not in seen:
                seen.add(key)
                results.append(DuplicatePair(source=p, target=tgt, score=round(score, 4)))

    results.sort(key=lambda d: (-d.score, d.source, d.target))
    return results
