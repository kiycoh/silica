# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""MuSiQue retrieval benchmark adapter — load, index, retrieval-only probe.

Measures the relatedness facade (RRF fusion over the embed + co-occurrence
legs) against MuSiQue gold supporting passages. Run on the pooled dev corpus
released by HippoRAG (data/musique_corpus.json + data/musique.json, 11,654
passages / 1,000 questions) the recall@k numbers are comparable 1:1 with the
tables published for HippoRAG / HippoRAG 2.

Protocol (closed corpus): the benchmark provides every source passage; the
probe never touches the web and never passes the personal-memory lane stores,
so nothing outside the bench vault can leak into the ranking.

  1. load  — one verbatim note per passage via ``commit_derived`` (the
             derived-artifact write channel: ground truth lives outside the
             vault, no LLM rewrite, no nucleate validators)
  2. index — bulk ``silica_embed_refresh`` + ``silica_cooccurrence_refresh``
  3. probe — ``related_notes_for_query(question)`` → recall@k / MRR against
             the ``is_supporting`` paragraphs

  uv run python -m tests.eval.musique --vault bench/musique \
      --corpus musique_corpus.json --questions musique.json --load --index
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
from pathlib import Path

_PID = re.compile(r"^p(\d+)$")
_KS = (2, 5, 10)
_EMBED_BATCH = 32
METRICS_PATH = Path(__file__).parent / "metrics.json"


# ---------------------------------------------------------------------------
# corpus normalization
# ---------------------------------------------------------------------------

def _text_of(item: dict) -> str:
    return (item.get("text") or item.get("paragraph_text") or "").strip()


def _key(item: dict) -> tuple[str, str]:
    """Identity of a passage across corpus and question files: (title, text)."""
    return ((item.get("title") or "").strip(), _text_of(item))


def _rel(idx: int) -> str:
    return f"corpus/p{idx:05d}.md"


def _pid_of(note_path: str) -> int | None:
    """Inverse of ``_rel`` for facade results (index keys carry no .md)."""
    m = _PID.match(Path(note_path).stem)
    return int(m.group(1)) if m else None


def _note_content(idx: int, item: dict) -> str:
    title = (item.get("title") or "").strip()
    return (
        "---\n"
        f"passage_id: {idx}\n"
        f"title: {json.dumps(title, ensure_ascii=False)}\n"
        "source: musique\n"
        "tags:\n  - benchmark\n"
        "---\n\n"
        f"{_text_of(item)}\n"
    )


# ---------------------------------------------------------------------------
# vault binding + steps
# ---------------------------------------------------------------------------

def bind_vault(vault: Path) -> None:
    """Point CONFIG/DRIVER at the bench vault and drop store singletons, so a
    prior in-process vault can never leak its indexes into the run."""
    import silica.driver
    import silica.kernel.cooccurrence as cooc_mod
    import silica.kernel.embed as embed_mod
    from silica.config import CONFIG

    CONFIG.vault_path = str(vault)
    CONFIG.backend = "fs"
    silica.driver._driver = None
    embed_mod.clear()
    cooc_mod.clear()


def load_corpus(corpus: list[dict], *, verbose: bool = False) -> dict:
    """Write one verbatim note per passage. Idempotent: existing note files
    are skipped, so a partial load can simply be rerun."""
    from silica.agent.commit import commit_derived
    from silica.config import CONFIG

    vault = Path(CONFIG.vault_path)
    committed = skipped = 0
    failures: list[dict] = []
    for idx, item in enumerate(corpus):
        rel = _rel(idx)
        if (vault / rel).exists():
            skipped += 1
            continue
        res = commit_derived(rel, _note_content(idx, item))
        if res.get("status") == "committed":
            committed += 1
        else:
            failures.append({"rel": rel, **res})
        if verbose and (idx + 1) % 1000 == 0:
            print(f"  load {idx + 1}/{len(corpus)}")
    if failures:
        print(f"load: {len(failures)} passages rejected (first: {failures[0]})")
    return {"committed": committed, "skipped": skipped, "failures": failures}


def build_indexes(*, embed: bool = True, force: bool = False) -> dict:
    from silica.tools.graph import silica_cooccurrence_refresh, silica_embed_refresh

    out: dict = {"cooccur": silica_cooccurrence_refresh(force=force)}
    if embed:
        out["embed"] = silica_embed_refresh(force=force)
    for leg, res in out.items():
        if "error" in res:
            print(f"index[{leg}]: {res['error']}")
    return out


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

def _recall(gold: set[int], ranked: list[int], k: int) -> float:
    return len(gold & set(ranked[:k])) / len(gold)


def _first_gold_rank(gold: set[int], ranked: list[int]) -> int | None:
    """1-based rank of the first gold passage in the fused ranking."""
    for i, pid in enumerate(ranked, start=1):
        if pid in gold:
            return i
    return None


def _embed_queries(texts: list[str]) -> list[list[float]] | None:
    from silica.agent.providers import get_embedder
    from silica.config import CONFIG

    emb = get_embedder(CONFIG)
    vecs: list[list[float]] = []
    try:
        for i in range(0, len(texts), _EMBED_BATCH):
            vecs.extend(emb.embed(texts[i : i + _EMBED_BATCH]))
    except Exception as e:
        print(f"query embedder unavailable ({e}) — embed leg abstains")
        return None
    return vecs


def probe(
    questions: list[dict],
    corpus: list[dict],
    *,
    k: int = 10,
    use_embedder: bool = True,
    use_cooccur: bool = True,
    use_rerank: bool = True,
    expand: bool = False,
    limit: int | None = None,
    verbose: bool = False,
) -> dict:
    """Retrieval-only evaluation: fused top-k per question vs gold supporting
    passages. Memory-lane stores are deliberately never passed (ADR-0019 lane
    stays out of the benchmark).

    When a reranker is configured (CONFIG.rerank_*), the cross-encoder reorders
    the fused first-stage top-k (reorder-only + granularity gate live in
    rerank_related; membership always belongs to the first stage)."""
    from silica.agent.providers import get_reranker
    from silica.config import CONFIG
    from silica.kernel.cooccurrence import get_cooccur_store
    from silica.kernel.embed import get_store
    from silica.kernel.relatedness import related_notes_for_query
    from silica.kernel.rerank import rerank_related

    key2idx: dict[tuple[str, str], int] = {}
    for i, item in enumerate(corpus):
        key2idx.setdefault(_key(item), i)

    cooccur_store = None
    if use_cooccur:
        cooccur_store = get_cooccur_store(lang=CONFIG.cooccurrence_lang)
        if len(cooccur_store) == 0:
            cooccur_store = None
    embed_store = get_store()
    if len(embed_store) == 0:
        embed_store = None

    questions = questions[:limit] if limit else questions
    qvecs = None
    if use_embedder and embed_store is not None:
        qvecs = _embed_queries([q["question"] for q in questions])

    legs = [name for name, s in (("embed", embed_store if qvecs else None),
                                 ("cooccur", cooccur_store)) if s is not None]
    reranker = get_reranker(CONFIG) if use_rerank else None
    ks = tuple(x for x in _KS if x <= k)
    rows: list[dict] = []
    skipped = unmappable = 0

    for qi, q in enumerate(questions):
        gold: set[int] = set()
        for para in q.get("paragraphs", []):
            if not para.get("is_supporting"):
                continue
            idx = key2idx.get(_key(para))
            if idx is None:
                unmappable += 1
            else:
                gold.add(idx)
        if not gold:
            skipped += 1
            continue

        related = related_notes_for_query(
            query_text=q["question"],
            query_vec=qvecs[qi] if qvecs else None,
            embed_store=embed_store if qvecs else None,
            cooccur_store=cooccur_store,
            k=k,
            expand=expand,  # 1-hop cooccur concept-neighbour reach (bridge-passage recall probe)
        )
        if reranker:
            related = rerank_related(reranker, q["question"], related, k=k)
        ranked = [pid for r in related if (pid := _pid_of(r.path)) is not None]
        row = {
            "id": q.get("id") or q.get("_id") or str(qi),
            "gold": sorted(gold),
            "top": ranked,
            "first_gold_rank": _first_gold_rank(gold, ranked),
        }
        row["hop"] = row["id"].split("__")[0] if "hop" in row["id"] else "?"
        for kk in ks:
            row[f"recall_at_{kk}"] = _recall(gold, ranked, kk)
        rows.append(row)
        if verbose:
            print(f"  {row['id']}: r@{ks[-1]}={row[f'recall_at_{ks[-1]}']:.2f} "
                  f"gold={row['gold']} top={ranked[:5]}")

    n = len(rows)
    metrics: dict = {
        "questions_evaluated": n,
        "questions_skipped": skipped,
        "unmappable_gold": unmappable,
    }
    for kk in ks:
        metrics[f"recall_at_{kk}"] = round(
            sum(r[f"recall_at_{kk}"] for r in rows) / n, 4) if n else 0.0
    metrics["mrr"] = round(
        sum(1.0 / r["first_gold_rank"] for r in rows if r["first_gold_rank"]) / n,
        4) if n else 0.0
    per_hop: dict[str, dict] = {}
    for hop in sorted({r["hop"] for r in rows}):
        hr = [r for r in rows if r["hop"] == hop]
        per_hop[hop] = {"n": len(hr)}
        for kk in ks:
            per_hop[hop][f"recall_at_{kk}"] = round(
                sum(r[f"recall_at_{kk}"] for r in hr) / len(hr), 4)
    metrics["per_hop"] = per_hop

    return {
        "generated_at": datetime.date.today().isoformat(),
        "benchmark": "musique",
        "vault": {"path": CONFIG.vault_path,
                  "corpus_notes": len(list((Path(CONFIG.vault_path) / "corpus").glob("*.md")))},
        "config": {
            "legs": "+".join(legs) or "none",
            "k": k,
            "embedding_model": getattr(CONFIG, "embedding_model", None) if qvecs else None,
            "cooccur_lang": getattr(cooccur_store, "lang", None),
            "rerank_model": getattr(CONFIG, "rerank_model", None) or None if reranker else None,
        },
        "metrics": metrics,
        "questions": rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(doc: dict) -> None:
    m, cfg = doc["metrics"], doc["config"]
    print(f"\nmusique — {m['questions_evaluated']} questions "
          f"({m['questions_skipped']} skipped, {m['unmappable_gold']} unmappable gold), "
          f"legs: {cfg['legs']}, corpus notes: {doc['vault']['corpus_notes']}")
    parts = [f"recall@{k} {m[f'recall_at_{k}']:.3f}"
             for k in _KS if f"recall_at_{k}" in m]
    print("  " + "  ".join(parts) + f"  mrr {m['mrr']:.3f}")
    for hop, hm in m["per_hop"].items():
        hop_parts = [f"r@{k} {hm[f'recall_at_{k}']:.3f}"
                     for k in _KS if f"recall_at_{k}" in hm]
        print(f"  {hop:<6} n={hm['n']:<5} " + "  ".join(hop_parts))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tests.eval.musique")
    ap.add_argument("--vault", required=True, help="bench vault directory (dedicated, not your personal vault)")
    ap.add_argument("--corpus", required=True, help="musique_corpus.json (HippoRAG pooled dev corpus)")
    ap.add_argument("--questions", help="musique.json (HippoRAG 1k dev questions); omit to only load/index")
    ap.add_argument("--load", action="store_true", help="write the corpus into the vault first")
    ap.add_argument("--index", action="store_true", help="build/refresh embed + cooccur indexes")
    ap.add_argument("--force", action="store_true", help="force full index rebuild")
    ap.add_argument("--no-embed", action="store_true", help="cooccur leg only (no embedder anywhere)")
    ap.add_argument("--no-cooccur", action="store_true", help="embed leg only (no co-occurrence leg)")
    ap.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder rerank pass")
    ap.add_argument("--expand", action="store_true", help="cooccur leg: add 1-hop concept neighbours (bridge-passage probe)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, help="evaluate only the first N questions")
    ap.add_argument("--out", help=f"report path (default {METRICS_PATH})")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    vault = Path(args.vault).expanduser().resolve()
    if args.load:
        vault.mkdir(parents=True, exist_ok=True)
    if not vault.is_dir():
        print(f"vault not found: {vault} (pass --load to create it)")
        return 2
    bind_vault(vault)

    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    if args.load:
        summary = load_corpus(corpus, verbose=args.verbose)
        print(f"load: committed {summary['committed']}, skipped {summary['skipped']}, "
              f"failed {len(summary['failures'])}")
    if args.index:
        for leg, res in build_indexes(embed=not args.no_embed, force=args.force).items():
            if "error" not in res:
                print(f"index[{leg}]: {res['indexed']} notes @ {res['index_path']}")

    if not args.questions:
        return 0
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    doc = probe(questions, corpus, k=args.k, use_embedder=not args.no_embed,
                use_cooccur=not args.no_cooccur, use_rerank=not args.no_rerank,
                expand=args.expand, limit=args.limit, verbose=args.verbose)
    out = Path(args.out) if args.out else METRICS_PATH
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print_summary(doc)
    print(f"\nreport → {out}")
    return 0
