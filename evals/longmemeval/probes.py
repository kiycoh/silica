# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Key-drift probes over frozen episodic stores (read-only, zero LLM).

Two probes with identical mechanics, differing only in how keys group:
  aggregative (question_type == "multi-session"): group gold-session facts by
    2-segment key prefix — does one category share a namespace, or scatter?
  knowledge-update: group by FULL key — supersede chains link only on
    identical keys, so scattered keys mean broken chains.

Per question: capture ceiling (gold sessions with >= 1 fact) and best-group
coverage (gold sessions covered by the single best key group). The 2026-07-15
aggregative probe showed capture at ceiling but best-prefix coverage 1/N:
key-drift, not retrieval, is the blocker.

Session ids are compared VERBATIM: `answer_...` prefixes are part of the id
(they appear as-is in haystack_session_ids and in fact runs). Never strip.

The `--embed-tau` view replays capture in true id order and lets a fact that
matches no live head by canonical key join the nearest live head by cosine
(>= tau): the chains a capture-side semantic matcher WOULD have formed.
Zero LLM; embeddings come from stored Fact.vec or a sidecar cache
(probe_vecs.json) filled once per vault.

CLI:
  uv run python -m evals.longmemeval.probes \
      --data bench/lme_mixed18.json --run-root bench/lme18_hyb
  # capture-side identity sweep:
  ... --embed-tau 0.70,0.75,0.80,0.85 --embed-repr key+text
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

PROBED_TYPES = ("multi-session", "knowledge-update")


def key_prefix(key: str, n: int = 2) -> str:
    return ".".join(key.split(".")[:n])


def cluster_keys(keys: list[str], *, max_df: int | None = None) -> dict[str, str]:
    """Post-hoc key clustering: connected components over shared stemmed tokens.

    Counterfactual view — what a mechanical clustering layer COULD merge that
    prompt-side key discipline provably does not (2026-07-15: three prompt
    exhortations, three misses). Transitive gluing is deliberate; best_size
    in the probe row is the blob check. max_df=K keeps only tokens shared by
    <= K keys (rare-token linkage; 2026-07-15 sweep: K=3 shatters the blob,
    frozen corpus covers 15/17 gold sessions at cluster sizes 2-10). Returns
    key -> component display name (lexicographically first member key,
    `(+N)` suffix for the rest)."""
    from silica.kernel.episodic import rare_token_components

    comp = rare_token_components(keys, max_df=max_df)
    members: dict[str, list[str]] = defaultdict(list)
    for k, root in comp.items():
        members[root].append(k)
    out: dict[str, str] = {}
    for group in members.values():
        first = min(group)
        name = first if len(group) == 1 else f"{first} (+{len(group) - 1})"
        for k in group:
            out[k] = name
    return out


def capture_sim(facts: list[dict], vecs: dict[str, list[float]], *,
                tau: float) -> tuple[dict[str, str], dict]:
    """Counterfactual capture replay: fact id -> chain root id.

    Mirrors EpisodicStore.capture's head lookup in true capture order
    (numeric id sequence), with one added arm: a fact matching no live head
    by canonical key joins the nearest live head by cosine when >= tau.
    Stats: `embed_joins` and every fallback decision as (nearest_cos, joined)
    — the tau calibration curve. Pure function of (facts, vecs, tau)."""
    from silica.kernel.episodic import _cosine, normalize_key

    by_nkey: dict[str, str] = {}   # nkey -> live head id (arm 1)
    live: dict[str, str] = {}      # live head id -> chain root id
    nkey_of: dict[str, str] = {}
    roots: dict[str, str] = {}
    cosines: list[tuple[float, bool]] = []
    joins = 0
    for f in sorted(facts, key=lambda x: int(x["id"].split("_")[-1])):
        fid, nkey = f["id"], normalize_key(f["key"])
        head = by_nkey.get(nkey)
        if head is None and (v := vecs.get(fid)):
            best, best_cos = None, 0.0
            for hid in live:
                hv = vecs.get(hid)
                c = _cosine(v, hv) if hv else 0.0
                if c > best_cos:
                    best, best_cos = hid, c
            if best is not None:
                joined = best_cos >= tau
                cosines.append((best_cos, joined))
                if joined:
                    head, joins = best, joins + 1
        if head is None:
            roots[fid] = live[fid] = fid
        else:
            root = live.pop(head)          # superseded head retires
            old = nkey_of.pop(head)
            if by_nkey.get(old) == head:
                del by_nkey[old]
            roots[fid] = live[fid] = root
        by_nkey[nkey] = fid
        nkey_of[fid] = nkey
    return roots, {"embed_joins": joins, "cosines": cosines}


def embed_groups(facts: list[dict], vecs: dict[str, list[float]], *,
                 tau: float) -> tuple[dict[str, str], dict]:
    """fact id -> group display name via capture_sim chains (naming mirrors
    product_groups: root key, `(+N)` suffix, `#id` on name collisions)."""
    roots, stats = capture_sim(facts, vecs, tau=tau)
    members: dict[str, list[str]] = defaultdict(list)
    for fid, root in roots.items():
        members[root].append(fid)
    by_id = {f["id"]: f for f in facts}
    names: dict[str, str] = {}
    claimed: dict[str, str] = {}
    for root, ids in members.items():
        if len(ids) == 1:
            names[ids[0]] = by_id[ids[0]]["key"]
            continue
        name = f"{by_id[root]['key']} (+{len(ids) - 1})"
        if claimed.setdefault(name, root) != root:
            name = f"{name} #{root}"
        for fid in ids:
            names[fid] = name
    return names, stats


def sim_vecs(facts: list[dict], repr_: str, vault: Path,
             embedder) -> dict[str, list[float]]:
    """fact id -> vec for capture_sim. repr `text` reuses stored Fact.vec;
    other reprs (and facts missing a stored vec) embed through `embedder`
    behind a sidecar cache (probe_vecs.json next to episodic.json), so
    reruns are deterministic and each content embeds exactly once. Raises
    when vectors are needed and no embedder is available — silent
    degradation would read as "no signal"."""
    import hashlib

    from silica.kernel.paths import index_dir_for

    def content(f: dict) -> str:
        if repr_ == "text":
            return f["text"]
        spaced = f["key"].replace(".", " ").replace("_", " ")
        return spaced if repr_ == "key" else f"{spaced}: {f['text']}"

    def ckey(f: dict) -> str:
        return hashlib.sha1(f"{repr_}\x00{content(f)}".encode()).hexdigest()

    vecs: dict[str, list[float]] = {}
    missing: list[dict] = []
    for f in facts:
        if repr_ == "text" and f.get("vec"):
            vecs[f["id"]] = f["vec"]
        else:
            missing.append(f)
    if not missing:
        return vecs
    cache_path = index_dir_for(str(vault)) / "probe_vecs.json"
    cache: dict[str, list[float]] = (
        json.loads(cache_path.read_text(encoding="utf-8"))
        if cache_path.is_file() else {})
    to_embed = [f for f in missing if ckey(f) not in cache]
    if to_embed:
        if embedder is None:
            raise RuntimeError(
                f"{len(to_embed)} fact(s) need embedding (repr={repr_}) "
                "and no embedder is available")
        for f, vec in zip(to_embed,
                          embedder.embed([content(f) for f in to_embed])):
            cache[ckey(f)] = list(vec)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    for f in missing:
        vecs[f["id"]] = cache[ckey(f)]
    return vecs


# Grouping policy of the reverted Layer C rev 2, kept HERE as a diagnostic
# view only (probe-validated: df<=3 covers 15/17 gold at sizes 2-10). The
# product writes no groups; this recomputes them in memory for inspection.
_PROBE_DF = 3
_PROBE_MAX_GROUP = 12


def product_groups(live: list[dict]) -> dict[str, str]:
    """fact id -> group display name via the (former) product regroup rule
    applied in memory to the live facts (a pure function of the key set)."""
    from silica.kernel.episodic import rare_token_components

    comp = rare_token_components([f["key"] for f in live], max_df=_PROBE_DF)
    members: dict[str, list[dict]] = defaultdict(list)
    for f in live:
        members[comp[f["key"]]].append(f)
    names: dict[str, str] = {}
    claimed: dict[str, str] = {}
    for group in members.values():
        grouped = 2 <= len(group) <= _PROBE_MAX_GROUP
        gid = min(f["id"] for f in group)
        by_id = {f["id"]: f for f in group}
        for f in group:
            if not grouped:
                names[f["id"]] = f["key"]
                continue
            name = f"{by_id[gid]['key']} (+{len(group) - 1})"
            if claimed.setdefault(name, gid) != gid:
                name = f"{name} #{gid}"
            names[f["id"]] = name
    return names


def _load_facts(vault: Path) -> list[dict]:
    from silica.kernel.paths import index_dir_for

    path = index_dir_for(str(vault)) / "episodic.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("facts", [])


def _load_live_facts(vault: Path) -> list[dict]:
    return [f for f in _load_facts(vault) if f.get("status") == "live"]


def probe_question(inst: dict, run_root: Path, *, normalize: bool = False,
                   cluster: bool = False, max_df: int | None = None,
                   product: bool = False, embed_tau: float | None = None,
                   embed_repr: str = "text", embedder=None) -> dict:
    """Probe one question's frozen store; returns a flat metrics dict.

    normalize=True groups keys in their canonical (Layer A) form — the
    store's effective key identity, since capture matches normalized.
    cluster=True groups by post-hoc token clustering instead (both types):
    the ceiling a mechanical clustering layer could reach on this store.
    product=True groups by the PRODUCT regroup rule applied to the live facts.
    embed_tau=t groups by the capture-order embedding-fallback simulation:
    the chains a capture-side semantic matcher WOULD have formed at cosine
    threshold t (row gains embed_joins + cosines)."""
    from silica.kernel.episodic import normalize_key
    from evals.longmemeval.runner import question_vault

    qid = inst["question_id"]
    qtype = inst["question_type"]
    gold = set(inst["answer_session_ids"])
    vault = question_vault(run_root, qid)
    live = _load_live_facts(vault)

    covered = {g for f in live for g in f["runs"] if g in gold}
    gold_facts = [f for f in live if gold & set(f["runs"])]

    embed_stats: dict | None = None
    if embed_tau is not None:
        facts = _load_facts(vault)
        gmap, embed_stats = embed_groups(
            facts, sim_vecs(facts, embed_repr, vault, embedder),
            tau=embed_tau)
        group_of = lambda f: gmap[f["id"]]  # noqa: E731
    elif product:
        gmap = product_groups(live)
        group_of = lambda f: gmap[f["id"]]  # noqa: E731
    elif cluster:
        components = cluster_keys(sorted({f["key"] for f in live}),
                                  max_df=max_df)
        group_of = lambda f: components[f["key"]]  # noqa: E731
    else:
        canon = normalize_key if normalize else (lambda k: k)
        if qtype == "knowledge-update":
            group_of = lambda f: canon(f["key"])  # noqa: E731
        else:
            group_of = lambda f: key_prefix(canon(f["key"]))  # noqa: E731
    by_group: dict[str, set[str]] = defaultdict(set)
    for f in gold_facts:
        by_group[group_of(f)] |= gold & set(f["runs"])
    sizes: dict[str, int] = defaultdict(int)
    for f in live:
        sizes[group_of(f)] += 1
    # Ties on coverage go to the SMALLEST group: the honest diagnostic when a
    # tiny precise cluster and a blob cover the same gold sessions.
    best_group, best_cov = max(by_group.items(),
                               key=lambda kv: (len(kv[1]), -sizes[kv[0]]),
                               default=("-", set()))
    row = {
        "question_id": qid,
        "question_type": qtype,
        "gold_sessions": len(gold),
        "captured_sessions": len(covered),
        "gold_facts": len(gold_facts),
        "groups": len(by_group),
        "best_group": best_group,
        "best_coverage": len(best_cov),
        # Blob check: LIVE facts (gold or not) riding in the best group.
        "best_size": sizes.get(best_group, 0),
        "group_coverage": {g: sorted(c) for g, c in
                           sorted(by_group.items(), key=lambda kv: -len(kv[1]))},
    }
    if embed_stats is not None:
        row["embed_joins"] = embed_stats["embed_joins"]
        row["cosines"] = embed_stats["cosines"]
    return row


def run_probes(data: list[dict], run_root: Path, *, normalize: bool = False,
               cluster: bool = False, max_df: int | None = None,
               product: bool = False, embed_tau: float | None = None,
               embed_repr: str = "text", embedder=None) -> list[dict]:
    return [probe_question(q, run_root, normalize=normalize, cluster=cluster,
                           max_df=max_df, product=product, embed_tau=embed_tau,
                           embed_repr=embed_repr, embedder=embedder)
            for q in data if q["question_type"] in PROBED_TYPES]


def render(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        joins = (f"  joins {r['embed_joins']}"
                 if "embed_joins" in r else "")
        lines.append(
            f"{r['question_id']:<16} {r['question_type']:<18} "
            f"capture {r['captured_sessions']}/{r['gold_sessions']}  "
            f"groups {r['groups']:>2}  best '{r['best_group']}' "
            f"covers {r['best_coverage']}/{r['gold_sessions']} "
            f"(size {r['best_size']}){joins}")
        for g, cov in list(r["group_coverage"].items())[:8]:
            lines.append(f"    {g:<46} {len(cov)}")
    return "\n".join(lines)


def cosine_summary(rows: list[dict]) -> str:
    """Fallback-decision distribution across all rows: is tau on a cliff
    (joins and rejects well separated) or in a soup?"""
    def q(xs: list[float]) -> str:
        if not xs:
            return "n=0"
        return (f"n={len(xs)} min={xs[0]:.3f} "
                f"med={xs[len(xs) // 2]:.3f} max={xs[-1]:.3f}")

    joined = sorted(c for r in rows for c, j in r.get("cosines", []) if j)
    rejected = sorted(c for r in rows for c, j in r.get("cosines", []) if not j)
    return (f"fallback joins:   {q(joined)}\n"
            f"fallback rejects: {q(rejected)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--normalize", action="store_true",
                    help="group keys in canonical (Layer A) form")
    ap.add_argument("--cluster", action="store_true",
                    help="group keys by post-hoc token clustering (ceiling view)")
    ap.add_argument("--max-df", type=int, default=None,
                    help="cluster only on tokens shared by <= K keys")
    ap.add_argument("--product", action="store_true",
                    help="group by the former product regroup rule (in-memory view)")
    ap.add_argument("--embed-tau", default=None,
                    help="capture-sim embedding fallback at these cosine "
                         "thresholds (comma-separated sweep, e.g. 0.7,0.8)")
    ap.add_argument("--embed-repr", default="text",
                    choices=("text", "key", "key+text"),
                    help="what the fallback embeds (default: text, reuses "
                         "stored Fact.vec)")
    args = ap.parse_args()
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    run_root = Path(args.run_root).expanduser().resolve()
    if args.embed_tau:
        embedder = None
        try:
            from silica.agent.providers import get_embedder
            from silica.config import CONFIG

            embedder = get_embedder(CONFIG)
        except Exception:
            pass  # sim_vecs raises only if vectors are actually needed
        for tau in (float(t) for t in args.embed_tau.split(",")):
            rows = run_probes(data, run_root, embed_tau=tau,
                              embed_repr=args.embed_repr, embedder=embedder)
            print(f"== embed view: repr={args.embed_repr} tau={tau} ==")
            print(render(rows))
            print(cosine_summary(rows))
        return
    print(render(run_probes(data, run_root,
                            normalize=args.normalize, cluster=args.cluster,
                            max_df=args.max_df, product=args.product)))


if __name__ == "__main__":
    main()
