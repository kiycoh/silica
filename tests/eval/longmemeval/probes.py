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

CLI:
  uv run python -m tests.eval.longmemeval.probes \
      --data bench/lme_mixed18.json --run-root bench/lme18_hyb
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
    from silica.kernel.episodic import key_tokens

    toks = {k: key_tokens(k) for k in keys}
    if max_df is not None:
        df: dict[str, int] = defaultdict(int)
        for ts in toks.values():
            for t in ts:
                df[t] += 1
        toks = {k: {t for t in ts if df[t] <= max_df} for k, ts in toks.items()}
    parent: dict[str, str] = {k: k for k in keys}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    owner: dict[str, str] = {}
    for k in keys:
        for t in toks[k]:
            if t in owner:
                parent[find(k)] = find(owner[t])
            else:
                owner[t] = k
    members: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        members[find(k)].append(k)
    out: dict[str, str] = {}
    for group in members.values():
        first = min(group)
        name = first if len(group) == 1 else f"{first} (+{len(group) - 1})"
        for k in group:
            out[k] = name
    return out


def _load_live_facts(vault: Path) -> list[dict]:
    from silica.kernel.paths import index_dir_for

    path = index_dir_for(str(vault)) / "episodic.json"
    if not path.is_file():
        return []
    facts = json.loads(path.read_text(encoding="utf-8")).get("facts", [])
    return [f for f in facts if f.get("status") == "live"]


def probe_question(inst: dict, run_root: Path, *, normalize: bool = False,
                   cluster: bool = False, max_df: int | None = None) -> dict:
    """Probe one question's frozen store; returns a flat metrics dict.

    normalize=True groups keys in their canonical (Layer A) form — the
    store's effective key identity, since capture matches normalized.
    cluster=True groups by post-hoc token clustering instead (both types):
    the ceiling a mechanical clustering layer could reach on this store."""
    from silica.kernel.episodic import normalize_key
    from tests.eval.longmemeval.runner import question_vault

    qid = inst["question_id"]
    qtype = inst["question_type"]
    gold = set(inst["answer_session_ids"])
    live = _load_live_facts(question_vault(run_root, qid))

    covered = {g for f in live for g in f["runs"] if g in gold}
    gold_facts = [f for f in live if gold & set(f["runs"])]

    if cluster:
        components = cluster_keys(sorted({f["key"] for f in live}),
                                  max_df=max_df)
        group_of = components.__getitem__
    else:
        canon = normalize_key if normalize else (lambda k: k)
        if qtype == "knowledge-update":
            group_of = canon
        else:
            group_of = lambda k: key_prefix(canon(k))  # noqa: E731
    by_group: dict[str, set[str]] = defaultdict(set)
    for f in gold_facts:
        by_group[group_of(f["key"])] |= gold & set(f["runs"])
    sizes: dict[str, int] = defaultdict(int)
    for f in live:
        sizes[group_of(f["key"])] += 1
    # Ties on coverage go to the SMALLEST group: the honest diagnostic when a
    # tiny precise cluster and a blob cover the same gold sessions.
    best_group, best_cov = max(by_group.items(),
                               key=lambda kv: (len(kv[1]), -sizes[kv[0]]),
                               default=("-", set()))
    return {
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


def run_probes(data: list[dict], run_root: Path, *, normalize: bool = False,
               cluster: bool = False, max_df: int | None = None) -> list[dict]:
    return [probe_question(q, run_root, normalize=normalize, cluster=cluster,
                           max_df=max_df)
            for q in data if q["question_type"] in PROBED_TYPES]


def render(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        lines.append(
            f"{r['question_id']:<16} {r['question_type']:<18} "
            f"capture {r['captured_sessions']}/{r['gold_sessions']}  "
            f"groups {r['groups']:>2}  best '{r['best_group']}' "
            f"covers {r['best_coverage']}/{r['gold_sessions']} "
            f"(size {r['best_size']})")
        for g, cov in list(r["group_coverage"].items())[:8]:
            lines.append(f"    {g:<46} {len(cov)}")
    return "\n".join(lines)


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
    args = ap.parse_args()
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    print(render(run_probes(data, Path(args.run_root).expanduser().resolve(),
                            normalize=args.normalize, cluster=args.cluster,
                            max_df=args.max_df)))


if __name__ == "__main__":
    main()
