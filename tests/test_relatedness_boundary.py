"""Architectural boundary for the relatedness facade.

kernel/relatedness.py is the single place where the two PROPOSE-legs
(embeddings, co-occurrence) are fused into a note ranking. Relatedness
*queries* must go through it; direct imports of the legs are reserved for:

  - index maintenance (build/refresh/delete entries),
  - pairwise cosine math that does not fit rank fusion (dedup windows,
    salience gates, missing-link thresholds),
  - constructing stores to inject INTO the facade.

This test pins the set of modules allowed to import the legs directly. A new
direct-import site fails here and forces an explicit decision: route through
the facade, or extend the allowlist with a justification.
"""
from pathlib import Path

import re

SILICA_ROOT = Path(__file__).resolve().parent.parent / "silica"

_LEG_IMPORT_RE = re.compile(
    r"from silica\.kernel\.(embed|cooccurrence) import|"
    r"import silica\.kernel\.(embed|cooccurrence)\b"
)

# The legs themselves are out of scope — they ARE the implementation.
LEGS = {"kernel/embed.py", "kernel/cooccurrence.py"}

# module (relative to silica/) → why direct leg access is legitimate
ALLOWED = {
    "kernel/relatedness.py":        "the facade itself",
    "kernel/run_substrate.py":      "constructs stores to inject into the facade",
    "kernel/graph_report/embed_signals.py": "pairwise cosine (missing links, dup pairs)",
    "kernel/graph_report/cooccur_delta.py": "co-occurrence delta + cosine-band filter + store injection",
    "kernel/graph_export.py":       "cluster labels via CooccurStore.community_labels, not relatedness ranking",
    "kernel/vault_map.py":          "session-start vault map via CooccurStore (to_networkx/node_label, top_stems), not relatedness ranking",
    "kernel/classify.py":           "L1 tokenizer/concept matching, not relatedness ranking",
    "kernel/keyphrase.py":          "pairwise cosine (candidate phrase vs document theme) for concept reranking, not note ranking",
    "router/coordinator.py":        "constructs stores to inject into the facade",
    "router/organize_fsm.py":       "L1 co-occurrence classification, not relatedness ranking",
    "router/orchestrator.py":       "co-occurrence index freshness hook (build_index)",
    "router/states/collision.py":   "constructs stores to inject into the facade",
    "router/states/setup.py":       "pairwise cosine (crossdedup, salience theme gate)",
    "router/states/write.py":       "incremental embed index refresh after writes",
    "router/states/finalize.py":    "embed index cleanup on rollback",
    "tools/graph.py":               "index refresh tools + raw semantic search by design",
    "tools/runners.py":             "pairwise cosine dedup windows (silica_dedup)",
}


def test_leg_imports_are_allowlisted():
    offenders = []
    for path in SILICA_ROOT.rglob("*.py"):
        rel = path.relative_to(SILICA_ROOT).as_posix()
        if rel in ALLOWED or rel in LEGS:
            continue
        if _LEG_IMPORT_RE.search(path.read_text(encoding="utf-8")):
            offenders.append(rel)
    assert not offenders, (
        "Direct kernel.embed/kernel.cooccurrence import outside the allowlist: "
        f"{offenders}. Route relatedness queries through kernel/relatedness.py, "
        "or extend ALLOWED in this test with a justification."
    )


def test_allowlist_has_no_stale_entries():
    stale = [
        rel for rel in ALLOWED
        if not (SILICA_ROOT / rel).exists()
        or not _LEG_IMPORT_RE.search((SILICA_ROOT / rel).read_text(encoding="utf-8"))
    ]
    assert not stale, (
        f"Allowlist entries no longer import the legs directly: {stale}. "
        "Remove them so the boundary stays tight."
    )
