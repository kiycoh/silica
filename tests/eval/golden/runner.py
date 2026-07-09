# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Golden coherence harness runner — CLI + manifest digest + report + metrics.

One run output, three consumers: acceptance read (the printed table),
regression gate (``tests/golden/test_golden_regression.py`` imports ``collect``
+ ``compare``), and calibration instrument (``--verbose``).

Phase 1 = cheap/deterministic tier (classify, links, integrity). Embedder-tier
probes (dedup, neighbors) print a visible SKIP row.

  uv run python -m tests.eval.golden --vault ~/Documents/Obsidian/test [--verbose]
  uv run python -m tests.eval.golden --vault <v> --freeze-baseline
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

from tests.eval.golden import probe_classify, probe_integrity, probe_links

BASELINE_PATH = Path(__file__).parent / "baseline.json"
METRICS_PATH = Path(__file__).parent / "metrics.json"

# Gate rules (single source — the pytest imports compare()).
GATED_DROP_2PP = ("classify.agreement", "links.recall")   # Phase 2 appends dedup.*, neighbors.*
GATED_EXACT_ONE = ("integrity.rate",)
_PRIMARIES = ("classify.agreement", "links.recall", "integrity.rate")


def _today() -> str:
    return datetime.date.today().isoformat()


def iter_notes(vault: Path) -> list[Path]:
    """Sorted ``*.md`` under the vault, excluding dot-directories.

    Single source of truth for both the digest and every probe, so the harness
    can never measure a note set the digest didn't hash.
    """
    return sorted(
        p for p in vault.rglob("*.md")
        if not any(part.startswith(".") for part in p.relative_to(vault).parts)
    )


def resolve_vault(cli: str | None) -> Path:
    """--vault > SILICA_VAULT; expanduser+resolve; exit 2 if missing."""
    raw = cli or os.getenv("SILICA_VAULT")
    if not raw:
        print("no vault: pass --vault or set SILICA_VAULT")
        raise SystemExit(2)
    p = Path(raw).expanduser().resolve()
    if not p.is_dir():
        print(f"vault not found: {p}")
        raise SystemExit(2)
    return p


def vault_digest(vault: Path) -> tuple[str, int]:
    """sha256 over sorted ``relpath\\0sha256(content)`` lines → ("sha256:<hex>", count).

    Plain hashlib — provenance.content_sha256 goes through the DRIVER, not reused.
    """
    notes = iter_notes(vault)
    h = hashlib.sha256()
    for p in notes:
        rel = p.relative_to(vault).as_posix()
        content_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        h.update(f"{rel}\0{content_hash}\n".encode("utf-8"))
    return f"sha256:{h.hexdigest()}", len(notes)


def config_snapshot(store) -> dict:
    from silica.config import CONFIG

    return {
        "tau_high": 0.55,
        "tau_low": 0.15,
        "sim_threshold_high": getattr(CONFIG, "sim_threshold_high", None),
        "sim_threshold_low": getattr(CONFIG, "sim_threshold_low", None),
        "embedding_model": getattr(CONFIG, "embedding_model", None),
        "cooccur_store": "present" if len(store) else "absent",
        "cooccur_lang": getattr(store, "lang", None),
        "taxonomy_derivation": probe_classify.DERIVATION,
        "taxonomy_top_n": probe_classify.TOP_N,
        "relatedness_legs": None,  # Phase 2
    }


def collect(vault: Path, *, tier: str = "cheap", verbose: bool = False) -> dict:
    """Library entry point (the gate imports this). Runs the cheap-tier probes
    against ``vault`` and returns the full run document."""
    import silica.driver
    from silica.config import CONFIG
    from silica.kernel.cooccurrence import CooccurStore, _index_path_for

    CONFIG.vault_path = str(vault)
    CONFIG.backend = "fs"
    silica.driver._driver = None
    store = CooccurStore(path=_index_path_for(str(vault)))

    metrics: dict[str, float] = {}

    c = probe_classify.run(vault, store, verbose=verbose)
    metrics["classify.agreement"] = c["agreement"]
    metrics["classify.uncategorized_rate"] = c["uncategorized_rate"]
    metrics["classify.notes"] = c["notes"]

    lk = probe_links.run(vault, verbose=verbose)
    metrics["links.recall"] = lk["recall"]
    metrics["links.extra_per_note"] = lk["extra_per_note"]
    metrics["links.links_evaluated"] = lk["links_evaluated"]
    metrics["links.notes_evaluated"] = lk["notes_evaluated"]

    ig = probe_integrity.run(vault, verbose=verbose)
    metrics["integrity.rate"] = ig["rate"]
    metrics["integrity.notes"] = ig["notes"]
    metrics["integrity.vault_structural_violations"] = ig["vault_structural_violations"]
    metrics["integrity.vault_style_flags"] = ig["vault_style_flags"]
    metrics["integrity.vault_notes_with_structural"] = ig["vault_notes_with_structural"]

    if tier in ("embedder", "all"):
        print("SKIP  dedup.*      — Phase 2 (embedder tier) not implemented")
        print("SKIP  neighbors.*  — Phase 2 (embedder tier) not implemented")

    # arbitrary single trend number — labeled as such, never gated
    metrics["coherence_index"] = round(sum(metrics[k] for k in _PRIMARIES) / len(_PRIMARIES), 4)

    digest, notes = vault_digest(vault)
    return {
        "generated_at": _today(),
        "tier": tier,
        "vault": {"digest": digest, "notes": notes, "path": str(vault)},
        "config": config_snapshot(store),
        "metrics": metrics,
    }


def compare(baseline: dict, doc: dict) -> list[str]:
    """The single gate rule (pytest imports this). Digest/config refusals happen
    BEFORE this, not here."""
    b, d = baseline["metrics"], doc["metrics"]
    fails: list[str] = []
    for key in GATED_DROP_2PP:
        if key in b and key in d and d[key] < b[key] - 0.02:
            fails.append(f"{key}: {d[key]:.3f} < baseline {b[key]:.3f} − 2pp")
    for key in GATED_EXACT_ONE:
        if key in d and d[key] != 1.0:
            fails.append(f"{key}: {d[key]:.3f} != 1.0 (any new violation fails)")
    return fails


def print_table(doc: dict, baseline: dict | None) -> None:
    v = doc["vault"]
    cfg = doc["config"]
    print(f"\nvault: {v['path']}  ({v['notes']} notes)")
    print(f"digest: {v['digest']}  tier: {doc['tier']}  "
          f"cooccur: {cfg['cooccur_store']}/{cfg['cooccur_lang']}  "
          f"embedder: {cfg['embedding_model']}")
    base_m = baseline["metrics"] if baseline else {}
    gated = set(GATED_DROP_2PP) | set(GATED_EXACT_ONE)
    print(f"\n{'metric':<38} {'value':>10} {'baseline':>10} {'delta':>9}  gate")
    for key in sorted(doc["metrics"]):
        val = doc["metrics"][key]
        base = base_m.get(key)
        delta = f"{val - base:+.3f}" if isinstance(base, (int, float)) else "—"
        base_s = f"{base:.3f}" if isinstance(base, (int, float)) else "—"
        mark = "GATE" if key in gated else ""
        print(f"{key:<38} {val:>10.3f} {base_s:>10} {delta:>9}  {mark}")


def _write_json(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tests.eval.golden")
    ap.add_argument("--vault")
    ap.add_argument("--tier", choices=["cheap", "embedder", "all"], default="cheap")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--freeze-baseline", action="store_true")
    args = ap.parse_args(argv)

    vault = resolve_vault(args.vault)
    try:
        doc = collect(vault, tier=args.tier, verbose=args.verbose)
    finally:
        import silica.driver
        silica.driver._driver = None

    _write_json(METRICS_PATH, doc)  # always

    if args.freeze_baseline:
        frozen = {**doc, "frozen_at": _today()}
        _write_json(BASELINE_PATH, frozen)
        print_table(doc, None)
        print(f"\nbaseline frozen → {BASELINE_PATH}")
        return 0

    baseline = json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else None
    if baseline is None:
        print_table(doc, None)
        print("\nno baseline yet — freeze one with --freeze-baseline")
        return 0

    if baseline["vault"]["digest"] != doc["vault"]["digest"]:
        print_table(doc, baseline)
        print("\nvault drifted — re-baseline deliberately with --freeze-baseline")
        return 1
    if baseline["config"]["cooccur_store"] != doc["config"]["cooccur_store"]:
        print_table(doc, baseline)
        print("\ncooccur mode changed — re-baseline deliberately with --freeze-baseline")
        return 1

    print_table(doc, baseline)
    fails = compare(baseline, doc)
    if fails:
        print("\nFAIL:")
        for f in fails:
            print(f"  {f}")
        return 1
    print("\nPASS")
    return 0
