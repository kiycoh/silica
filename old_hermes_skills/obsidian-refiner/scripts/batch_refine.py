#!/usr/bin/env python3
"""Batch Refiner: iterates a folder, triages each note, and emits
deterministic ops (split/normalize) + an enrich queue for the Router.

Usage:
    python3 batch_refine.py --folder <TARGET_FOLDER> \
        [--det-out /tmp/deterministic_ops.json] \
        [--enrich-out /tmp/enrich_queue.json] \
        [--dry-run]
"""
# --- hermes_common bootstrap (uniform across all hermes skills) ---
import os, sys
_p = os.path.dirname(os.path.abspath(__file__))
while _p != os.path.dirname(_p) and not os.path.isdir(os.path.join(_p, "hermes_common")):
    _p = os.path.dirname(_p)
if _p not in sys.path:
    sys.path.insert(0, _p)
# --- end bootstrap ---

# Allow sibling script imports (inspect_note, normalize_frontmatter, split_monolith)
_scripts = os.path.dirname(os.path.abspath(__file__))
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

import argparse, json, glob
from pathlib import Path

from inspect_note import inspect
from split_monolith import build_ops
from normalize_frontmatter import normalize


def triage(report):
    """Classify an inspect report into one of: decouple, reformat, enrich, ok."""
    if report["mode_hint"] == "decouple":
        return "decouple"
    if report["is_lean"] or report["is_empty"]:
        return "enrich"
    if report["frontmatter_issues"]:
        return "reformat"
    return "ok"


def batch(folder, dry_run=False):
    """Process all .md files in folder. Returns (det_ops, enrich_queue, summary)."""
    md_files = sorted(glob.glob(os.path.join(folder, "**", "*.md"), recursive=True))
    if not md_files:
        print(f"[BATCH] No .md files found in {folder}", file=sys.stderr)
        return [], [], {"total": 0}

    det_ops = []
    enrich_queue = []
    summary = {"total": len(md_files), "decouple": 0, "reformat": 0, "enrich": 0, "ok": 0, "errors": []}

    for path in md_files:
        try:
            report = inspect(path)
        except Exception as e:
            summary["errors"].append({"path": path, "error": str(e)})
            continue

        category = triage(report)
        summary[category] = summary.get(category, 0) + 1

        if dry_run:
            print(f"  [{category.upper():>9}] {path}  (chars={report['char_count']}, "
                  f"issues={len(report['frontmatter_issues'])})", file=sys.stderr)
            continue

        if category == "decouple":
            parent = os.path.dirname(path)
            hub = Path(path).stem
            try:
                ops, titles = build_ops(path, parent, hub)
                det_ops.extend(ops)
            except SystemExit as e:
                summary["errors"].append({"path": path, "error": str(e)})

        elif category == "reformat":
            ops = normalize(path)
            det_ops.extend(ops)

        elif category == "enrich":
            # Normalize tags deterministically first (if needed), then queue for LLM
            norm_ops = normalize(path)
            det_ops.extend(norm_ops)
            enrich_queue.append({
                "path": path,
                "title": Path(path).stem,
                "char_count": report["char_count"],
                "is_empty": report["is_empty"],
            })
        # category == "ok": skip

    return det_ops, enrich_queue, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch refiner: triage + deterministic ops")
    ap.add_argument("--folder", required=True, help="Target folder with .md notes")
    ap.add_argument("--det-out", default="/tmp/deterministic_ops.json",
                    help="Output path for deterministic bulk_writer ops")
    ap.add_argument("--enrich-out", default="/tmp/enrich_queue.json",
                    help="Output path for enrich queue (Router/LLM)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print triage results without writing ops files")
    a = ap.parse_args()

    if not os.path.isdir(a.folder):
        sys.exit(f"[BATCH] Error: {a.folder} is not a directory")

    det_ops, enrich_queue, summary = batch(a.folder, dry_run=a.dry_run)

    if a.dry_run:
        print(f"\n[BATCH DRY-RUN] {json.dumps(summary, indent=2)}", file=sys.stderr)
        raise SystemExit(0)

    # Write deterministic ops
    with open(a.det_out, "w", encoding="utf-8") as f:
        json.dump(det_ops, f, ensure_ascii=False, indent=2)

    # Write enrich queue
    with open(a.enrich_out, "w", encoding="utf-8") as f:
        json.dump(enrich_queue, f, ensure_ascii=False, indent=2)

    print(f"[BATCH] {summary['total']} notes triaged: "
          f"{summary['decouple']} decouple, {summary['reformat']} reformat, "
          f"{summary['enrich']} enrich, {summary['ok']} ok, "
          f"{len(summary['errors'])} errors", file=sys.stderr)
    print(f"[BATCH] Deterministic ops: {len(det_ops)} -> {a.det_out}", file=sys.stderr)
    print(f"[BATCH] Enrich queue: {len(enrich_queue)} -> {a.enrich_out}", file=sys.stderr)

    # Print machine-readable summary to stdout
    print(json.dumps(summary, indent=2))
