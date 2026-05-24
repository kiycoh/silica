"""Pre-distill inbox + vault excerpts into a payload-ready JSON for the Fact Distiller subagent.

Reads a recon.py JSON report, extracts targeted excerpts from inbox files and
colliding vault notes for each candidate concept, and emits a compact JSON
payload. The Distiller consumes this directly — no further `read_file`
round-trips inside its context.

Design:
- SRP: one function per responsibility (load / classify / locate / excerpt / assemble).
- KISS: stdlib only, regex-driven, flat dict output.
- Heading-aware extraction for markdown sections; window fallback otherwise.
- Atomic-note shortcut: vault notes under FULL_INCLUDE_THRESHOLD are passed whole.

Note on Agentic Optimization (mirrors recon.py rationale):
Even when running inside execute_code we read files locally via stdlib rather
than via hermes_tools RPC. Trailing N read_file RPCs through the tool layer
would exceed Hermes' default 50 tool-call cap and add socket latency per file.
"""
import argparse
import json
import re
import sys
from pathlib import Path

# Dynamic Hermes Tools Integration (parity with recon.py / bulk_writer.py)
try:
    import hermes_tools  # noqa: F401
    HAS_HERMES = True
except ImportError:
    HAS_HERMES = False


# ---- Config ---------------------------------------------------------------

DEFAULT_WINDOW = 450          # Chars on each side of a non-heading concept match.
MAX_EXCERPT_CHARS = 2000      # Hard per-excerpt cap (inbox or vault).
MAX_OCCURRENCES = 2           # Max non-overlapping windows per concept per file.
FULL_INCLUDE_THRESHOLD = 6000 # Atomic-note cap from linter.py; include whole note below this.


# ---- I/O ------------------------------------------------------------------

def load_recon(path: Path) -> list:
    """Load the recon JSON, tolerating accidentally-captured stderr prefix lines.

    `recon.py` historically wrote a `[RECON STATS]` line to stderr. Hermes's
    `terminal()` tool can fold stderr into stdout, contaminating the captured
    file. We scan forward to the first `[` or `{` character, parse from there,
    and warn on stderr if we had to skip anything so the Router can diagnose.
    """
    raw = path.read_text(encoding="utf-8")
    # Find the first JSON-array or JSON-object start character that actually decodes as JSON.
    json_start = None
    for i, ch in enumerate(raw):
        if ch in "[{":
            try:
                json.loads(raw[i:])
                json_start = i
                break
            except json.JSONDecodeError:
                continue

    if json_start is None:
        raise ValueError(f"recon report at {path} contains no JSON content")

    if json_start > 0:
        skipped = raw[:json_start].strip()
        sys.stderr.write(
            f"[DISTILLER-PAYLOAD] Warning: skipped {json_start} bytes of non-JSON "
            f"prefix in {path} before parsing. Likely captured stderr. "
            f"Skipped content: {skipped[:200]!r}\n"
        )

    try:
        return json.loads(raw[json_start:])
    except json.JSONDecodeError as e:
        # Surface line/col within the cleaned slice so the Router can pinpoint.
        sys.stderr.write(
            f"[DISTILLER-PAYLOAD] Fatal: recon JSON malformed after prefix strip. "
            f"Error at line {e.lineno}, column {e.colno} (char {e.pos}): {e.msg}\n"
            f"Context: {raw[json_start + max(0, e.pos - 60):json_start + e.pos + 60]!r}\n"
        )
        raise


def safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


# ---- Classification -------------------------------------------------------

def classify_action(collision: dict, in_new_concepts: bool) -> str:
    """Translate recon's priority tier into a Distiller action hint."""
    if in_new_concepts:
        return "create"
    if collision is None:
        return "skip"
    if collision["best_match"] == "title":
        return "enrich"
    if collision["total_hits"] >= 3:
        return "review"
    return "likely_skip"


# ---- Heading-aware section extraction -------------------------------------

def compile_concept_regex(c: str) -> re.Pattern:
    """Compile a regex pattern for a concept ensuring word boundary matching.

    If a concept starts or ends with non-alphanumeric characters, the standard \b
    boundary check is skipped for that edge to prevent regex match failures on
    concepts containing special characters (e.g. parentheses or punctuation).
    """
    escaped = re.escape(c)
    start_b = r'\b' if c and re.match(r'\w', c) else ''
    end_b = r'\b' if c and re.search(r'\w$', c) else ''
    return re.compile(rf'{start_b}{escaped}{end_b}', re.IGNORECASE)


def find_heading(content: str, concept: str):
    """Return the first ATX heading line (H1–H4) containing the concept, or None."""
    escaped = re.escape(concept)
    start_b = r'\b' if concept and re.match(r'\w', concept) else ''
    end_b = r'\b' if concept and re.search(r'\w$', concept) else ''
    pattern = re.compile(
        rf'^(#{{1,4}})\s+.*{start_b}{escaped}{end_b}.*$',
        re.IGNORECASE | re.MULTILINE,
    )
    return pattern.search(content)


def extract_section(content: str, heading_match) -> str:
    """Grab text from the heading to the next equal-or-higher heading."""
    level = len(heading_match.group(1))
    # `(?!#)` prevents matching deeper-level headings (e.g. ## must not match ###).
    next_pattern = re.compile(rf'^#{{1,{level}}}(?!#)\s+', re.MULTILINE)
    next_match = next_pattern.search(content, pos=heading_match.end())
    end = next_match.start() if next_match else len(content)
    return content[heading_match.start():end].strip()


# ---- Window fallback ------------------------------------------------------

def expand_to_double_newline(content: str, start: int, end: int) -> tuple[int, int]:
    """Expand start and end indices to nearest double newline boundaries.

    Prevents truncating paragraphs, lists, code blocks, or LaTeX equations in markdown.
    """
    new_start = content.rfind('\n\n', 0, start)
    if new_start == -1:
        new_start = 0
    else:
        new_start += 2  # Skip past the double newline
    new_end = content.find('\n\n', end)
    if new_end == -1:
        new_end = len(content)
    return new_start, new_end


def safe_truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars without cutting through blocks or lines if possible.

    Tries to find a double newline boundary or a single newline boundary in the latter
    half of the text before resorting to a hard character limit truncation.
    """
    if len(text) <= max_chars:
        return text
    truncated_idx = text.rfind('\n\n', 0, max_chars)
    if truncated_idx != -1 and truncated_idx > max_chars // 2:
        return text[:truncated_idx].strip()
    truncated_idx = text.rfind('\n', 0, max_chars)
    if truncated_idx != -1 and truncated_idx > max_chars // 2:
        return text[:truncated_idx].strip()
    return text[:max_chars].strip()


def extract_windows(content: str, concept: str, window: int, max_occ: int) -> list:
    """Grab non-overlapping snippets around concept occurrences aligned to double newline boundaries."""
    pattern = compile_concept_regex(concept)
    windows = []
    last_end = -1
    for m in pattern.finditer(content):
        if len(windows) >= max_occ:
            break
        
        # Initial character-based window
        start = max(0, m.start() - window)
        end = min(len(content), m.end() + window)
        
        # Expand boundaries to double newlines to avoid truncation
        start, end = expand_to_double_newline(content, start, end)
        
        if start < last_end:
            continue
            
        windows.append(content[start:end].strip())
        last_end = end
    return windows


# ---- Excerpt orchestration ------------------------------------------------

def extract_excerpt(file_path: Path, concept: str, window: int) -> str:
    """Heading-aware extraction with window fallback. Capped at MAX_EXCERPT_CHARS."""
    content = safe_read(file_path)
    if not content:
        return ""
    heading = find_heading(content, concept)
    if heading:
        return safe_truncate(extract_section(content, heading), MAX_EXCERPT_CHARS)
    windows = extract_windows(content, concept, window, MAX_OCCURRENCES)
    if not windows:
        return ""
    return safe_truncate("\n\n[...]\n\n".join(windows), MAX_EXCERPT_CHARS)


def vault_content_or_excerpt(vault_path: Path, concept: str, window: int, is_title_match: bool) -> str:
    """For title matches on atomic notes, include the full note. Otherwise excerpt."""
    content = safe_read(vault_path)
    if not content:
        return ""
    if is_title_match and len(content) <= FULL_INCLUDE_THRESHOLD:
        return content.strip()
    return extract_excerpt(vault_path, concept, window)


# ---- Assembly -------------------------------------------------------------

def build_concept_entry(name: str, inbox_path: Path, collision: dict,
                        in_new_concepts: bool, window: int) -> dict:
    entry = {
        "name": name,
        "action_hint": classify_action(collision, in_new_concepts),
        "inbox_excerpt": extract_excerpt(inbox_path, name, window),
    }
    if collision and collision.get("hits"):
        best = collision["hits"][0]
        is_title = collision["best_match"] == "title"
        entry["vault_collision"] = {
            "path": best["path"],
            "match_type": collision["best_match"],
            "total_hits": collision["total_hits"],
            "excerpt": vault_content_or_excerpt(Path(best["path"]), name, window, is_title),
        }
    else:
        entry["vault_collision"] = None
    return entry


def build_payload(recon_reports: list, window: int) -> dict:
    batches = []
    for report in recon_reports:
        inbox_path = Path(report["file"])
        concepts = []
        for collision in report.get("collisions", []):
            concepts.append(build_concept_entry(
                name=collision["name"],
                inbox_path=inbox_path,
                collision=collision,
                in_new_concepts=False,
                window=window,
            ))
        for new_name in sorted(report.get("new_concepts", [])):
            concepts.append(build_concept_entry(
                name=new_name,
                inbox_path=inbox_path,
                collision=None,
                in_new_concepts=True,
                window=window,
            ))
        batches.append({"inbox_file": str(inbox_path), "concepts": concepts})
    return {"schema_version": 1, "batches": batches}


# ---- Partitioning ---------------------------------------------------------

def partition_by_concepts(payload: dict, max_concepts: int) -> list:
    """Greedy bin-pack the payload's batches into chunks of ≤max_concepts.

    If a single batch contains more than max_concepts, we split it into
    sub-batches with the same inbox_file but sliced concepts to keep
    each chunk ≤ max_concepts.
    """
    split_batches = []
    for batch in payload["batches"]:
        concepts = batch["concepts"]
        if len(concepts) <= max_concepts:
            split_batches.append(batch)
        else:
            for i in range(0, len(concepts), max_concepts):
                split_batches.append({
                    "inbox_file": batch["inbox_file"],
                    "concepts": concepts[i:i + max_concepts]
                })

    chunks = []
    current = []
    current_count = 0
    for batch in split_batches:
        batch_count = len(batch["concepts"])
        if current and (current_count + batch_count > max_concepts):
            chunks.append(current)
            current = []
            current_count = 0
        current.append(batch)
        current_count += batch_count
    if current:
        chunks.append(current)
    return [
        {"schema_version": payload["schema_version"], "batches": chunk}
        for chunk in chunks
    ]


def out_path_for_partition(out: Path, index: int) -> Path:
    """Insert '_{index}' before the extension. /tmp/p.json → /tmp/p_0.json."""
    return out.with_name(f"{out.stem}_{index}{out.suffix}")


# ---- Stats ----------------------------------------------------------------

def payload_stats(payload: dict, rendered_size: int) -> str:
    n_batches = len(payload["batches"])
    n_concepts = sum(len(b["concepts"]) for b in payload["batches"])
    actions = {}
    for b in payload["batches"]:
        for c in b["concepts"]:
            actions[c["action_hint"]] = actions.get(c["action_hint"], 0) + 1
    action_summary = ", ".join(f"{k}={v}" for k, v in sorted(actions.items()))
    return (f"[DISTILLER-PAYLOAD] {n_batches} inbox files, {n_concepts} concepts "
            f"({action_summary}), {rendered_size} chars")


# ---- CLI ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Pre-distill recon report into payload-ready JSON for the Fact Distiller."
    )
    ap.add_argument("--recon-report", required=True, type=Path,
                    help="Path to recon.py JSON output")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                    help=f"Chars on each side of a non-heading match (default {DEFAULT_WINDOW})")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only N inbox files from the recon report (after --offset)")
    ap.add_argument("--offset", type=int, default=0,
                    help="Skip the first N inbox files from the recon report")
    ap.add_argument("--max-concepts", type=int, default=7,
                    help="Partition output into multiple files, each with ≤N concepts. "
                         "Requires --out. Output files are <out_stem>_<i><out_suffix>.")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="Limit the number of partition files generated to at most N. "
                         "Only applied when --max-concepts is set.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path for payload JSON (default: stdout). "
                         "Required when --max-concepts is set.")
    args = ap.parse_args()

    if args.max_concepts is not None and args.out is None:
        print(json.dumps({"error": "--max-concepts requires --out"}))
        sys.exit(1)

    if not args.recon_report.exists():
        print(json.dumps({"error": f"recon report {args.recon_report} not found"}))
        sys.exit(1)

    reports = load_recon(args.recon_report)
    if args.offset:
        reports = reports[args.offset:]
    if args.limit is not None:
        reports = reports[:args.limit]

    payload = build_payload(reports, args.window)

    if args.max_concepts is not None:
        # Partition mode: bin-pack and emit numbered files.
        chunks = partition_by_concepts(payload, args.max_concepts)
        if args.max_batches is not None:
            chunks = chunks[:args.max_batches]
        total_concepts = sum(len(b["concepts"]) for chunk in chunks for b in chunk["batches"])
        for i, chunk in enumerate(chunks):
            out_path = out_path_for_partition(args.out, i)
            rendered = json.dumps(chunk, ensure_ascii=False, indent=2)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(rendered, encoding="utf-8")
            sys.stderr.write(payload_stats(chunk, len(rendered)).replace(
                "[DISTILLER-PAYLOAD]", f"[DISTILLER-PAYLOAD batch {i}]") + f" → {out_path}\n")
        avg = total_concepts / len(chunks) if chunks else 0
        sys.stderr.write(
            f"[DISTILLER-PAYLOAD] Partitioned {total_concepts} concepts into "
            f"{len(chunks)} batches (max {args.max_concepts}/batch, avg {avg:.1f})\n"
        )
        return

    # Single-output mode (unchanged).
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    sys.stderr.write(payload_stats(payload, len(rendered)) + "\n")


if __name__ == "__main__":
    main()