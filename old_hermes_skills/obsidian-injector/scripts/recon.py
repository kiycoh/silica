"""Mechanical recon for the Hermes obsidian-injector pipeline.

Reads inbox markdown, extracts candidate concepts via heuristics, checks for
collisions against the vault, emits a compact JSON report for the Router's
semantic phase.

Design:
- SRP: one function per responsibility (normalize / extract / dedupe / search / score / format).
- KISS: regex heuristics, flat data, stdlib only.
- Aggressive noise filter for PDF-converted-slide artifacts.
- Title-vs-body weighting: a vault filename match dominates body mentions.
"""
import argparse
import json
import os
import re
from pathlib import Path

# Dynamic Hermes Tools Integration
try:
    import hermes_tools
    HAS_HERMES = True
except ImportError:
    HAS_HERMES = False

# Note on Agentic Optimization:
# Although HAS_HERMES tells us if we're running inside the execute_code tool,
# we intentionally perform vault file-system traversal (os.walk) and content reading
# locally using Python standard library functions.
# Trailing through Hermes RPC read_file / search_files for multiple files would:
# 1. Instantly exceed Hermes' default limit of 50 tool calls per execution (max_tool_calls).
# 2. Add RPC socket serialization latency to every file lookup.
# Therefore, local walking is used for high-performance recon, while hermes_tools is 
# available for session logging if needed.


# ---- Config ---------------------------------------------------------------

MIN_LEN, MAX_LEN = 3, 50
TITLE_BONUS = 50
TOP_K_HITS = 3

# Common Italian stopwords and administrative/slide metadata terms to prevent false positive concept extraction
STOPWORDS = {
    # Italian prepositions, articles, conjunctions, pronouns
    "di", "da", "in", "con", "su", "per", "tra", "fra", "a", "e", "o", "ma", "se", "anche", "come",
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "del", "dello", "della", "dei", "degli",
    "delle", "al", "allo", "alla", "ai", "agli", "alle", "dal", "dallo", "dalla", "dagli", "dalle",
    "nel", "nello", "nella", "nei", "negli", "nelle", "sul", "sullo", "sulla", "sui", "sugli", "sulle",
    "che", "chi", "cui", "cosa", "quale", "quali", "questo", "questa", "questi", "queste", "quello",
    "quella", "quelli", "quelle", "mio", "tuo", "suo", "nostro", "vostro", "loro", "dei", "del", "altro",
    # Common slide metadata / administrative terms
    "parte", "testo", "esame", "contenuti", "libri", "unipa", "anno", "corso", "appunti",
    "lezione", "capitolo", "studio", "domande", "risposte", "esercizio", "esercizi", "tema", "temi",
    "prof", "professore", "docente", "università", "universita", "sito", "web", "link", "online",
    "slide", "slides", "presentazione", "pagine", "pagina", "riferimenti", "argomenti", "riassunto"
}


# Patterns that, if matched, disqualify a candidate. Tuned against
# slide-deck-converted markdown (PDF-bullet artifacts, slide section markers,
# rhetorical-question templates, "ACRONYM: expansion" duplicates).
NOISE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'^(Capitolo|Lezione|Esercizio)\b[:\s]',          # "Capitolo 1:", "Lezione 5"
    r'^(Riassunto|Argomenti|Riferimenti)\s*$',        # bare slide section words
    r'\((continua|segue)\)\s*$',                      # "(continua)" suffix
    r'^q\s',                                          # PDF 'q' bullet artifact
    r'^[A-Z]{2,6}:\s',                                # "DSL: digital subscriber line" (acronym already extracted)
    r"^Cos'?\xe8\b",                                  # "Cos'è X" Italian rhetorical
    r'^\s*\d+[\.\)\-]\s+',                            # "1. Foo", "1) Foo"
    r'^\s*\d{4}[\-\u2013]\d{4}',                      # year ranges "1961-1972"
    r':\s*$',                                         # trailing colon (slide marker)
    r'\?\s*$',                                        # rhetorical questions
    r'\s+vs\.?\s+',                                   # "X vs Y" comparison fragments
    r'^(continua|segue)\b',
]]

# Strip only leading non-word chars (bullets, dingbats, PUA glyphs from PDF
# font-remap like \uf071) and trailing whitespace. Trailing punctuation is
# preserved because it's the signal we filter on (colons, question marks).
LEADING_GARBAGE = re.compile(r'^[\W_]+')

# ---- Normalization & filtering --------------------------------------------

def normalize(s: str) -> str:
    """Strip leading non-word chars and trailing whitespace; collapse spaces.

    Trailing punctuation is preserved on purpose: noise filters depend on it.
    """
    s = LEADING_GARBAGE.sub('', s)
    s = re.sub(r'\s+', ' ', s).rstrip()
    return s

def is_concept(s: str) -> bool:
    """Apply length, content, and noise-pattern filters."""
    if s.lower().strip() in STOPWORDS:
        return False
    if not (MIN_LEN <= len(s) <= MAX_LEN):
        return False
    if not re.search(r'[A-Za-z\u00C0-\u00FF]{3,}', s):
        return False
    return not any(p.search(s) for p in NOISE_PATTERNS)

# ---- Extraction (one function per source pattern) -------------------------

def from_headings(content: str) -> set:
    return {m.group(1) for m in re.finditer(r'^#{1,4}\s+(.+?)\s*$', content, re.MULTILINE)}

def from_bold(content: str) -> set:
    return {m.group(1) for m in re.finditer(r'\*\*(.+?)\*\*', content)}

def from_acronyms(content: str) -> set:
    """2-6 char uppercase tokens, common in tech notes (TCP, HTTP, IPv6, SMTP)."""
    return set(re.findall(r'\b[A-Z]{2,6}\b', content))

def extract_concepts(path: Path) -> set:
    try:
        content = path.read_text(encoding='utf-8')
    except OSError:
        return set()
    raw = from_headings(content) | from_bold(content) | from_acronyms(content)
    return {c for c in (normalize(r) for r in raw) if is_concept(c)}

# ---- Dedup ----------------------------------------------------------------

def dedupe(concepts: set) -> set:
    """Case-insensitive dedup; prefer the longer/more-cased form."""
    chosen: dict[str, str] = {}
    for c in concepts:
        key = c.lower()
        if key not in chosen or len(c) > len(chosen[key]):
            chosen[key] = c
    return set(chosen.values())

# ---- Vault search ---------------------------------------------------------

def iter_vault_md(vault_dir: Path):
    for root, dirs, files in os.walk(vault_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in files:
            if f.endswith('.md'):
                yield Path(root) / f

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


def is_title_match(c: str, stem: str) -> bool:
    """Determine if a concept matches a note's stem (title) in a robust way.

    Checks:
    1. Exact case-insensitive match.
    2. Concept inside stem using safe boundary regex.
    3. Stem inside concept using safe boundary regex.
    4. Word-set overlap (if one set of words is a subset of the other),
       which handles swapped word orders and parenthesized annotations.
    """
    c_lower = c.lower()
    stem_lower = stem.lower()
    if c_lower == stem_lower:
        return True

    # Check if concept is in the stem using the safe boundary regex
    pat_c = compile_concept_regex(c)
    if pat_c.search(stem):
        return True

    # Check if stem is in the concept using the safe boundary regex
    pat_s = compile_concept_regex(stem)
    if pat_s.search(c):
        return True

    # Word-set overlap fallback: check if one is a subset of the other
    c_words = set(re.findall(r'\w+', c_lower))
    s_words = set(re.findall(r'\w+', stem_lower))
    if c_words and s_words:
        if c_words.issubset(s_words) or s_words.issubset(c_words):
            return True

    return False


def search_vault(concepts: set, vault_dir: Path, exclude_root: Path | None = None) -> dict:
    """Map concept -> [(path, body_count, in_title)].
    Skip anything under exclude_root (the inbox) or inside a 'done' subtree,
    so an inbox nested within the vault never self-collides.
    """
    exclude_root = exclude_root.resolve() if exclude_root else None
    patterns = {c: compile_concept_regex(c) for c in concepts}
    hits: dict[str, list] = {c: [] for c in concepts}
    for p in iter_vault_md(vault_dir):
        rp = p.resolve()
        if exclude_root and (exclude_root == rp or exclude_root in rp.parents):
            continue
        if 'done' in rp.parts:
            continue
        try:
            content = rp.read_text(encoding='utf-8')
        except OSError:
            continue
        for c, pat in patterns.items():
            count = len(pat.findall(content))
            if count == 0:
                continue
            in_title = is_title_match(c, rp.stem)
            hits[c].append((str(rp), count, in_title))
    return hits

# ---- Scoring & ranking ----------------------------------------------------

def hit_score(body_count: int, in_title: bool) -> int:
    """Title match dominates body mentions; tiebreak by body count."""
    return body_count + (TITLE_BONUS if in_title else 0)

def rank_hits(raw: list, top_k: int = TOP_K_HITS) -> list:
    return sorted(raw, key=lambda h: hit_score(h[1], h[2]), reverse=True)[:top_k]

# ---- Report assembly ------------------------------------------------------

def collision_priority(c: dict) -> tuple:
    """Sort key encoding Router-friendly decision priority.

    Tier 0: title match    -> spoke with this exact name exists; enrich-vs-skip needs attention
    Tier 1: body, high hits -> menzionato ovunque ma no spoke; create-new-spoke candidate
    Tier 2: body, low hits  -> menzioni sparse; likely skip
    Within tier, higher total_hits first.
    """
    if c["best_match"] == "title":
        tier = 0
    elif c["total_hits"] >= 3:
        tier = 1
    else:
        tier = 2
    return (tier, -c["total_hits"])

def file_report(filepath: str, concepts: set, all_hits: dict) -> dict:
    collisions, new_concepts = [], []
    for c in concepts:
        raw = all_hits.get(c, [])
        if not raw:
            new_concepts.append(c)
            continue
        ranked = rank_hits(raw)
        collisions.append({
            "name": c,
            "total_hits": sum(h[1] for h in raw),
            "best_match": "title" if ranked[0][2] else "body",
            "hits": [{"path": p, "count": ct, "in_title": t} for p, ct, t in ranked],
        })
    collisions.sort(key=collision_priority)
    new_concepts.sort()
    return {"file": filepath, "collisions": collisions, "new_concepts": new_concepts}

def run_recon(inbox_dir: Path, vault_dir: Path, limit: int = None, offset: int = 0) -> list:
    per_file: dict[str, set] = {}
    all_concepts: set = set()
    
    # Sort files alphabetically to ensure deterministic subset ordering
    files = sorted(list(inbox_dir.glob('**/*.md')))
    
    # Filter out files that are inside a "done" subdirectory
    filtered_files = []
    for f in files:
        if 'done' in f.parts:
            continue
        filtered_files.append(f)
        
    if limit is not None:
        filtered_files = filtered_files[offset:offset+limit]
    else:
        filtered_files = filtered_files[offset:]
        
    for md in filtered_files:
        cs = dedupe(extract_concepts(md))
        per_file[str(md)] = cs
        all_concepts |= cs

    if not all_concepts:
        return [{"file": fp, "collisions": [], "new_concepts": []} for fp in per_file]

    hits = search_vault(all_concepts, vault_dir, exclude_root=inbox_dir)
    return [file_report(fp, cs, hits) for fp, cs in per_file.items()]

# ---- Output formatting ----------------------------------------------------

PRIORITY_LABELS = {
    0: "ENRICH (title match — spoke exists)",
    1: "REVIEW (body match, high hits — potential new spoke)",
    2: "LIKELY SKIP (sparse mentions)",
}

def render_human(reports: list, vault_root: Path) -> str:
    """Markdown renderer for terminal/debugging use; not consumed by the Router."""
    def rel(p: str) -> str:
        try: return str(Path(p).relative_to(vault_root))
        except ValueError: return p

    lines = []
    unique_vault_notes = set()
    for r in reports:
        lines.append(f"\n## {Path(r['file']).name}")
        lines.append(f"_{rel(r['file'])}_\n")
        if r["collisions"]:
            current_tier = None
            for c in r["collisions"]:
                tier = collision_priority(c)[0]
                if tier != current_tier:
                    lines.append(f"\n### {PRIORITY_LABELS[tier]}")
                    current_tier = tier
                lines.append(f"- **{c['name']}** ({c['total_hits']} hits, best={c['best_match']})")
                for h in c["hits"]:
                    unique_vault_notes.add(h["path"])
                    flag = " 🎯" if h["in_title"] else ""
                    lines.append(f"  - `{rel(h['path'])}` ({h['count']}){flag}")
        if r["new_concepts"]:
            lines.append(f"\n### NEW ({len(r['new_concepts'])})")
            lines.append(", ".join(r["new_concepts"]))
            
    lines.append(f"\n=== RECON STATS ===")
    lines.append(f"Inbox notes processed: {len(reports)}")
    lines.append(f"Vault notes intersected: {len(unique_vault_notes)}")
    return "\n".join(lines)

# ---- CLI ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="Mechanical recon for Hermes obsidian-injector")
    parser.add_argument("--inbox", required=True, type=Path)
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit recon to the first N files sorted alphabetically (ignores 'done' subfolders)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Offset from the first file to start processing (ignores 'done' subfolders)")
    parser.add_argument("--format", choices=["json", "human"], default="json",
                        help="json (default, for Router) or human (markdown, for terminal review)")
    args = parser.parse_args()

    if not args.inbox.exists() or not args.vault.exists():
        print(json.dumps({"error": "inbox or vault path does not exist"}))
        exit(1)

    reports = run_recon(args.inbox, args.vault, limit=args.limit, offset=args.offset)

    if args.format == "human":
        # Human mode: emit the stats line for terminal readers and render markdown.
        unique_vault_notes = set()
        for r in reports:
            for c in r.get("collisions", []):
                for h in c.get("hits", []):
                    unique_vault_notes.add(h["path"])
        sys.stderr.write(
            f"\n[RECON STATS] Processed {len(reports)} inbox notes "
            f"intersected with {len(unique_vault_notes)} unique vault notes.\n"
        )
        print(render_human(reports, args.vault))
    else:
        # JSON mode (default, Router-consumed): stdout must be PARSEABLE JSON ONLY.
        # No stderr stats — Hermes's terminal() tool can fold stderr into the
        # captured output stream and contaminate downstream json.load().
        print(json.dumps(reports, indent=2, ensure_ascii=False))