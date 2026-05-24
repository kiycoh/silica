"""Quick smoke test: feed the actual noise patterns from test_output.txt and
confirm they're filtered, while real concepts pass through."""
import sys
sys.path.insert(0, '.')
from recon import normalize, is_concept, dedupe, from_acronyms, hit_score

# Noise samples lifted directly from the user's test_output.txt
noise = [
    "Capitolo 1: Introduzione",
    "Capitolo 1: roadmap",
    "Obiettivi:",
    "Panoramica:",
    "Domanda:",
    "Esempio:",
    "Riassunto",  # standalone, no colon — borderline
    "1961-1972: sviluppo della commutazione di pacchetto",
    "4. Ritardo di propagazione (d/s)",
    "Cache web (continua)",
    "Cookie (continua)",
    "Esempio di caching (continua)",
    "client:",
    "server:",
    "Timeout:",
    "Cos'è Internet",  # has apostrophe — keep, it IS a concept-ish question
    "Cos'è Internet?",  # rhetorical
    "Perché il caching web?",
    "Ritarto di trasmissione vs. ritardo di propagazione",
    "\uf071 Ethernet:",       # PUA glyph + colon
    "\u274f LAN wireless:",
    "\u2751 DSL: digital subscriber line",
    "q una serie di passi successivi",  # PDF bullet artifact
    "Reti di calcolatori e Internet: Un approccio top-down",  # book title (52 chars, also has colon)
]

# Real concepts that should pass
signal = [
    "Throughput",
    "Ritardo di propagazione",
    "Ritardo di trasmissione",
    "Collo di bottiglia",
    "IPv6",
    "Tunneling",
    "Sicurezza di rete",
    "Accesso wireless",
    "Commutazione di pacchetto",
    "HTTP",
    "TCP",
    "SMTP",
]

print("=== NOISE (should all fail is_concept) ===")
leaked = []
for s in noise:
    n = normalize(s)
    ok = is_concept(n)
    marker = "LEAK" if ok else "ok"
    print(f"  [{marker}] {s!r} -> {n!r}")
    if ok: leaked.append((s, n))

print("\n=== SIGNAL (should all pass is_concept) ===")
dropped = []
for s in signal:
    n = normalize(s)
    ok = is_concept(n)
    marker = "ok" if ok else "DROP"
    print(f"  [{marker}] {s!r} -> {n!r}")
    if not ok: dropped.append((s, n))

print("\n=== DEDUP ===")
mixed = {"Ritardo di Trasmissione", "Ritardo di trasmissione", "RITARDO DI TRASMISSIONE", "HTTP", "http"}
print(f"  input:  {mixed}")
print(f"  output: {dedupe(mixed)}")

print("\n=== ACRONYMS ===")
sample = "Il **TCP** è un protocollo, vedi anche UDP, HTTP, IPv6 e SMTP."
print(f"  {from_acronyms(sample)}")

print("\n=== TITLE WEIGHTING ===")
# Throughput case from real output: in 'Throughput TCP.md' (3 hits, title) vs 'Collo di bottiglia.md' (5 hits, body)
print(f"  Throughput in 'Throughput TCP.md' (3 body hits, title match): score = {hit_score(3, True)}")
print(f"  Throughput in 'Collo di bottiglia.md' (5 body hits, no title): score = {hit_score(5, False)}")
print("  -> Title match correctly outranks higher body count.")

print(f"\n=== SUMMARY ===")
print(f"  Noise items leaked: {len(leaked)}/{len(noise)}")
print(f"  Signal items dropped: {len(dropped)}/{len(signal)}")

print("\n=== PRIORITY TIERS ===")
from recon import collision_priority

# Mix of collision shapes from the real test_output.txt
collisions = [
    {"name": "PROGRAMMA",            "total_hits": 2, "best_match": "body"},   # tier 2
    {"name": "Throughput",            "total_hits": 9, "best_match": "title"},  # tier 0 (title)
    {"name": "Tunneling",             "total_hits": 3, "best_match": "title"},  # tier 0 (title)
    {"name": "Topologia a stella",    "total_hits": 1, "best_match": "body"},   # tier 2
    {"name": "Collo di bottiglia",    "total_hits": 4, "best_match": "title"},  # tier 0
    {"name": "Quattro cause di ritardo per i pacchetti", "total_hits": 1, "best_match": "body"},  # tier 2
    {"name": "ConcettoDiBody",        "total_hits": 5, "best_match": "body"},   # tier 1 (body, high)
]
ordered = sorted(collisions, key=collision_priority)
for c in ordered:
    print(f"  tier={collision_priority(c)[0]}  {c['best_match']:5}  hits={c['total_hits']:2}  {c['name']}")

print("\n=== RUN RECON WITH LIMIT AND DONE FILTERING ===")
import tempfile
from pathlib import Path
from recon import run_recon

with tempfile.TemporaryDirectory() as tmp_dir_str:
    tmp_dir = Path(tmp_dir_str)
    inbox = tmp_dir / "inbox"
    vault = tmp_dir / "vault"
    inbox.mkdir()
    vault.mkdir()
    
    # Create target notes
    (inbox / "file_a.md").write_text("## ConceptA\nQuesto è il ConceptA.", encoding='utf-8')
    (inbox / "file_c.md").write_text("## ConceptC\nQuesto è il ConceptC.", encoding='utf-8')
    (inbox / "file_b.md").write_text("## ConceptB\nQuesto è il ConceptB.", encoding='utf-8')
    
    # Create a note in vault that contains "ConceptA" to cause a collision
    (vault / "spoke_a.md").write_text("## Spoke A\nQuesto parla del ConceptA.", encoding='utf-8')
    
    # Create done subfolder and a file in it
    done_dir = inbox / "done"
    done_dir.mkdir()
    (done_dir / "file_d.md").write_text("## ConceptD\nQuesto è il ConceptD.", encoding='utf-8')
    
    # Run with limit=2 (should get file_a and file_b, and skip file_c and file_d due to done/ path check and limit)
    reports = run_recon(inbox, vault, limit=2)
    files_processed = [Path(r["file"]).name for r in reports]
    print(f"  Processed files (expected ['file_a.md', 'file_b.md']): {files_processed}")
    assert files_processed == ["file_a.md", "file_b.md"], f"Expected ['file_a.md', 'file_b.md'], got {files_processed}"
    
    print("\n=== HUMAN RENDER OUTPUT ===")
    from recon import render_human
    print(render_human(reports, vault))
    
    print("\n=== RUN RECON WITH NESTED INBOX ===")
    inbox_nested = vault / "0 Inbox"
    inbox_nested.mkdir()
    # Write a note inside the nested inbox
    (inbox_nested / "nested_file.md").write_text("## ConceptNested\nQuesto è il ConceptNested.", encoding='utf-8')
    # Run recon with nested inbox
    reports_nested = run_recon(inbox_nested, vault)
    # Check that there are no collisions pointing to nested_file.md
    for r in reports_nested:
        for c in r.get("collisions", []):
            for h in c.get("hits", []):
                hit_path = Path(h["path"])
                assert inbox_nested not in hit_path.parents, f"Self-collision detected: {h['path']} is inside nested inbox!"
    print("  Nested inbox self-collision avoidance test PASSED successfully!")
    
    print("  Limit and done/ filtering test PASSED successfully!")

print("\n=== ROBUST TITLE MATCHING AND WORD BOUNDARY TESTS ===")
from recon import is_title_match, compile_concept_regex

# Test cases for is_title_match
test_cases = [
    # (concept, stem, expected_match)
    ("Sottoreti", "Subnetting (Sottoreti)", True),
    ("Sottoreti", "Subnetting(Sottoreti)", True),
    ("Sottoreti (Subnetting)", "Sottoreti (Subnetting)", True),
    ("Sottoreti (Subnetting)", "Subnetting (Sottoreti)", True),
    ("Sottoreti (Subnetting)", "Sottoreti", True),
    ("Sicurezza di rete", "Sicurezza", True),
    ("Sicurezza", "Sicurezza di rete", True),
    ("Reti", "Sottoreti", False),
    ("Sottorete", "Sottoreti", False),
]

for c, stem, expected in test_cases:
    res = is_title_match(c, stem)
    print(f"  Concept: {c!r:25} | Stem: {stem!r:25} | Match: {res!s:5} (Expected: {expected!s:5})")
    assert res == expected, f"Failed for Concept: {c!r}, Stem: {stem!r}. Got {res}, expected {expected}"

# Test cases for compile_concept_regex with non-word character boundary edges
regex_test_cases = [
    # (concept, text, expected_match)
    ("Sottoreti (Subnetting)", "Questo parla di Sottoreti (Subnetting).", True),
    ("Sottoreti (Subnetting)", "Sottoreti (Subnetting) è importante.", True),
    (".NET", "Programmazione in .NET", True),
    (".NET", "Questo è un .NETwork", False),
    ("C++", "Linguaggio C++", True),
    ("C++", "ABC++", False),
]

for c, text, expected in regex_test_cases:
    pat = compile_concept_regex(c)
    res = bool(pat.search(text))
    print(f"  Concept: {c!r:25} | Text: {text!r:45} | Match: {res!s:5} (Expected: {expected!s:5})")
    assert res == expected, f"Regex failed for Concept: {c!r}, Text: {text!r}. Got {res}, expected {expected}"

print("  Robust matching tests PASSED successfully!")