import re

MIN_LEN, MAX_LEN = 3, 50
TITLE_BONUS = 50
TOP_K_HITS = 3

_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n?", re.DOTALL)

STOPWORDS = {
    "di", "da", "in", "con", "su", "per", "tra", "fra", "a", "e", "o", "ma", "se", "anche", "come",
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "del", "dello", "della", "dei", "degli",
    "delle", "al", "allo", "alla", "ai", "agli", "alle", "dal", "dallo", "dalla", "dagli", "dalle",
    "nel", "nello", "nella", "nei", "negli", "nelle", "sul", "sullo", "sulla", "sui", "sugli", "sulle",
    "che", "chi", "cui", "cosa", "quale", "quali", "questo", "questa", "questi", "queste", "quello",
    "quella", "quelli", "quelle", "mio", "tuo", "suo", "nostro", "vostro", "loro", "dei", "del", "altro",
    "parte", "testo", "esame", "contenuti", "libri", "unipa", "anno", "corso", "appunti",
    "lezione", "capitolo", "studio", "domande", "risposte", "esercizio", "esercizi", "tema", "temi",
    "prof", "professore", "docente", "università", "universita", "sito", "web", "link", "online",
    "slide", "slides", "presentazione", "pagine", "pagina", "riferimenti", "argomenti", "riassunto"
}

NOISE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'^(Capitolo|Lezione|Esercizio)\b[:\s]',
    r'^(Riassunto|Argomenti|Riferimenti)\s*$',
    r'\((continua|segue)\)\s*$',
    r'^q\s',
    r'^[A-Z]{2,6}:\s',
    r"^Cos'?\xe8\b",
    r'^\s*\d+[\.\)\-]\s+',
    r'^\s*\d{4}[\-\u2013]\d{4}',
    r':\s*$',
    r'\?\s*$',
    r'\s+vs\.?\s+',
    r'^(continua|segue)\b',
]]
LEADING_GARBAGE = re.compile(r'^[\W_]+')

def normalize(s: str) -> str:
    s = LEADING_GARBAGE.sub('', s)
    return re.sub(r'\s+', ' ', s).rstrip()

def is_concept(s: str) -> bool:
    if s.lower().strip() in STOPWORDS:
        return False
    if not (MIN_LEN <= len(s) <= MAX_LEN):
        return False
    if not re.search(r'[A-Za-z\u00C0-\u00FF]{3,}', s):
        return False
    return not any(p.search(s) for p in NOISE_PATTERNS)

def from_headings(content: str) -> set:
    return {m.group(1) for m in re.finditer(r'^#{1,4}\s+(.+?)\s*$', content, re.MULTILINE)}

def from_bold(content: str) -> set:
    return {m.group(1) for m in re.finditer(r'\*\*(.+?)\*\*', content)}

def _strip_frontmatter(content: str) -> str:
    return _FRONTMATTER_RE.sub('', content, count=1)

def from_acronyms(content: str) -> set:
    return set(re.findall(r'\b[A-Z]{2,6}\b', content))

def extract_concepts(content: str) -> set:
    body = _strip_frontmatter(content)
    raw = from_headings(body) | from_bold(body) | from_acronyms(body)
    return dedupe({c for c in (normalize(r) for r in raw) if is_concept(c)})

def dedupe(concepts: set) -> set:
    chosen: dict[str, str] = {}
    for c in concepts:
        key = c.lower()
        if key not in chosen or len(c) > len(chosen[key]):
            chosen[key] = c
    return set(chosen.values())

def is_title_match(c: str, stem: str) -> bool:
    c_lower, stem_lower = c.lower(), stem.lower()
    if c_lower == stem_lower: return True
    if c_lower in stem_lower or stem_lower in c_lower: return True
    c_words = set(re.findall(r'\w+', c_lower))
    s_words = set(re.findall(r'\w+', stem_lower))
    if c_words and s_words and (c_words.issubset(s_words) or s_words.issubset(c_words)):
        return True
    return False

def hit_score(body_count: int, in_title: bool) -> int:
    return body_count + (TITLE_BONUS if in_title else 0)

def rank_hits(raw: list, top_k: int = TOP_K_HITS) -> list:
    return sorted(raw, key=lambda h: hit_score(h["count"], h["in_title"]), reverse=True)[:top_k]

def collision_priority(c: dict) -> tuple:
    if c["best_match"] == "title": return (0, -c["total_hits"])
    if c["total_hits"] >= 3: return (1, -c["total_hits"])
    return (2, -c["total_hits"])
