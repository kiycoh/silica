# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Note<->source provenance ledger (spec-hermes-coherence §3).

Append-only record of which notes derive from which version (sha256) of an
nucleated source file, keyed by source basename. Written at CLEANUP alongside
archiving (silica.router.states.finalize, sibling to _log_nucleate_completion)
and read by graph_report (source drift section) and /nucleate (re-nucleate of a
modified source warning).

Storage: `<vault_path>/provenance.json` — a JSON array of records:
    {"source": "lecture-03.md", "sha256": "…", "run_id": "…",
     "date": "2026-07-02", "notes": ["Concepts/Note A", "Concepts/Note B"]}
`notes` entries are vault-relative note paths without the `.md` extension
(RunManifestEntry.path, which strips it) — NOT the same form as graph_report
node ids, which carry `.md` (driver index keys). Callers that intersect the
two (e.g. graph_report's source-drift section) must strip `.md` at the seam.

No hash lives in note frontmatter (user-facing noise) — provenance lives
only here, ledger-side.

Kernel-only: no router/capabilities imports (import-linter boundary).
Additive: a missing/unreadable/unwritable store degrades to "no records"
everywhere — nothing here ever raises out to its caller.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PROVENANCE_FILENAME = "provenance.json"


def _resolve_vault_path(vault_path: str | None) -> str | None:
    if vault_path:
        return vault_path
    try:
        from silica.config import CONFIG

        return getattr(CONFIG, "vault_path", None) or None
    except Exception:
        return None


def _store_path(vault_path: str | None) -> Path | None:
    resolved = _resolve_vault_path(vault_path)
    if not resolved:
        return None
    return Path(resolved) / DEFAULT_PROVENANCE_FILENAME


def read_records(
    source: str | None = None,
    *,
    vault_path: str | None = None,
) -> list[dict[str, Any]]:
    """All provenance records, optionally filtered to one source basename.

    Missing store or unreadable file degrade to [] (additive: absence of
    the store must look like today's behaviour). Corrupt content is
    quarantined first (*.corrupt.<stamp>, surfaced by doctor): this ledger
    is authoritative — run_id/sha history is not reconstructible from the
    vault — and a later append_record would otherwise clobber the corrupt
    bytes with a fresh array.
    """
    path = _store_path(vault_path)
    if not path or not path.exists():
        return []
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"expected JSON array, got {type(records).__name__}")
    except Exception as exc:
        from silica.kernel.paths import quarantine

        dest = quarantine(path)
        logger.warning("provenance: corrupt store quarantined to %s: %s", dest or path, exc)
        return []
    if source is not None:
        records = [r for r in records if isinstance(r, dict) and r.get("source") == source]
    return records


def append_record(
    source: str,
    sha256: str,
    run_id: str,
    notes: list[str],
    *,
    vault_path: str | None = None,
    date: str | None = None,
) -> bool:
    """Append one record for `source`. Best-effort: swallows I/O errors and
    returns False rather than raising — CLEANUP must never fail on this.

    Idempotent on (source, sha256, run_id): a resumed run re-entering
    CLEANUP for the same file fires this again with an unchanged triple —
    that must not duplicate the record (mirrors run_log.append_log_line's
    resume-safety).
    """
    path = _store_path(vault_path)
    if not path:
        return False

    record = {
        "source": source,
        "sha256": sha256,
        "run_id": run_id,
        "date": date or datetime.now().strftime("%Y-%m-%d"),
        "notes": list(notes),
    }

    try:
        existing = read_records(vault_path=vault_path)
        if any(
            r.get("source") == source and r.get("sha256") == sha256 and r.get("run_id") == run_id
            for r in existing
        ):
            return False
        existing.append(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.debug("provenance: append failed (non-fatal): %s", exc)
        return False
    return True


def drifted_notes(
    *,
    vault_path: str | None = None,
) -> list[tuple[str, str]]:
    """`[(note, source_basename), ...]` for notes derived from a superseded
    source version.

    Rule (spec-hermes-coherence §3): for each source with >=2 records at
    different sha256 values, take the most recent record whose sha differs
    from the source's CURRENT (latest) sha — its notes that do NOT appear
    in ANY record carrying the current sha are drifted.
    """
    records = read_records(vault_path=vault_path)
    by_source: dict[str, list[dict]] = {}
    for r in records:
        src = r.get("source")
        if src:
            by_source.setdefault(src, []).append(r)

    out: list[tuple[str, str]] = []
    for source, recs in by_source.items():
        shas = {r.get("sha256") for r in recs}
        if len(shas) < 2:
            continue
        current_sha = recs[-1].get("sha256")
        current_notes: set[str] = set()
        for r in recs:
            if r.get("sha256") == current_sha:
                current_notes.update(r.get("notes") or [])
        old_recs = [r for r in recs if r.get("sha256") != current_sha]
        if not old_recs:
            continue
        last_old = old_recs[-1]
        for note in last_old.get("notes") or []:
            if note not in current_notes:
                out.append((note, source))
    return out


def _norm_note_ref(p: str) -> str:
    """Fold a note reference to a comparable key: vault-relative POSIX, no
    `.md`, casefolded. Provenance stores RunManifestEntry.path (already
    vault-relative, no `.md`); a caller may pass an absolute or `.md`-suffixed
    path, so relativize when possible and degrade to a plain strip otherwise."""
    ref = p or ""
    try:
        from silica.kernel.paths import to_vault_relative

        ref = to_vault_relative(ref, ensure_md=False)
    except Exception:
        ref = ref.replace("\\", "/").strip("/")
    return ref.removesuffix(".md").casefold()


def note_authored_by(
    note_path: str,
    source: str,
    *,
    vault_path: str | None = None,
) -> bool:
    """True when `source` (a source basename) already authored `note_path`.

    Reads the provenance ledger: on any prior run, did this exact source file
    write or patch this note? The patch executor uses it to make a re-ingest
    idempotent — a source must not re-append its own concepts into the notes it
    already wrote (each re-append is a redundant "Note aggiuntive (da <source>)"
    block). A genuinely new concept has no prior authored note, so it still
    flows to a fresh write; a DIFFERENT source enriching the same note still
    patches. Matches any recorded version of the source (an A->B->A edit still
    counts). Absent/unreadable ledger degrades to False.
    """
    target = _norm_note_ref(note_path)
    if not target:
        return False
    for r in read_records(source, vault_path=vault_path):
        if any(_norm_note_ref(n) == target for n in (r.get("notes") or [])):
            return True
    return False


def check_renucleate(
    source: str,
    incoming_sha256: str,
    *,
    vault_path: str | None = None,
) -> tuple[bool, int]:
    """`(is_modified, notes_derived_from_the_prior_version)`.

    Used by /nucleate right before staging a file: warns when the inbox file
    about to be re-nucleated carries a different sha256 than the last known
    record for that source basename. No prior record -> (False, 0) — a
    first nucleate is not a re-nucleate.
    """
    recs = read_records(source, vault_path=vault_path)
    if not recs:
        return False, 0
    last = recs[-1]
    if last.get("sha256") == incoming_sha256:
        return False, 0
    return True, len(last.get("notes") or [])


# --- Span grounding (verbatim-contract gate) --------------------------------
# The distiller must carry formulas and code verbatim from the source excerpt
# (distiller_prompt "Content Quality Requirements"). Prose is rewritten and
# translated by design, so it can't be checked mechanically — but math and
# code can: a $$...$$ / ```...``` span in the output that cannot be located
# in the source excerpt is a fabrication candidate. Warn-only signal:
# re-typesetting ASCII math into LaTeX is sanctioned by the prompt's own
# few-shot example, so a span class is gated only when the source itself
# uses that markup ($ for math, ``` for code).

_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")

MIN_GROUNDABLE_CHARS = 12  # normalized; shorter spans ($x$, \top) match anywhere
GROUNDING_FLOOR = 0.85     # matched-char fraction under LOCAL difflib alignment
LOCALITY_WINDOW = 2        # matched blocks must fit in a window of N * len(span)

_NUMERAL_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


def _local_match_fraction(s: str, src: str) -> float:
    """Best matched-char fraction of *s* with all blocks inside one source
    window of LOCALITY_WINDOW * len(s). Global scatter would let a formula
    recombined from fragments across the excerpt self-ground; localization
    is the whole point of the gate."""
    from difflib import SequenceMatcher

    # blocks under 3 chars are coincidence ('v', ')'), not localization —
    # they inflate the fraction exactly on recombined formulas
    blocks = [b for b in SequenceMatcher(None, s, src, autojunk=False).get_matching_blocks() if b.size >= 3]
    if not blocks:
        return 0.0
    window = LOCALITY_WINDOW * len(s)
    best = 0
    for i in range(len(blocks)):  # blocks are few; O(n²) is fine
        lo = blocks[i].b
        best = max(best, sum(b.size for b in blocks[i:] if b.b + b.size <= lo + window))
    return best / len(s)


def ungrounded_spans(body: str, source: str) -> list[str]:
    """Verbatim-contract spans of *body* (math, fenced code) not locatable in *source*.

    Returns the offending spans (whitespace-normalized); empty list means
    fully grounded or nothing gateable. A span class is checked only when
    *source* itself contains that markup — LaTeX in the output for an
    ASCII-math source is legitimate re-typesetting, not drift.

    Two independent checks per span (either failing flags it):
    - numeric literals (≥2 chars) must appear verbatim in the source —
      numbers survive re-typesetting and translation, so an absent constant
      is the sharpest fabrication signal (altered 0.01→0.1, invented ε=10⁻⁸);
    - fuzzy match must be LOCAL (see _local_match_fraction).
    """
    spans: list[str] = []
    if "```" in source:
        spans += _FENCE_RE.findall(body)
    if "$" in source:
        # ponytail: "$" also matches currency; acceptable for a warn-only gate
        rest = _FENCE_RE.sub("", body)
        spans += _DISPLAY_MATH_RE.findall(rest)
        spans += _INLINE_MATH_RE.findall(_DISPLAY_MATH_RE.sub("", rest))

    src = _norm_ws(source)
    out: list[str] = []
    for span in spans:
        s = _norm_ws(span)
        if len(s) < MIN_GROUNDABLE_CHARS or s in src:
            continue
        numerals = [n for n in _NUMERAL_RE.findall(s) if len(n) >= 2]
        if any(n not in src for n in numerals):
            out.append(s)
            continue
        if _local_match_fraction(s, src) < GROUNDING_FLOOR:
            out.append(s)
    return out


# Extractive invariant: markers the model may prepend to a copied span (list
# bullets, ordered markers, headings, blockquotes) and inline wikilinks the
# autolink phase injects later — stripped before the substring check so added
# structure never reads as content drift.
_LEADING_MARKER_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)]|#{1,6}|>)\s+")
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_EXTRACTIVE_MIN_CHARS = 12  # normalized; shorter residues match trivially


def _norm_extract(s: str) -> str:
    """Whitespace + straight-apostrophe normalization for substring matching.
    Case is preserved: a case change is an edit, not a copied span."""
    return _norm_ws(s.replace("’", "'").replace("‘", "'"))


def nonextractive_lines(body: str, source: str) -> list[str]:
    """Body content-lines that are NOT verbatim spans of *source*.

    The `extractive` distill profile must SELECT spans from the transcript, not
    rewrite them; this is the mechanical check enforcing that. Split the body
    into lines, strip structural markers (bullets, headings, blockquotes) and
    inline wikilinks (autolink adds those post-write), normalize whitespace and
    apostrophes, and require each remaining line to be a substring of the
    normalized source. Returns the offending lines; empty means fully
    extractive.

    ponytail: lines under MIN chars after stripping are skipped — a short
    residue substring-matches almost any source, so enforcing them only yields
    false rejections; paraphrase and fabrication are prose-length anyway.
    """
    src = _norm_extract(source)
    out: list[str] = []
    for raw in body.splitlines():
        line = _WIKILINK_RE.sub(r"\1", _LEADING_MARKER_RE.sub("", raw))
        norm = _norm_extract(line)
        if len(norm) < _EXTRACTIVE_MIN_CHARS:
            continue
        if norm not in src:
            out.append(norm)
    return out


def content_sha256(source_path: str) -> str:
    """SHA-256 hex digest of a source file's content.

    Mirrors silica.router.orchestrator.InjectorFSM.run()'s hashing exactly
    (DRIVER.read_note(...).content.encode("utf-8"), falling back to raw file
    bytes) so a value computed here (e.g. by the /nucleate pre-check) compares
    equal to the sha256 CLEANUP later records for an unmodified file. Never
    raises — returns "" when the file can't be read either way.
    """
    try:
        from silica.driver import DRIVER

        content_bytes = DRIVER.read_note(source_path).content.encode("utf-8")
        return hashlib.sha256(content_bytes).hexdigest()
    except Exception:
        try:
            content_bytes = Path(source_path).read_bytes()
            return hashlib.sha256(content_bytes).hexdigest()
        except OSError:
            return ""
