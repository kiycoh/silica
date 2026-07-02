"""Note<->source provenance ledger (spec-hermes-coherence §3).

Append-only record of which notes derive from which version (sha256) of an
ingested source file, keyed by source basename. Written at CLEANUP alongside
archiving (silica.router.states.finalize, sibling to _log_ingest_completion)
and read by graph_report (source drift section) and /ingest (re-ingest of a
modified source warning).

Storage: `<vault_path>/provenance.json` — a JSON array of records:
    {"source": "lezione-03.md", "sha256": "…", "run_id": "…",
     "date": "2026-07-02", "notes": ["Concepts/Nota A", "Concepts/Nota B"]}
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


def _store_path(vault_path: str | None, filename: str) -> Path | None:
    resolved = _resolve_vault_path(vault_path)
    if not resolved:
        return None
    return Path(resolved) / filename


def read_records(
    source: str | None = None,
    *,
    vault_path: str | None = None,
    filename: str = DEFAULT_PROVENANCE_FILENAME,
) -> list[dict[str, Any]]:
    """All provenance records, optionally filtered to one source basename.

    Missing store, unreadable file, or corrupt JSON all degrade to []
    (additive: absence of the store must look like today's behaviour).
    """
    path = _store_path(vault_path, filename)
    if not path or not path.exists():
        return []
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            return []
    except Exception as exc:
        logger.debug("provenance: read failed (non-fatal): %s", exc)
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
    filename: str = DEFAULT_PROVENANCE_FILENAME,
    date: str | None = None,
) -> bool:
    """Append one record for `source`. Best-effort: swallows I/O errors and
    returns False rather than raising — CLEANUP must never fail on this.

    Idempotent on (source, sha256, run_id): a resumed run re-entering
    CLEANUP for the same file fires this again with an unchanged triple —
    that must not duplicate the record (mirrors run_log.append_log_line's
    resume-safety).
    """
    path = _store_path(vault_path, filename)
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
        existing = read_records(vault_path=vault_path, filename=filename)
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
    filename: str = DEFAULT_PROVENANCE_FILENAME,
) -> list[tuple[str, str]]:
    """`[(note, source_basename), ...]` for notes derived from a superseded
    source version.

    Rule (spec-hermes-coherence §3): for each source with >=2 records at
    different sha256 values, take the most recent record whose sha differs
    from the source's CURRENT (latest) sha — its notes that do NOT appear
    in ANY record carrying the current sha are drifted.
    """
    records = read_records(vault_path=vault_path, filename=filename)
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


def check_reingest(
    source: str,
    incoming_sha256: str,
    *,
    vault_path: str | None = None,
    filename: str = DEFAULT_PROVENANCE_FILENAME,
) -> tuple[bool, int]:
    """`(is_modified, notes_derived_from_the_prior_version)`.

    Used by /ingest right before staging a file: warns when the inbox file
    about to be re-ingested carries a different sha256 than the last known
    record for that source basename. No prior record -> (False, 0) — a
    first ingest is not a re-ingest.
    """
    recs = read_records(source, vault_path=vault_path, filename=filename)
    if not recs:
        return False, 0
    last = recs[-1]
    if last.get("sha256") == incoming_sha256:
        return False, 0
    return True, len(last.get("notes") or [])


def content_sha256(source_path: str) -> str:
    """SHA-256 hex digest of a source file's content.

    Mirrors silica.router.orchestrator.InjectorFSM.run()'s hashing exactly
    (DRIVER.read_note(...).content.encode("utf-8"), falling back to raw file
    bytes) so a value computed here (e.g. by the /ingest pre-check) compares
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
