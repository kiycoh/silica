# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""L1/L2 Note classifier — assigns notes to taxonomy folders.

Two classification legs, applied in sequence:

  L1 (deterministic, zero LLM cost):
    Uses the co-occurrence index (CooccurStore) to extract each note's concept
    profile, then scores that profile against each FolderRule's themes via stem
    overlap. Reuses _seed_from_text / _profile_from_seeds from relatedness.py
    (concept-level granularity reconciliation).

  L2 (semantic, LLM arbiter — only for ambiguous notes):
    Notes whose best L1 score falls in [tau_low, tau_high] are sent to the LLM
    in a single batched call. The LLM receives the note title, a snippet, and
    the candidate folder descriptions, and returns a folder choice.

Abstaining rules:
  - If both L1 and L2 abstain (empty concept profile + no LLM), the note is
    assigned to taxonomy.uncategorized.
  - If the target folder equals the note's current folder, needs_move=False
    and the note is excluded from the move plan.
"""
from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# L2 arbiter — constrained-decoding wrapper models
#
# json_schema (response_format) requires an object root, so the arbiter's
# per-note assignments are wrapped in an object instead of returned as a
# bare JSON array.
# ---------------------------------------------------------------------------

class FolderAssignment(BaseModel):
    index: int
    folder: str


class ArbitrationResult(BaseModel):
    assignments: list[FolderAssignment]


# Ambiguous band: scores in this range trigger the LLM arbiter.
_DEFAULT_TAU_HIGH = 0.55   # above → clear L1 winner
_DEFAULT_TAU_LOW  = 0.15   # below → uncategorized (no signal)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class Classification:
    """Result of classifying a single note against the taxonomy."""

    note_path: str                    # vault-relative path (with .md)
    current_folder: str               # parent folder of note_path
    target_folder: str                # best-matching taxonomy folder
    confidence: float                 # 0–1, from L1 overlap score or 1.0 for LLM
    evidence: str                     # "cooccur" | "keyword" | "llm" | "uncategorized"
    needs_move: bool                  # True iff target_folder != current_folder
    title: str = ""                   # note title (stem)
    rule_themes: list[str] = field(default_factory=list)   # matched rule's themes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _current_folder(note_path: str) -> str:
    """Return the parent folder of a vault-relative note path."""
    parts = note_path.replace("\\", "/").rsplit("/", 1)
    return parts[0] if len(parts) > 1 else ""


def _note_title(note_path: str) -> str:
    """Stem of the note filename without .md extension."""
    return os.path.splitext(os.path.basename(note_path))[0]


def _read_body(note_path: str) -> str | None:
    """Read a note's body (frontmatter stripped) via the driver; None on failure."""
    try:
        from silica.driver import DRIVER
        from silica.kernel import frontmatter
        nc = DRIVER.read_note(note_path)
        _data, _fm, body = frontmatter.split(nc.content)
        return body
    except Exception:
        return None


def _stems_from_body(body: str, lang: str) -> dict[str, int]:
    """Tokenize a note body into {stem: count} (same pipeline as the cooccur index)."""
    from silica.kernel.cooccurrence import tokenize
    from silica.kernel.text import clean_body
    return dict(Counter(
        stem
        for sentence in tokenize(clean_body(body, fences=False), stem_lang=lang)
        for stem, _surface in sentence
    ))


def _extract_year(val: Any) -> int | None:
    """Robustly extract a 4-digit year from a value."""
    if not val:
        return None
    if isinstance(val, int):
        if 1900 <= val <= 2100:
            return val
        return None
    val_str = str(val).strip()
    import re
    m = re.match(r'^(\d{4})', val_str)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None


def _get_file_fs_year(note_path: str) -> int | None:
    """Retrieve the filesystem modification year for a note path."""
    from pathlib import Path
    from silica.config import CONFIG
    vault_path = getattr(CONFIG, "vault_path", "")
    if not vault_path:
        return None
    full_path = Path(vault_path) / note_path
    if not full_path.exists():
        return None
    try:
        stat_result = full_path.stat()
        mtime = stat_result.st_mtime
        import datetime
        dt = datetime.datetime.fromtimestamp(mtime, datetime.timezone.utc)
        return dt.year
    except Exception:
        return None


def _evaluate_metadata_filter(note_path: str, props: dict, filt: Any) -> bool:
    """Evaluate if note properties pass a MetadataFilter."""
    key = filt.key.lower()
    
    # Normalize props keys to lowercase for case-insensitivity
    normalized_props = {k.lower(): v for k, v in props.items()}
    val = normalized_props.get(key)
    
    # Fallback to filesystem if key is date/created-related and not in frontmatter props
    if val is None and key in ("date", "created", "created_at", "creation_date"):
        val = _get_file_fs_year(note_path)
        
    if val is None:
        return False
        
    op = filt.operator.lower()
    target_val_str = str(filt.value).lower()
    
    if op == "equals":
        return str(val).lower() == target_val_str
        
    elif op == "contains":
        if isinstance(val, list):
            return any(target_val_str in str(item).lower() for item in val)
        return target_val_str in str(val).lower()
        
    elif op in ("year_equals", "year_greater_than", "year_less_than"):
        year = _extract_year(val)
        if year is None:
            return False
        try:
            target_year = int(filt.value)
        except ValueError:
            return False
            
        if op == "year_equals":
            return year == target_year
        elif op == "year_greater_than":
            return year > target_year
        elif op == "year_less_than":
            return year < target_year
            
    return False


def _score_note_against_rules(
    note_path: str,
    concept_stems: dict[str, int],
    taxonomy: Any,  # Taxonomy
    props: dict,
    *,
    lang: str = "english",
) -> tuple[str, float, str, list[str]]:
    """Score note concept profile against each FolderRule.

    Returns (best_folder, best_score, evidence, matched_themes).
    evidence is "keyword" if a keyword exact-match fired, else "cooccur".
    """
    from silica.kernel.cooccurrence import tokenize

    title = _note_title(note_path)
    title_lower = title.lower()

    best_folder = taxonomy.uncategorized
    best_score = 0.0
    best_evidence = "uncategorized"
    best_themes: list[str] = []

    for rule in taxonomy.rules:
        # Check metadata filters first (gate)
        if hasattr(rule, "metadata_filters") and rule.metadata_filters:
            passed_all = True
            for filt in rule.metadata_filters:
                if not _evaluate_metadata_filter(note_path, props, filt):
                    passed_all = False
                    break
            if not passed_all:
                continue

        score = 0.0
        evidence = "cooccur"

        # --- Keyword hit in title (fast exact match) ---
        kw_hits = sum(1 for k in rule.keyword_set() if k in title_lower)
        kw_score = min(kw_hits * 0.4, 0.8)
        if kw_hits:
            evidence = "keyword"
        score += kw_score

        # --- Theme overlap via co-occurrence stems ---
        if rule.themes:
            rule_stems: set[str] = set()
            for theme in rule.themes:
                for sentence in tokenize(theme, stem_lang=lang, stopword_lang=lang):
                    rule_stems.update(stem for stem, _s in sentence)

            overlap = sum(count for stem, count in concept_stems.items() if stem in rule_stems)
            total_rule_stems = len(rule_stems) if rule_stems else 1
            # Normalised: proportion of rule stems that the note covers, weighted by frequency
            theme_score = min(overlap / (total_rule_stems * 3), 0.6)  # cap at 0.6
            score += theme_score

        if score > best_score:
            best_score = score
            best_folder = rule.folder
            best_evidence = evidence
            best_themes = list(rule.themes)

    return best_folder, round(best_score, 4), best_evidence, best_themes


def _llm_arbitrate(
    ambiguous: list[tuple[str, str, list[Any]]],  # (note_path, snippet, candidate_rules)
    taxonomy: Any,  # Taxonomy
) -> dict[str, str]:
    """Batch LLM call for ambiguous notes. Returns {note_path: folder}."""
    if not ambiguous:
        return {}

    from silica.agent.llm import call_llm
    from silica.config import CONFIG
    from silica.kernel.sanitize import parse_json

    # Build a concise prompt listing all ambiguous notes and their candidate folders
    folder_list = "\n".join(
        f"  - folder: \"{r.folder}\" — {r.description or ', '.join(r.themes[:3])}"
        for r in taxonomy.rules
    )
    uncategorized_line = f"  - folder: \"{taxonomy.uncategorized}\" — no clear match"

    note_entries = []
    for idx, (note_path, snippet, _rules) in enumerate(ambiguous):
        title = _note_title(note_path)
        note_entries.append(
            f"{idx}. title: \"{title}\"\n   snippet: \"{snippet[:200].strip()}\""
        )
    notes_block = "\n".join(note_entries)

    system_prompt = (
        "You are a vault organizer. Assign each note to exactly one folder from the list.\n"
        "Return a JSON object: {\"assignments\": [{\"index\": 0, \"folder\": \"Concepts/AI\"}, ...]}\n"
        "Use only the folder values listed. No explanation."
    )
    user_msg = (
        f"Available folders:\n{folder_list}\n{uncategorized_line}\n\n"
        f"Notes to classify:\n{notes_block}"
    )

    try:
        response = call_llm(
            model=CONFIG.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            tools=None,
            response_format=ArbitrationResult,
        )
        parsed, _ = parse_json(response.text or "", strict=False)
    except Exception as exc:
        logger.warning("LLM arbiter failed (%s) — falling back to uncategorized", exc)
        return {note_path: taxonomy.uncategorized for note_path, _s, _r in ambiguous}

    assignments = parsed.get("assignments") if isinstance(parsed, dict) else None
    if not isinstance(assignments, list):
        logger.warning("LLM arbiter response missing 'assignments' list: %r — falling back", parsed)
        return {note_path: taxonomy.uncategorized for note_path, _s, _r in ambiguous}

    valid_folders = {r.folder for r in taxonomy.rules} | {taxonomy.uncategorized}
    result: dict[str, str] = {}
    for entry in assignments:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("index")
        folder = entry.get("folder", "")
        if idx is None or idx >= len(ambiguous):
            continue
        note_path = ambiguous[idx][0]
        result[note_path] = folder if folder in valid_folders else taxonomy.uncategorized

    # Fill any missing entries
    for note_path, _s, _r in ambiguous:
        result.setdefault(note_path, taxonomy.uncategorized)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_notes(
    note_paths: list[str],
    taxonomy: Any,  # Taxonomy
    *,
    cooccur_store: Any | None = None,
    llm_arbiter: bool = True,
    tau_high: float = _DEFAULT_TAU_HIGH,
    tau_low: float = _DEFAULT_TAU_LOW,
    lang: str | None = None,
    props_map: dict[str, dict] | None = None,
    move_uncategorized: bool = False,
) -> list[Classification]:
    """Classify notes against taxonomy rules.

    Args:
        note_paths:    vault-relative paths (with .md) to classify.
        taxonomy:      validated Taxonomy instance.
        cooccur_store: pre-loaded CooccurStore; loaded from disk when None.
        llm_arbiter:   if True, send ambiguous notes (score in [tau_low, tau_high])
                       to the LLM for a final verdict.
        tau_high:      notes with score >= tau_high are classified by L1 alone.
        tau_low:       notes with score < tau_low go straight to uncategorized.
        lang:          stemmer language; defaults to CooccurStore.lang or "english".
        props_map:     optional map of note_path -> properties dict for testing.
        move_uncategorized: if False (default), notes that match no rule stay
                       where they are (needs_move=False) instead of being
                       collected into taxonomy.uncategorized.

    Returns:
        One Classification per note_path (same order).
    """
    if not taxonomy.rules:
        # No rules → everything is uncategorized
        return [
            Classification(
                note_path=p,
                current_folder=_current_folder(p),
                target_folder=taxonomy.uncategorized,
                confidence=0.0,
                evidence="uncategorized",
                needs_move=move_uncategorized and _current_folder(p) != taxonomy.uncategorized,
                title=_note_title(p),
            )
            for p in note_paths
        ]

    # Load co-occurrence store
    if cooccur_store is None:
        try:
            from silica.kernel.cooccurrence import get_cooccur_store
            cooccur_store = get_cooccur_store()
        except Exception as exc:
            logger.warning("classify: CooccurStore unavailable (%s) — using empty index", exc)
            cooccur_store = None

    effective_lang = lang or (getattr(cooccur_store, "lang", None) or "english")

    # --- L1 pass ---
    results: list[Classification] = []
    ambiguous: list[tuple[str, str, list[Any]]] = []    # (note_path, snippet, rules)
    ambiguous_indices: list[int] = []

    for note_path in note_paths:
        current_folder = _current_folder(note_path)
        title = _note_title(note_path)
        idx_path = note_path.removesuffix(".md")

        # Extract concept profile from co-occurrence index
        concept_stems: dict[str, int] = {}
        if cooccur_store is not None:
            concept_stems = cooccur_store.note_nodes(idx_path)
            if not concept_stems:
                # Try without subpath prefix
                concept_stems = cooccur_store.note_nodes(title)

        # Fallback: note absent from the index (or no index at all) — tokenize
        # the body on the fly so classification never depends on index freshness.
        cached_body: str | None = None
        if not concept_stems:
            cached_body = _read_body(note_path)
            if cached_body:
                concept_stems = _stems_from_body(cached_body, effective_lang)

        # Retrieve note properties
        props = {}
        if props_map and note_path in props_map:
            props = props_map[note_path]
        else:
            try:
                from silica.driver import DRIVER
                props = DRIVER.props_of(note_path) or {}
            except Exception:
                pass

        best_folder, best_score, evidence, best_themes = _score_note_against_rules(
            note_path, concept_stems, taxonomy, props, lang=effective_lang
        )

        if best_score >= tau_high:
            # Clear L1 winner
            results.append(Classification(
                note_path=note_path,
                current_folder=current_folder,
                target_folder=best_folder,
                confidence=best_score,
                evidence=evidence,
                needs_move=current_folder != best_folder,
                title=title,
                rule_themes=best_themes,
            ))
        elif best_score < tau_low:
            # No signal — uncategorized
            results.append(Classification(
                note_path=note_path,
                current_folder=current_folder,
                target_folder=taxonomy.uncategorized,
                confidence=best_score,
                evidence="uncategorized",
                needs_move=move_uncategorized and current_folder != taxonomy.uncategorized,
                title=title,
            ))
        else:
            # Ambiguous — placeholder; filled by L2 or left as best L1 guess
            placeholder = Classification(
                note_path=note_path,
                current_folder=current_folder,
                target_folder=best_folder,
                confidence=best_score,
                evidence=evidence,
                needs_move=current_folder != best_folder,
                title=title,
                rule_themes=best_themes,
            )
            ambiguous_indices.append(len(results))
            results.append(placeholder)
            # Collect snippet for LLM context: first 300 chars of note body
            if cached_body is None:
                cached_body = _read_body(note_path)
            snippet = (cached_body or "")[:300]
            ambiguous.append((note_path, snippet, taxonomy.rules))

    # --- L2 pass (LLM arbiter) ---
    if llm_arbiter and ambiguous:
        logger.info(
            "classify: %d/%d notes in ambiguous band — sending to LLM arbiter",
            len(ambiguous), len(note_paths),
        )
        llm_choices = _llm_arbitrate(ambiguous, taxonomy)
        for list_idx, note_path in zip(ambiguous_indices, [a[0] for a in ambiguous]):
            chosen_folder = llm_choices.get(note_path, taxonomy.uncategorized)
            c = results[list_idx]
            current = c.current_folder
            results[list_idx] = Classification(
                note_path=c.note_path,
                current_folder=current,
                target_folder=chosen_folder,
                confidence=1.0,
                evidence="llm",
                needs_move=(
                    current != chosen_folder
                    and (move_uncategorized or chosen_folder != taxonomy.uncategorized)
                ),
                title=c.title,
                rule_themes=c.rule_themes,
            )

    return results
