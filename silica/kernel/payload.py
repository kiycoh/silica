# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

import re
from typing import Any
from silica.driver import DRIVER

DEFAULT_WINDOW = 450          # Chars on each side of a non-heading concept match.
MAX_EXCERPT_CHARS = 2000      # Hard per-excerpt cap.
MAX_OCCURRENCES = 2           # Max non-overlapping windows per concept per file.
FULL_INCLUDE_THRESHOLD = 6000 # Include whole note below this.

def classify_action(collision: dict | None, in_new_concepts: bool) -> str:
    """Mechanical action hint for a concept."""
    if in_new_concepts:
        return "create"
    if not collision:
        return "skip"
    if collision.get("best_match") == "title":
        return "enrich"
    if collision.get("total_hits", 0) >= 3:
        return "review"
    return "likely_skip"

def compile_concept_regex(c: str) -> re.Pattern:
    escaped = re.escape(c)
    start_b = r'\b' if c and re.match(r'\w', c) else ''
    end_b = r'\b' if c and re.search(r'\w$', c) else ''
    return re.compile(rf'{start_b}{escaped}{end_b}', re.IGNORECASE)

def find_heading(content: str, concept: str):
    escaped = re.escape(concept)
    start_b = r'\b' if concept and re.match(r'\w', concept) else ''
    end_b = r'\b' if concept and re.search(r'\w$', concept) else ''
    pattern = re.compile(
        rf'^(#{{1,4}})\s+.*{start_b}{escaped}{end_b}.*$',
        re.IGNORECASE | re.MULTILINE,
    )
    return pattern.search(content)

def extract_section(content: str, heading_match) -> str:
    level = len(heading_match.group(1))
    next_pattern = re.compile(rf'^#{{1,{level}}}(?!#)\s+', re.MULTILINE)
    next_match = next_pattern.search(content, pos=heading_match.end())
    end = next_match.start() if next_match else len(content)
    return content[heading_match.start():end].strip()

def expand_to_double_newline(content: str, start: int, end: int) -> tuple[int, int]:
    new_start = content.rfind('\n\n', 0, start)
    if new_start == -1:
        new_start = 0
    else:
        new_start += 2
    new_end = content.find('\n\n', end)
    if new_end == -1:
        new_end = len(content)
    return new_start, new_end

def safe_truncate(text: str, max_chars: int) -> str:
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
    pattern = compile_concept_regex(concept)
    windows: list[str] = []
    last_end = -1
    for m in pattern.finditer(content):
        if len(windows) >= max_occ:
            break
        start = max(0, m.start() - window)
        end = min(len(content), m.end() + window)
        start, end = expand_to_double_newline(content, start, end)
        if start < last_end:
            continue
        windows.append(content[start:end].strip())
        last_end = end
    return windows

def extract_excerpt_from_content(content: str, concept: str, window: int) -> str:
    if not content:
        return ""
    from silica.kernel.media import strip_images
    content = strip_images(content)
    heading = find_heading(content, concept)
    if heading:
        return safe_truncate(extract_section(content, heading), MAX_EXCERPT_CHARS)
    windows = extract_windows(content, concept, window, MAX_OCCURRENCES)
    if not windows:
        return ""
    return safe_truncate("\n\n[...]\n\n".join(windows), MAX_EXCERPT_CHARS)

def extract_excerpt_from_note(note_name: str, concept: str, window: int) -> str:
    try:
        nc = DRIVER.read_note(note_name)
        return extract_excerpt_from_content(nc.content, concept, window)
    except RuntimeError:
        return ""

def vault_content_or_excerpt(note_name: str, concept: str, window: int, is_title_match: bool) -> str:
    try:
        nc = DRIVER.read_note(note_name)
        if is_title_match and len(nc.content) <= FULL_INCLUDE_THRESHOLD:
            return nc.content.strip()
        return extract_excerpt_from_content(nc.content, concept, window)
    except RuntimeError:
        return ""

def build_concept_entry(
    name: str,
    inbox_content: str,
    collision: dict | None,
    in_new_concepts: bool,
    window: int,
) -> dict:
    entry: dict[str, Any] = {
        "name": name,
        # collision=None WITHOUT in_new_concepts is hardcoded "create" (legacy).
        "action_hint": classify_action(collision, in_new_concepts)
        if (collision is not None or in_new_concepts) else "create",
        "inbox_excerpt": extract_excerpt_from_content(inbox_content, name, window),
    }
    if collision and collision.get("hits"):
        best = collision["hits"][0]
        is_title = collision.get("best_match") == "title"
        entry["vault_collision"] = {
            "path": best["path"],
            "match_type": collision.get("best_match"),
            "total_hits": collision.get("total_hits", 0),
            "excerpt": vault_content_or_excerpt(best["path"], name, window, is_title),
        }
    else:
        entry["vault_collision"] = None
    return entry

def build_payload(recon_reports: list, window: int) -> dict:
    batches = []
    for report in recon_reports:
        inbox_name = report["file"]
        try:
            inbox_content = DRIVER.read_note(inbox_name).content
        except RuntimeError:
            inbox_content = ""
        concepts = []
        for collision in report.get("collisions", []):
            concepts.append(build_concept_entry(
                name=collision["name"],
                inbox_content=inbox_content,
                collision=collision,
                in_new_concepts=False,
                window=window,
            ))
        for new_name in sorted(report.get("new_concepts", [])):
            concepts.append(build_concept_entry(
                name=new_name,
                inbox_content=inbox_content,
                collision=None,
                in_new_concepts=True,
                window=window,
            ))
        batches.append({"inbox_file": inbox_name, "concepts": concepts})
    return {"schema_version": 1, "batches": batches}
