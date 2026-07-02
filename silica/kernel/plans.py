"""plans — deterministic plan-status accounting for plans/ notes.

Twin of codedocs: walks plans/*.md frontmatter, counts by `status:` enum.
No LLM, no git. Status outside the enum is ignored here (the linter warns).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from silica.kernel import frontmatter

VALID_STATUS = {"todo", "in-progress", "blocked", "done"}


def iter_plan_notes(vault: Path | str):
    """Yield (note_path, data) for every note under plans/ with frontmatter."""
    vault = Path(vault)
    plans_dir = vault / "plans"
    if not plans_dir.is_dir():
        return
    for md in sorted(plans_dir.rglob("*.md")):
        try:
            content = md.read_text(encoding="utf-8")
        except OSError:
            continue
        data, _, _ = frontmatter.split(content)
        if not data:
            continue
        yield md, data


def status_counts(vault: Path | str) -> dict[str, int]:
    """Count plans per valid status. Empty dict when there are no plans."""
    statuses = (str(data.get("status") or "").strip() for _, data in iter_plan_notes(vault))
    return dict(Counter(s for s in statuses if s in VALID_STATUS))
