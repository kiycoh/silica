"""L1 Taxonomy schema — declarative folder↔theme mapping for /organize.

The taxonomy is a YAML file that declares a list of FolderRule entries,
each binding a vault folder to a set of themes/keywords/concepts.

The LLM generates this file from a natural-language prompt; the kernel
validates and applies it deterministically via Pydantic.

Default path: {vault_path}/_silica/taxonomy.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class MetadataFilter(BaseModel):
    """Filter to match note frontmatter or properties."""

    key: str = Field(description="Frontmatter key, e.g. 'date', 'created', 'tags'")
    operator: str = Field(description="Comparison operator, e.g. 'equals', 'contains', 'year_equals', 'year_greater_than', 'year_less_than'")
    value: Any = Field(description="Value to compare against")


class FolderRule(BaseModel):
    """A single taxonomy rule: binds a vault folder to a set of themes."""

    folder: str = Field(
        description="Vault-relative destination folder path, e.g. 'Concepts/AI'"
    )
    themes: list[str] = Field(
        default_factory=list,
        description="Semantic labels for this folder (used for concept matching)",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Exact-match terms — presence in note title/tags bumps score",
    )
    description: str = Field(
        default="",
        description="Human-readable description (used in LLM generation prompt)",
    )
    metadata_filters: list[MetadataFilter] = Field(
        default_factory=list,
        description="Metadata filters that a note must pass to match this rule",
    )

    def keyword_set(self) -> frozenset[str]:
        """Lower-cased keyword set for O(1) lookup."""
        return frozenset(k.lower() for k in self.keywords)


class Taxonomy(BaseModel):
    """Validated taxonomy document parsed from YAML."""

    version: int = Field(default=1)
    rules: list[FolderRule] = Field(default_factory=list)
    uncategorized: str = Field(
        default="Uncategorized",
        description="Fallback folder for notes that match no rule",
    )
    scope: str = Field(
        default="",
        description="Restrict classification to this vault subfolder (empty = vault-wide)",
    )

    @model_validator(mode="after")
    def enforce_scope_prefixes(self) -> "Taxonomy":
        if self.scope:
            prefix = self.scope.replace("\\", "/").rstrip("/")
            
            # Ensure uncategorized starts with scope
            uncat_path = self.uncategorized.replace("\\", "/")
            if not (uncat_path == prefix or uncat_path.startswith(prefix + "/")):
                self.uncategorized = f"{prefix}/{self.uncategorized.lstrip('/')}"
            
            # Ensure each rule folder starts with scope
            for rule in self.rules:
                r_folder = rule.folder.replace("\\", "/")
                if not (r_folder == prefix or r_folder.startswith(prefix + "/")):
                    rule.folder = f"{prefix}/{rule.folder.lstrip('/')}"
        return self

    # --- I/O ---

    @classmethod
    def from_yaml(cls, path: Path) -> "Taxonomy":
        """Parse and validate a taxonomy YAML file."""
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        return cls.model_validate(raw)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Taxonomy":
        """Parse and validate from a dict (e.g. from an LLM response)."""
        return cls.model_validate(data)

    def to_yaml(self, path: Path) -> None:
        """Serialise to a YAML file, creating parent dirs as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = self.model_dump()
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(doc, fh, allow_unicode=True, sort_keys=False, default_flow_style=False)

    # --- Matching ---

    def best_folder(self, themes: list[str], title: str = "") -> tuple[str, float]:
        """Return (folder, score) for the best-matching rule.

        Scoring (additive, capped at 1.0):
          - Keyword exact match in title → +0.4 per keyword hit (capped at 0.8)
          - Theme overlap: proportion of rule themes matching concept set → [0, 0.6]

        Returns (uncategorized, 0.0) when no rule matches at all.
        """
        if not self.rules:
            return self.uncategorized, 0.0

        concept_set = frozenset(t.lower() for t in themes)
        title_lower = title.lower()

        best_folder = self.uncategorized
        best_score = 0.0

        for rule in self.rules:
            score = 0.0

            # Keyword hits in title
            kw_hits = sum(1 for k in rule.keyword_set() if k in title_lower)
            score += min(kw_hits * 0.4, 0.8)

            # Theme overlap: what fraction of the rule's themes appear in concept_set?
            if rule.themes:
                rule_themes_lower = frozenset(t.lower() for t in rule.themes)
                overlap = len(concept_set & rule_themes_lower)
                # Partial credit: any single matching theme already scores 0.3
                theme_score = overlap / len(rule_themes_lower) * 0.6 if overlap > 0 else 0.0
                score += theme_score

            if score > best_score:
                best_score = score
                best_folder = rule.folder

        return best_folder, round(best_score, 4)

    def folder_for(self, themes: list[str], title: str = "") -> str:
        """Convenience: return only the folder name (no score)."""
        folder, _ = self.best_folder(themes, title=title)
        return folder


# ---------------------------------------------------------------------------
# Default taxonomy path helper
# ---------------------------------------------------------------------------

_DEFAULT_TAXONOMY_REL = "_silica/taxonomy.yaml"


def default_taxonomy_path() -> Path:
    """Return the default taxonomy path inside the configured vault."""
    from silica.config import CONFIG
    vault = getattr(CONFIG, "vault_path", "") or ""
    if vault:
        return Path(vault) / _DEFAULT_TAXONOMY_REL
    return Path.cwd() / _DEFAULT_TAXONOMY_REL


def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    """Load the taxonomy from disk; fall back to an empty taxonomy on error."""
    import logging
    logger = logging.getLogger(__name__)
    from silica.config import CONFIG

    if path:
        p = Path(path)
        vault = getattr(CONFIG, "vault_path", "") or ""
        if not p.is_absolute() and vault:
            vp = Path(vault) / p
            if vp.exists():
                p = vp
    else:
        p = default_taxonomy_path()

    if not p.exists():
        logger.warning("taxonomy: file not found at %s — using empty taxonomy", p)
        return Taxonomy()
    try:
        return Taxonomy.from_yaml(p)
    except Exception as exc:
        logger.error("taxonomy: failed to parse %s: %s — using empty taxonomy", p, exc)
        return Taxonomy()


# ---------------------------------------------------------------------------
# LLM-generation prompt helper
# ---------------------------------------------------------------------------

TAXONOMY_GENERATION_PROMPT = """\
You are an expert knowledge manager. The user wants to reorganize their Obsidian vault.
Given their natural-language intent and the list of actual note titles present in their vault below, generate a taxonomy.yaml document that maps vault folders to thematic categories.

Rules for generation:
1. Each folder must be a valid vault-relative path (use forward slashes, no leading slash).
2. If the user intent specifies organizing a specific subfolder/input folder (e.g. "organize the Research Notes folder") or a scope is provided, set scope to that folder name (e.g. scope: "Research Notes") and ensure all rule folders (and uncategorized) are subfolders under that scope (e.g. "Research Notes/DeepSeek").
3. themes: a list of 3-8 specific semantic labels for that folder.
4. keywords: a list of 1-5 exact terms that unambiguously signal membership.
5. metadata_filters: a list of filters if the user intent specifies constraints on metadata (e.g., date, year, tags, etc.). Each filter has:
   - key: the property name (e.g., "date", "created", "tags")
   - operator: comparison operator ("equals", "contains", "year_equals", "year_greater_than", "year_less_than")
   - value: the value to compare against (e.g. 2026, "recipe")
   For example, if the user wants documents written in 2026:
     metadata_filters:
       - key: "date"
         operator: "year_equals"
         value: 2026
6. Include an 'uncategorized' fallback folder for notes that match nothing. If scope is set, uncategorized should be inside the scope (e.g. "Research Notes/Uncategorized").
7. Keep the list of rules minimal and non-overlapping.
8. Align the rules and keywords with the actual note titles provided. Create folders, themes, and keywords that cover the diverse topics represented by the notes (e.g., if you see notes about BDI, FIPA-ACL, GAIA, robotics, neural networks, or coordination, create corresponding folders and keywords for them).

Return ONLY a YAML document (no markdown fences, no commentary) matching this schema:
```
version: 1
scope: ""               # optional: restrict to a vault subfolder
uncategorized: "Uncategorized"
rules:
  - folder: "Concepts/AI"
    themes: ["machine learning", "deep learning", "transformers"]
    keywords: ["LLM", "GPT"]
    description: "AI and ML concepts"
    metadata_filters: []
  - folder: "Archive/2026"
    themes: ["documents", "records"]
    keywords: []
    description: "Documents written in 2026"
    metadata_filters:
      - key: "date"
         operator: "year_equals"
         value: 2026
```

User intent: {user_intent}

Scope: {scope}

Actual note titles in scope:
{note_titles}
"""


TAXONOMY_MERGE_BLOCK = """\

Existing taxonomy (the user's current standing rules):
{existing_yaml}

Merge instructions:
1. Treat the existing rules as standing directives: preserve them unless the new user intent explicitly contradicts or refines them.
2. Add or modify only what the new intent requires; do not drop, rename, or reword unrelated folders, themes, or keywords.
3. Keep the existing 'scope' and 'uncategorized' values unless the new intent overrides them.
4. Return the COMPLETE merged taxonomy (existing rules + changes), not a delta.
"""
