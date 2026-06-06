import logging
import os
import yaml
from pathlib import Path
from typing import Any, Dict

from silica.router.overlay import merge_overlay, OverlayError

logger = logging.getLogger(__name__)


def _domains_dir() -> Path:
    override = os.getenv("SILICA_DOMAINS_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "domains"


def load_recipe(recipe_name: str = "injector", domain: str | None = None) -> Dict[str, Any]:
    """Load the base recipe, optionally overlaid by a Domain Pack."""
    recipe_path = Path(__file__).resolve().parent.parent / "recipes" / f"{recipe_name}.yaml"
    if not recipe_path.exists():
        raise FileNotFoundError(f"Recipe file not found: {recipe_path}")
    with open(recipe_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    if not domain:
        return base

    pack_path = _domains_dir() / f"{domain}.yaml"
    if not pack_path.exists():
        logger.warning("Domain Pack '%s' not found at %s; using base recipe", domain, pack_path)
        return base
    with open(pack_path, "r", encoding="utf-8") as f:
        overlay = yaml.safe_load(f) or {}
    try:
        return merge_overlay(base, overlay)
    except OverlayError as e:
        logger.error("Domain Pack '%s' rejected: %s; using base recipe", domain, e)
        return base
