"""Distiller cascade (Tier 2 cost): escalation config, provider role, run_distiller routing."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from silica.agent.providers import get_provider
from silica.config import SilicaConfig


def test_escalation_fields_default_unset():
    cfg = SilicaConfig()
    assert cfg.distill_escalation_model is None
    assert cfg.distill_escalation_provider is None


def test_escalation_provider_derived_from_prefix():
    cfg = SilicaConfig()
    cfg.distill_escalation_model = "openrouter/deepseek/deepseek-v4"
    assert cfg.distill_escalation_provider == "openrouter"


def test_escalation_provider_bare_model_defaults_lmstudio():
    cfg = SilicaConfig()
    cfg.distill_escalation_model = "qwen3-30b"
    assert cfg.distill_escalation_provider == "lmstudio"


def test_escalation_role_uses_escalation_config():
    cfg = SimpleNamespace(
        distill_escalation_provider="openrouter",
        distill_escalation_model="openrouter/big/model",
        provider="lmstudio", model="local-model",
    )
    p = get_provider(cfg, role="escalation")
    assert p.model == "big/model"  # provider prefix stripped, vendor path kept


def test_escalation_role_falls_back_to_router():
    cfg = SimpleNamespace(
        distill_escalation_provider=None, distill_escalation_model=None,
        provider="lmstudio", model="qwen3-30b",
    )
    p = get_provider(cfg, role="escalation")
    assert p.model == "qwen3-30b"
