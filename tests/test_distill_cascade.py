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


import os

from silica.config import CONFIG
from silica.kernel.prep_delegation import run_distiller


def _fake_response(text='{"updates": []}'):
    r = MagicMock()
    r.text = text
    r.finish_reason = "stop"
    return r


def _run(escalate):
    """run_distiller with a fake provider; return (get_provider, model_limits, provider) mocks."""
    provider = MagicMock()
    provider.call_llm.return_value = _fake_response()
    with patch.dict(os.environ, {"MODEL_CONTEXT_WINDOW": "0", "DISTILLER_MAX_TOKENS": "0"}), \
         patch("silica.agent.providers.get_provider", return_value=provider) as gp, \
         patch("silica.agent.providers.model_limits", return_value=(262144, 8192)) as ml:
        run_distiller(payload={"schema_version": 1, "batches": []},
                      target="Notes", language="English", escalate=escalate)
    return gp, ml, provider


def test_default_call_uses_worker_role_and_distiller_pin():
    gp, _ml, provider = _run(escalate=False)
    assert gp.call_args.kwargs.get("role") == "worker"
    assert provider.call_llm.call_args.kwargs["openrouter_provider"] == \
        CONFIG.openrouter_provider_distiller


def test_escalated_call_uses_escalation_role_and_drops_pin():
    gp, _ml, provider = _run(escalate=True)
    assert gp.call_args.kwargs.get("role") == "escalation"
    assert provider.call_llm.call_args.kwargs["openrouter_provider"] is None


def test_escalated_limits_resolve_escalation_model():
    with patch.object(CONFIG, "distill_escalation_model", "openrouter/big/model"), \
         patch.object(CONFIG, "_distill_escalation_provider", "openrouter"):
        _gp, ml, _provider = _run(escalate=True)
    assert ml.call_args.args[1] == "openrouter/big/model"
