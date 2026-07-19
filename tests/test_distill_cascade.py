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


def _delegate_fsm(chunk, steer=None):
    """Minimal FSM stand-in that survives handle_delegate end-to-end."""
    fsm = SimpleNamespace()
    fsm._get_chunks_from_context_if_empty = MagicMock()
    fsm._chunks = [chunk]
    fsm._current_chunk_idx = 0
    fsm._current_file_idx = 0
    fsm._chunk_flat_to_fi_ci = {0: (0, 0)}
    fsm._file_chunks = {}
    fsm.inbox_file = "in.md"
    fsm.target_dir = "Notes"
    fsm.hub = "[[Hub]]"
    fsm.seen_override = None
    fsm.context = {"file_0_language": "English"}
    if steer:
        fsm.context["chunk_0_steer_context"] = steer
    fsm.manifest = MagicMock()
    fsm.manifest.titles.return_value = []
    fsm.progress = MagicMock()
    fsm.progress.digest.return_value = "digest"
    fsm.progress.started_at = "2026-07-18T00:00:00"
    fsm.progress.is_checkpoint_done.return_value = None
    fsm._chunk_task_id = lambda cap, idx=None: f"f0_c0_{cap}"
    fsm._prefetcher = None
    fsm._progress_note = MagicMock()
    fsm._make_tmp = MagicMock(return_value="/tmp/distill.json")
    fsm._chunk_ctx = {}
    fsm._transition_success = MagicMock()
    return fsm


def _chunk_one_concept():
    return {"schema_version": 1,
            "batches": [{"inbox_file": "in.md", "concepts": [{"name": "c", "excerpt": "x"}]}]}


def test_first_attempt_does_not_escalate():
    from silica.router.states import distill as d
    fsm = _delegate_fsm(_chunk_one_concept())
    with patch.object(d.orch.CONFIG, "distill_concurrency", 1), \
         patch.object(d, "run_distiller", return_value={"updates": []}) as rd, \
         patch("silica.kernel.episodic.capture_from_distill"):
        d.handle_delegate(fsm)
    assert rd.call_args.kwargs["escalate"] is False
    assert "escalations" not in fsm.context


def test_steer_retry_escalates_and_counts():
    from silica.router.states import distill as d
    fsm = _delegate_fsm(_chunk_one_concept(), steer="## Steering feedback\nfix op 1")
    with patch.object(d.orch.CONFIG, "distill_concurrency", 1), \
         patch.object(d, "run_distiller", return_value={"updates": []}) as rd, \
         patch("silica.kernel.episodic.capture_from_distill"):
        d.handle_delegate(fsm)
    assert rd.call_args.kwargs["escalate"] is True
    assert rd.call_args.kwargs["steer_context"].startswith("## Steering")
    assert fsm.context["escalations"] == 1


def test_empty_chunk_skips_distiller_but_completes():
    from silica.router.states import distill as d
    fsm = _delegate_fsm({"schema_version": 1, "batches": []})
    with patch.object(d.orch.CONFIG, "distill_concurrency", 1), \
         patch.object(d, "run_distiller") as rd, \
         patch("silica.kernel.episodic.capture_from_distill"):
        d.handle_delegate(fsm)
    rd.assert_not_called()
    fsm._make_tmp.assert_called_once_with({"updates": []})
    fsm._transition_success.assert_called_once()
