"""Cache-stable distiller prompt: byte-stable per-file prefix + cache markers."""
from unittest.mock import MagicMock, patch

from silica.kernel.prep_delegation import render_prompt, run_distiller


def test_render_prompt_explicit_language_skips_detection():
    a = render_prompt(target="Notes", language="Italian",
                      source_text="the quick brown fox jumps over the lazy dog")
    b = render_prompt(target="Notes", language="Italian",
                      source_text="cane gatto albero casa sole mare monte fiume")
    assert a == b  # different source samples, identical template
    assert "Italian" in a


def _fake_response(text='{"updates": []}'):
    r = MagicMock()
    r.text = text
    r.finish_reason = "stop"
    return r


def _capture_messages(**kwargs):
    """Run run_distiller with a fake provider; return the messages sent."""
    provider = MagicMock()
    provider.call_llm.return_value = _fake_response()
    with patch("silica.agent.providers.get_provider", return_value=provider), \
         patch("silica.agent.providers.model_limits", return_value=(262144, 8192)):
        run_distiller(payload={"schema_version": 1, "batches": []},
                      target="Notes", **kwargs)
    return provider.call_llm.call_args.kwargs["messages"]


def test_message_split_and_cache_markers():
    msgs = _capture_messages(language="English")
    assert [m["role"] for m in msgs] == ["system", "user"]
    sys_parts = msgs[0]["content"]
    assert isinstance(sys_parts, list)
    assert sys_parts[0]["cache_control"] == {"type": "ephemeral"}
    user_parts = msgs[1]["content"]
    assert user_parts[0]["cache_control"] == {"type": "ephemeral"}
    assert len(user_parts) == 1  # no steer → single ctx part


def test_steer_is_separate_trailing_part_and_prefix_stable():
    base = _capture_messages(language="English")
    steered = _capture_messages(language="English",
                                steer_context="## Steering feedback\nfix op 2")
    # system message byte-identical across retry
    assert steered[0] == base[0]
    # ctx part byte-identical; steer appended as its own part
    assert steered[1]["content"][0] == base[1]["content"][0]
    assert len(steered[1]["content"]) == 2
    assert "Steering feedback" in steered[1]["content"][1]["text"]


def test_system_message_stable_across_chunks_of_same_file():
    a = _capture_messages(language="English",
                          ledger_digest="chunk 1 done", substrate="## Related\nA")
    b = _capture_messages(language="English",
                          ledger_digest="chunks 1-2 done", substrate="## Related\nB")
    assert a[0] == b[0]  # dynamic content never touches the system prefix
