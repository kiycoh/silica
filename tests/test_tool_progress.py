from unittest.mock import patch, MagicMock, PropertyMock
import pytest
from silica.config import CONFIG
from silica.cli import _handle_slash_command
from silica.agent.progress import make_progress_callback, _redact, _cap
from silica.agent.events import ToolStartEvent, ToolCompleteEvent, ToolErrorEvent
from silica.agent.loop import run_agent
from silica.agent.llm import LLMResponse, ToolCall
from silica.tools import TOOLS

def test_verbose_slash_command_cycle():
    orig_mode = CONFIG.tool_progress
    try:
        CONFIG.tool_progress = "off"
        messages = []
        
        # off -> new
        _handle_slash_command("/verbose", messages)
        assert CONFIG.tool_progress == "new"
        
        # new -> all
        _handle_slash_command("/verbose", messages)
        assert CONFIG.tool_progress == "all"
        
        # all -> verbose
        _handle_slash_command("/verbose", messages)
        assert CONFIG.tool_progress == "verbose"
        
        # verbose -> off
        _handle_slash_command("/verbose", messages)
        assert CONFIG.tool_progress == "off"
    finally:
        CONFIG.tool_progress = orig_mode


def test_callback_noop_when_off(capsys):
    orig_mode = CONFIG.tool_progress
    CONFIG.tool_progress = "off"
    
    try:
        cb = make_progress_callback()
        event = ToolStartEvent(name="test_tool", args={}, call_id="1", iteration=1)
        cb(event)
        
        captured = capsys.readouterr()
        assert captured.out == ""
    finally:
        CONFIG.tool_progress = orig_mode


def test_redact_patterns():
    # Test credentials redaction
    assert "api_key=[REDACTED]" in _redact('api_key = "abc-123"')
    assert "token=[REDACTED]" in _redact('token:123')
    assert "secret=[REDACTED]" in _redact('"secret" : "password"')
    
    # Fail-closed test
    assert _redact(None) is None


def test_cap_behavior():
    # Text with many lines
    long_lines = "\n".join(f"line {i}" for i in range(20))
    capped_lines = _cap(long_lines, max_lines=5)
    assert "omitted" in capped_lines
    assert capped_lines.count("\n") <= 6
    
    # Text with many characters
    long_chars = "a" * 1000
    capped_chars = _cap(long_chars, max_chars=100)
    assert "omitted" in capped_chars
    assert len(capped_chars) <= 150


@patch("silica.agent.loop.call_llm")
def test_agent_loop_swallows_callback_exceptions(mock_call_llm):
    response1 = LLMResponse(
        text=None,
        tool_calls=[ToolCall(id="tc1", name="silica_read_note", args={"name": "test_note"})],
        assistant_message={"role": "assistant", "tool_calls": []},
        usage={}
    )
    response2 = LLMResponse(
        text="Final answer",
        tool_calls=[],
        assistant_message={"role": "assistant", "content": "Final answer"},
        usage={}
    )
    mock_call_llm.side_effect = [response1, response2]
    
    with patch.dict(TOOLS, {"silica_read_note": MagicMock()}):
        TOOLS["silica_read_note"].run.return_value = "note content"
        
        def bad_callback(event):
            raise ValueError("bad callback")
            
        messages = [{"role": "user", "content": "hello"}]
        
        ans = run_agent(messages, model="test_model", tool_progress_callback=bad_callback)
        assert ans == "Final answer"


def test_tool_error_event_always_emitted(capsys):
    orig_mode = CONFIG.tool_progress
    CONFIG.tool_progress = "off"
    
    try:
        cb = make_progress_callback()
        event = ToolErrorEvent(name="error_tool", call_id="1", error="Some error", iteration=1)
        cb(event)
        
        captured = capsys.readouterr()
        assert "error_tool" in captured.out
        assert "Some error" in captured.out
    finally:
        CONFIG.tool_progress = orig_mode


def test_callback_modes_output(capsys):
    orig_mode = CONFIG.tool_progress
    cb = make_progress_callback()
    
    try:
        # Test "new" mode
        CONFIG.tool_progress = "new"
        cb(ToolStartEvent(name="toolA", args={"x": 1}, call_id="1", iteration=1))
        captured = capsys.readouterr()
        assert "toolA" in captured.out
        
        # Same tool consecutive call should be skipped in "new" mode
        cb(ToolStartEvent(name="toolA", args={"x": 2}, call_id="2", iteration=2))
        captured = capsys.readouterr()
        assert captured.out == ""
        
        # Different tool should print
        cb(ToolStartEvent(name="toolB", args={"x": 1}, call_id="3", iteration=3))
        captured = capsys.readouterr()
        assert "toolB" in captured.out
        
        # Test "all" mode
        CONFIG.tool_progress = "all"
        cb(ToolStartEvent(name="toolA", args={"x": 1}, call_id="4", iteration=4))
        captured = capsys.readouterr()
        assert "toolA" in captured.out
        assert "x" in captured.out
        
        # Test "verbose" mode
        CONFIG.tool_progress = "verbose"
        cb(ToolStartEvent(name="toolA", args={"x": 1}, call_id="5", iteration=5))
        captured = capsys.readouterr()
        assert "args:" in captured.out
        
        cb(ToolCompleteEvent(name="toolA", args={"x": 1}, call_id="5", result="some result", duration_s=1.23, iteration=5))
        captured = capsys.readouterr()
        assert "result:" in captured.out
        assert "some result" in captured.out
        
    finally:
        CONFIG.tool_progress = orig_mode


def test_reasoning_event_renders_when_enabled(capsys):
    from silica.agent.events import ReasoningEvent
    from silica.agent.progress import make_progress_callback
    orig_thinking = CONFIG.show_thinking
    orig_tool_progress = CONFIG.tool_progress
    orig_verbose = CONFIG.verbose
    try:
        cb = make_progress_callback()
        event = ReasoningEvent(text="This is my deep reasoning process.", iteration=1)

        # Case 1: show_thinking=True, verbose=False, progress=all
        CONFIG.show_thinking = True
        CONFIG.verbose = False
        CONFIG.tool_progress = "all"
        cb(event)
        captured = capsys.readouterr()
        assert "thinking" in captured.out.lower()
        assert "reasoning" in captured.out.lower()

        # Case 2: show_thinking=False, verbose=False, progress=all
        CONFIG.show_thinking = False
        CONFIG.verbose = False
        CONFIG.tool_progress = "all"
        cb(event)
        captured = capsys.readouterr()
        assert captured.out == ""

        # Case 3: show_thinking=False, verbose=True, progress=all
        CONFIG.show_thinking = False
        CONFIG.verbose = True
        CONFIG.tool_progress = "all"
        cb(event)
        captured = capsys.readouterr()
        assert "thinking" in captured.out.lower()
        assert "reasoning" in captured.out.lower()

        # Case 4: show_thinking=False, verbose=False, progress=verbose
        CONFIG.show_thinking = False
        CONFIG.verbose = False
        CONFIG.tool_progress = "verbose"
        cb(event)
        captured = capsys.readouterr()
        assert "thinking" in captured.out.lower()
        assert "reasoning" in captured.out.lower()

    finally:
        CONFIG.show_thinking = orig_thinking
        CONFIG.tool_progress = orig_tool_progress
        CONFIG.verbose = orig_verbose


@patch("litellm.completion")
def test_llm_captures_reasoning(mock_completion):
    from silica.agent.llm import call_llm
    
    mock_message = MagicMock()
    mock_message.content = "My answer"
    mock_message.tool_calls = []
    mock_message.reasoning_content = "Thinking hard..."
    
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = {}
    
    mock_completion.return_value = mock_resp
    
    messages = [{"role": "user", "content": "hello"}]
    res = call_llm(model="test_model", messages=messages)
    
    assert res.reasoning == "Thinking hard..."
    assert res.text == "My answer"
    assert res.assistant_message == {"role": "assistant", "content": "My answer"}

    mock_message2 = MagicMock()
    mock_message2.content = "Answer with blocks"
    mock_message2.tool_calls = []
    mock_message2.reasoning_content = None
    mock_message2.thinking_blocks = [{"thinking": "Block reasoning"}]
    
    mock_choice2 = MagicMock()
    mock_choice2.message = mock_message2
    
    mock_resp2 = MagicMock()
    mock_resp2.choices = [mock_choice2]
    mock_resp2.usage = {}
    
    mock_completion.return_value = mock_resp2
    
    res2 = call_llm(model="test_model", messages=messages)
    assert res2.reasoning == "Block reasoning"
    assert res2.assistant_message == {"role": "assistant", "content": "Answer with blocks"}


@patch("litellm.completion")
def test_llm_openrouter_include_reasoning(mock_completion):
    from silica.agent.llm import call_llm
    
    mock_message = MagicMock()
    mock_message.content = "My answer"
    mock_message.tool_calls = []
    mock_message.reasoning_content = "Thinking hard..."
    
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = {}
    mock_completion.return_value = mock_resp
    
    messages = [{"role": "user", "content": "hello"}]
    
    orig_thinking = CONFIG.show_thinking
    orig_verbose = CONFIG.verbose
    try:
        # Test openrouter model with show_thinking=True
        CONFIG.show_thinking = True
        CONFIG.verbose = False
        call_llm(model="openrouter/some-model", messages=messages)
        mock_completion.assert_called_with(
            model="openrouter/some-model",
            messages=messages,
            include_reasoning=True,
            timeout=120.0
        )
        
        # Test openrouter model with show_thinking=False and verbose=True
        CONFIG.show_thinking = False
        CONFIG.verbose = True
        call_llm(model="openrouter/some-model", messages=messages)
        mock_completion.assert_called_with(
            model="openrouter/some-model",
            messages=messages,
            include_reasoning=True,
            timeout=120.0
        )
        
        # Test non-openrouter model
        call_llm(model="openai/gpt-4o", messages=messages)
        args, kwargs = mock_completion.call_args
        assert "include_reasoning" not in kwargs
        assert kwargs.get("timeout") == 120.0
        
    finally:
        CONFIG.show_thinking = orig_thinking
        CONFIG.verbose = orig_verbose


def test_thinking_slash_toggle():
    orig_thinking = CONFIG.show_thinking
    try:
        messages = []
        CONFIG.show_thinking = True
        _handle_slash_command("/thinking", messages)
        assert CONFIG.show_thinking is False
        
        _handle_slash_command("/thinking", messages)
        assert CONFIG.show_thinking is True
    finally:
        CONFIG.show_thinking = orig_thinking


def test_print_banner_styles(capsys):
    from silica.ui.banner import print_banner
    from rich.console import Console, ConsoleDimensions
    
    orig_style = CONFIG.banner_style
    try:
        # Minimal style
        CONFIG.banner_style = "minimal"
        print_banner()
        captured = capsys.readouterr()
        assert "silica" in captured.out
        assert "Obsidian-native agent" in captured.out

        # Wordmark/Crystal style with large terminal
        with patch.object(Console, "width", new_callable=PropertyMock, return_value=100), \
             patch.object(Console, "size", new_callable=PropertyMock, return_value=ConsoleDimensions(100, 40)):
            CONFIG.banner_style = "wordmark"
            print_banner()
            captured = capsys.readouterr()
            assert "___" in captured.out or "\\/" in captured.out
            assert "Obsidian-native agent" in captured.out

            # Since crystal is removed, it should behave like wordmark
            CONFIG.banner_style = "crystal"
            print_banner()
            captured = capsys.readouterr()
            assert "___" in captured.out or "\\/" in captured.out
            assert "Obsidian-native agent" in captured.out
    finally:
        CONFIG.banner_style = orig_style


