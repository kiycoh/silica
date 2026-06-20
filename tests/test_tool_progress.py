from unittest.mock import patch, MagicMock, PropertyMock
import pytest
from silica.config import CONFIG
from silica.cli import _handle_slash_command
from silica.ui.renderer import make_progress_callback, _redact, _cap
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
        cb(ToolStartEvent(name="silica_read_note", args={"name": "noteA"}, call_id="1", iteration=1))
        captured = capsys.readouterr()
        assert "Reading note" in captured.out
        assert "noteA" in captured.out
        
        # Same tool consecutive call should be skipped in "new" mode
        cb(ToolStartEvent(name="silica_read_note", args={"name": "noteB"}, call_id="2", iteration=2))
        captured = capsys.readouterr()
        assert captured.out == ""
        
        # Different tool should print
        cb(ToolStartEvent(name="silica_search", args={"query": "searchQ"}, call_id="3", iteration=3))
        captured = capsys.readouterr()
        assert "Searching notes" in captured.out
        assert "searchQ" in captured.out
        
        # Test "all" mode
        CONFIG.tool_progress = "all"
        cb(ToolStartEvent(name="silica_read_note", args={"name": "noteA"}, call_id="4", iteration=4))
        captured = capsys.readouterr()
        assert "Reading note" in captured.out
        assert "noteA" in captured.out
        
        # Test "verbose" mode
        CONFIG.tool_progress = "verbose"
        cb(ToolStartEvent(name="silica_read_note", args={"name": "noteA"}, call_id="5", iteration=5))
        captured = capsys.readouterr()
        assert "Reading note" in captured.out
        assert "noteA" in captured.out
        
        cb(ToolCompleteEvent(name="silica_read_note", args={"name": "noteA"}, call_id="5", result="some result", duration_s=1.23, iteration=5))
        captured = capsys.readouterr()
        assert "Reading note" in captured.out
        assert "some result" in captured.out
        
    finally:
        CONFIG.tool_progress = orig_mode


def test_reasoning_event_renders_when_enabled(capsys):
    from silica.agent.events import ReasoningEvent
    from silica.ui.renderer import make_progress_callback
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
    assert res.assistant_message["role"] == "assistant"
    assert res.assistant_message["content"] == "My answer"
    assert res.assistant_message["reasoning_content"] == "Thinking hard..."

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
    assert res2.assistant_message["role"] == "assistant"
    assert res2.assistant_message["content"] == "Answer with blocks"
    assert res2.assistant_message["thinking_blocks"] == [{"thinking": "Block reasoning"}]


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
            max_tokens=256000,
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
            max_tokens=256000,
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


def test_stage_track_centers_on_running_phase():
    from silica.ui.renderer import _stage_track
    phases = [
        {"phase": "recon",      "status": "done",    "elapsed": 1.0},
        {"phase": "cross-dedup","status": "done",    "elapsed": 0.5},
        {"phase": "payload",    "status": "done",    "elapsed": 0.8},
        {"phase": "salience",   "status": "done",    "elapsed": 0.3},
        {"phase": "collision",  "status": "running", "elapsed": None},
    ]
    track = _stage_track(phases, console_width=120)
    plain = track.plain
    # Running phase is visible
    assert "◉ collision" in plain
    # Phases in window are visible (collision is index 4; window is indices 1-7)
    assert "✓ payload" in plain
    assert "✓ salience" in plain
    # Phases outside window are NOT visible (recon is index 0, outside window start=1)
    assert "✓ recon" not in plain
    # Leading ellipsis present because window doesn't start at 0
    assert plain.startswith("…")


def test_stage_track_empty_shows_pending_from_start():
    from silica.ui.renderer import _stage_track
    track = _stage_track([], console_width=120)
    plain = track.plain
    # No running phase → center=0, window starts at 0, no leading ellipsis
    assert not plain.startswith("…")
    assert "· recon" in plain
    assert "· cross-dedup" in plain


def test_injector_panel_height_constant_across_widths():
    """Regression: the injector panel (Panel > Group > stage track) must keep the same
    height on a narrow vs wide console. A wrapping track grew the panel between frames
    and tore the Live region on a small / non-fullscreen terminal."""
    import io
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.text import Text
    from silica.ui.renderer import _stage_track
    phases = [
        {"phase": "payload",   "status": "done",    "elapsed": 0.8},
        {"phase": "salience",  "status": "done",    "elapsed": 0.3},
        {"phase": "collision", "status": "running", "elapsed": None},
    ]

    def panel_height(width: int) -> int:
        buf = io.StringIO()
        c = Console(file=buf, width=width)
        c.print(Panel(Group(Text("  spinner…"), _stage_track(phases, width)),
                      title="injector · some/long/inbox/path/with a long file name.md"))
        return len(buf.getvalue().rstrip("\n").split("\n"))

    heights = {w: panel_height(w) for w in (30, 50, 80, 120)}
    assert len(set(heights.values())) == 1, f"panel height varies with width: {heights}"


def test_stage_track_failed_phase_shown():
    from silica.ui.renderer import _stage_track
    phases = [
        {"phase": "recon",      "status": "done",   "elapsed": 1.0},
        {"phase": "cross-dedup","status": "failed",  "elapsed": 0.2},
    ]
    track = _stage_track(phases, console_width=120)
    plain = track.plain
    assert "✗ cross-dedup" in plain


def _make_mock_batch(kind: str = "refine") -> dict:
    """Minimal _batch dict for micro-phase tests (bypasses terminal-gated BatchRunStartEvent)."""
    from unittest.mock import MagicMock
    return {
        "run_id": "r1", "kind": kind, "label": "Concepts",
        "total": 5, "done": 0, "failed": 0,
        "start_time": 0.0, "progress_obj": MagicMock(),
        "task_id": 0, "current_label": "", "micro_phase": "",
    }


def test_batch_micro_phase_tracked_from_work_feedback():
    import silica.agent.bus as bus_mod
    from silica.agent.events import WorkFeedbackEvent
    from silica.ui.renderer import make_progress_callback

    orig_bus = bus_mod.BUS
    bus_mod.BUS = bus_mod.EventBus()
    try:
        cb = make_progress_callback()
        cb._batch = _make_mock_batch("refine")

        bus_mod.BUS.publish("work/feedback", WorkFeedbackEvent(
            item_id="i1", kind="refine", phase="calling_llm"
        ))
        assert cb._batch["micro_phase"] == "calling_llm"
    finally:
        cb.close()
        bus_mod.BUS = orig_bus


def test_batch_micro_phase_ignored_for_wrong_kind():
    import silica.agent.bus as bus_mod
    from silica.agent.events import WorkFeedbackEvent
    from silica.ui.renderer import make_progress_callback

    orig_bus = bus_mod.BUS
    bus_mod.BUS = bus_mod.EventBus()
    try:
        cb = make_progress_callback()
        cb._batch = _make_mock_batch("refine")

        bus_mod.BUS.publish("work/feedback", WorkFeedbackEvent(
            item_id="i1", kind="dedup", phase="reading"
        ))
        assert cb._batch["micro_phase"] == ""
    finally:
        cb.close()
        bus_mod.BUS = orig_bus


def test_batch_micro_phase_resets_on_ledger_next_complete():
    import json
    import silica.agent.bus as bus_mod
    from silica.agent.events import WorkFeedbackEvent, ToolCompleteEvent
    from silica.ui.renderer import make_progress_callback

    orig_bus = bus_mod.BUS
    orig_mode = CONFIG.tool_progress
    bus_mod.BUS = bus_mod.EventBus()
    try:
        CONFIG.tool_progress = "all"  # must not be "off" — ToolCompleteEvent is behind mode guard
        cb = make_progress_callback()
        cb._batch = _make_mock_batch("refine")

        # Set micro phase via BUS
        bus_mod.BUS.publish("work/feedback", WorkFeedbackEvent(
            item_id="i1", kind="refine", phase="committing"
        ))
        assert cb._batch["micro_phase"] == "committing"

        # ledger_next complete (not done — advances done counter, resets micro_phase)
        cb(ToolCompleteEvent(
            name="silica_ledger_next",
            args={},
            call_id="c1",
            result=json.dumps({"done": False, "payload": {"note_paths": ["a/b.md"]}}),
            duration_s=0.1,
            iteration=1,
        ))
        assert cb._batch["micro_phase"] == ""
    finally:
        cb.close()
        bus_mod.BUS = orig_bus
        CONFIG.tool_progress = orig_mode


def test_live_aware_handler_resolves_stderr_dynamically():
    """Regression for torn panels: a log handler that caches ``sys.stderr`` at
    construction writes raw bytes while a ``rich.Live`` has redirected stderr to its
    coordinating proxy → the live region tears. The handler must resolve ``sys.stderr``
    at emit time so Live can print the log above the region cleanly."""
    import sys, io
    from silica.ui.logging import LiveAwareStreamHandler

    h = LiveAwareStreamHandler()
    orig = sys.stderr
    proxy = io.StringIO()  # stand-in for rich.Live's FileProxy
    try:
        sys.stderr = proxy  # simulate Live.start() redirect
        assert h.stream is proxy
    finally:
        sys.stderr = orig
    # Live.stop() restores stderr → handler follows it back, no stale reference.
    assert h.stream is orig


def test_injector_progress_not_given_its_own_live():
    """Regression: the embedded Progress must NOT own an active Live. Progress.start()
    opens a second Live on the global console that double-renders the bar — an orphan
    progress bar above the panel on small consoles. The outer self._live drives it."""
    from unittest.mock import patch, PropertyMock
    from rich.console import Console
    from silica.agent.events import ToolStartEvent
    from silica.ui.renderer import make_progress_callback

    orig_mode = CONFIG.tool_progress
    cb = make_progress_callback()
    try:
        CONFIG.tool_progress = "all"
        with patch.object(Console, "is_terminal", new_callable=PropertyMock, return_value=True), \
             patch.object(cb, "_update_live", lambda: None):  # isolate: no real outer Live
            cb(ToolStartEvent(name="silica_run_injector",
                              args={"inbox_files": ["a.md"]}, call_id="1", iteration=1))
        assert cb._inject_progress is not None
        assert cb._inject_progress.live.is_started is False
    finally:
        cb.close()
        CONFIG.tool_progress = orig_mode


def test_inject_progress_does_not_overflow_on_phase_refires():
    """Regression: a phase can transition to `done` more than once within a single
    injector run (retries / deferred reprocessing). The progress counter must track
    DISTINCT completed phases, not accumulate every `done` event — otherwise it shows
    nonsense like 68/16."""
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn,
    )
    from silica.ui.renderer import make_progress_callback, _PHASE_LABELS

    cb = make_progress_callback()
    try:
        # Mimic what ToolStartEvent(silica_run_injector) sets up, without the
        # terminal-gated path.
        cb._inject_progress = Progress(
            SpinnerColumn(), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
            auto_refresh=False,
        )
        cb._inject_task_id = cb._inject_progress.add_task("", total=len(_PHASE_LABELS))
        cb._pipeline_phases = []
        cb._phase_start_times = {}

        phases = ["payload", "salience", "collision"]

        def run_pass():
            for p in phases:
                cb._on_pipeline_phase(p, "running", None)
                cb._on_pipeline_phase(p, "done", 0.1)

        # Three passes over the same phases (e.g. validate fallback re-runs the chain).
        run_pass()
        run_pass()
        run_pass()

        completed = cb._inject_progress.tasks[cb._inject_task_id].completed
        # 3 distinct phases completed — NOT 9 (3 passes × 3), and never above total.
        assert completed == len(phases)
        assert completed <= len(_PHASE_LABELS)
    finally:
        cb.close()


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
        assert "Your personal note curator agent" in captured.out

        # Wordmark/Crystal style with large terminal
        with patch.object(Console, "width", new_callable=PropertyMock, return_value=100), \
             patch.object(Console, "size", new_callable=PropertyMock, return_value=ConsoleDimensions(100, 40)):
            CONFIG.banner_style = "wordmark"
            print_banner()
            captured = capsys.readouterr()
            # wordmark renders multi-line ASCII/block art; minimal would be a single line
            assert len(captured.out.splitlines()) > 2
            assert "Your personal note curator agent" in captured.out

            # crystal is removed — falls back to minimal banner
            CONFIG.banner_style = "crystal"
            print_banner()
            captured = capsys.readouterr()
            assert "silica" in captured.out
            assert "Your personal note curator agent" in captured.out
    finally:
        CONFIG.banner_style = orig_style


