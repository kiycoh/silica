from unittest.mock import patch, MagicMock, PropertyMock
import pytest
from silica.config import CONFIG
from silica.cli import _handle_slash_command
from silica.ui.renderer import make_progress_callback, _redact
from silica.ui.console import CONSOLE
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


def test_callback_modes_output(capsys, monkeypatch):
    orig_mode = CONFIG.tool_progress
    # This test exercises the non-interactive (plain print) branch; pin the console
    # off-terminal so it doesn't take the Live branch when FORCE_COLOR/a TTY is present.
    monkeypatch.setattr(CONSOLE, "_force_terminal", False)
    cb = make_progress_callback()

    try:
        # Test "new" mode
        CONFIG.tool_progress = "new"
        cb(ToolStartEvent(name="silica_read_note", args={"name": "noteA"}, call_id="1", iteration=1))
        captured = capsys.readouterr()
        assert "read" in captured.out
        assert "noteA" in captured.out

        # Same tool consecutive call should be skipped in "new" mode
        cb(ToolStartEvent(name="silica_read_note", args={"name": "noteB"}, call_id="2", iteration=2))
        captured = capsys.readouterr()
        assert captured.out == ""

        # Different tool should print
        cb(ToolStartEvent(name="silica_search", args={"query": "searchQ"}, call_id="3", iteration=3))
        captured = capsys.readouterr()
        assert "search" in captured.out
        assert "searchQ" in captured.out

        # Test "all" mode
        CONFIG.tool_progress = "all"
        cb(ToolStartEvent(name="silica_read_note", args={"name": "noteA"}, call_id="4", iteration=4))
        captured = capsys.readouterr()
        assert "read" in captured.out
        assert "noteA" in captured.out

        # Test "verbose" mode
        CONFIG.tool_progress = "verbose"
        cb(ToolStartEvent(name="silica_read_note", args={"name": "noteA"}, call_id="5", iteration=5))
        captured = capsys.readouterr()
        assert "read" in captured.out
        assert "noteA" in captured.out

        cb(ToolCompleteEvent(name="silica_read_note", args={"name": "noteA"}, call_id="5", result="some result", duration_s=1.23, iteration=5))
        captured = capsys.readouterr()
        assert "read" in captured.out
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


def test_injector_block_height_constant_across_widths():
    """Regression: the injector live block (spinner header + indented stage track) must
    keep the same height on a narrow vs wide console. A wrapping track grew the block
    between frames and tore the Live region on a small / non-fullscreen terminal."""
    import io
    from rich.console import Console, Group
    from rich.padding import Padding
    from rich.text import Text
    from silica.ui.renderer import _stage_track
    phases = [
        {"phase": "payload",   "status": "done",    "elapsed": 0.8},
        {"phase": "salience",  "status": "done",    "elapsed": 0.3},
        {"phase": "collision", "status": "running", "elapsed": None},
    ]

    def block_height(width: int) -> int:
        buf = io.StringIO()
        c = Console(file=buf, width=width)
        header = Text(" injector · some/long/inbox/path/with a long file name.md",
                      no_wrap=True, overflow="ellipsis")
        c.print(Group(header, Padding(_stage_track(phases, width), (0, 0, 0, 2))))
        return len(buf.getvalue().rstrip("\n").split("\n"))

    heights = {w: block_height(w) for w in (30, 50, 80, 120)}
    assert len(set(heights.values())) == 1, f"block height varies with width: {heights}"


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


def test_batch_micro_phase_tracked_from_work_feedback(monkeypatch):
    import silica.agent.bus as bus_mod
    from silica.agent.events import WorkFeedbackEvent
    from silica.ui.renderer import make_progress_callback

    # Logic-only test: pin off-terminal so the batch panel (full of MagicMocks)
    # is never rendered via the Live branch when FORCE_COLOR/a TTY is present.
    monkeypatch.setattr(CONSOLE, "_force_terminal", False)
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


def test_batch_micro_phase_resets_on_ledger_next_complete(monkeypatch):
    import json
    import silica.agent.bus as bus_mod
    from silica.agent.events import WorkFeedbackEvent, ToolCompleteEvent
    from silica.ui.renderer import make_progress_callback

    # Logic-only test: pin off-terminal (see sibling) to avoid rendering MagicMocks.
    monkeypatch.setattr(CONSOLE, "_force_terminal", False)
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
                              args={"inbox_files": ["a.md", "b.md"]}, call_id="1", iteration=1))
        assert cb._inject_progress is not None
        assert cb._inject_progress.live.is_started is False
    finally:
        cb.close()
        CONFIG.tool_progress = orig_mode


def test_injector_single_file_has_no_bar():
    """A 0/1→1/1 bar is noise: the file bar only appears for multi-file runs."""
    from unittest.mock import patch, PropertyMock
    from rich.console import Console
    from silica.agent.events import ToolStartEvent
    from silica.ui.renderer import make_progress_callback

    orig_mode = CONFIG.tool_progress
    cb = make_progress_callback()
    try:
        CONFIG.tool_progress = "all"
        with patch.object(Console, "is_terminal", new_callable=PropertyMock, return_value=True), \
             patch.object(cb, "_update_live", lambda: None):
            cb(ToolStartEvent(name="silica_run_injector",
                              args={"inbox_files": ["a.md"]}, call_id="1", iteration=1))
        assert cb._inject_progress is None
    finally:
        cb.close()
        CONFIG.tool_progress = orig_mode


def test_phase_refires_do_not_touch_file_bar():
    """The bar tracks FILES processed, not phases — so phase re-fires (retries /
    deferred reprocessing) never move it. The phase track still dedups by label."""
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn,
    )
    from silica.ui.renderer import make_progress_callback

    cb = make_progress_callback()
    try:
        cb._inject_progress = Progress(
            SpinnerColumn(), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
            auto_refresh=False,
        )
        cb._inject_task_id = cb._inject_progress.add_task("", total=3)  # 3 files
        cb._pipeline_phases = []
        cb._phase_start_times = {}

        phases = ["payload", "salience", "collision"]
        for _ in range(3):  # three passes over the same phases
            for p in phases:
                cb._on_pipeline_phase(p, "running", None)
                cb._on_pipeline_phase(p, "done", 0.1)

        # Bar untouched by phase events; track deduped to 3 distinct phases.
        assert cb._inject_progress.tasks[cb._inject_task_id].completed == 0
        assert len(cb._pipeline_phases) == len(phases)
    finally:
        cb.close()


def test_count_files_done():
    from silica.router.orchestrator import _count_files_done
    # file 0 → chunks 0,1 ; file 1 → chunk 2 ; file 2 → chunk 3
    flat_map = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (2, 0)}
    assert _count_files_done(flat_map, upto_idx=0) == 0   # nothing past 0
    assert _count_files_done(flat_map, upto_idx=2) == 1   # file 0 fully behind
    assert _count_files_done(flat_map, upto_idx=4) == 3   # all done
    assert _count_files_done({}, upto_idx=5) == 0


def _capture_file_progress(fn):
    """Run fn() with the run-progress hook installed; return [(done, total), ...]."""
    from silica.ui import renderer
    seen: list[tuple[int, int]] = []
    renderer._set_run_progress_hook(lambda d, t, label="": seen.append((d, t)))
    try:
        fn()
    finally:
        renderer._set_run_progress_hook(None)
    return seen


def test_committed_file_counts_toward_bar_done():
    """Regression: an already-committed (dedup'd) file is in the denominator
    (len(inbox_files)) but is skipped before PAYLOAD, so it never enters the
    flat map. It must still count as done, or the bar stalls below 100%."""
    from silica.router.orchestrator import InjectorFSM

    with patch("silica.kernel.ledger.get_ledger"):
        fsm = InjectorFSM(inbox_files=["Inbox/a.md", "Inbox/b.md"], target_dir="Concepts")
    fsm._committed_file_indices = {0}          # file 0 already nucleated → skipped
    fsm._chunk_flat_to_fi_ci = {0: (1, 0)}     # only file 1 got payloaded
    fsm._chunks = [{}]

    seen = _capture_file_progress(lambda: fsm._emit_files_progress(len(fsm._chunks)))
    done, total = seen[-1]
    assert (done, total) == (2, 2), f"bar stalled at {done}/{total} — committed file not counted"


def test_file_advance_surfaces_finished_file():
    """Regression: finishing a file (advancing to the next) must emit progress,
    else a run of 1-chunk files sits at 0/N until the very last chunk."""
    from silica.router.orchestrator import InjectorFSM, InjectorState

    with patch("silica.kernel.ledger.get_ledger"):
        fsm = InjectorFSM(inbox_files=["Inbox/a.md", "Inbox/b.md"], target_dir="Concepts")
    fsm._chunk_flat_to_fi_ci = {0: (0, 0)}     # file 0 payloaded, one chunk
    fsm._chunks = [{}]
    fsm._current_chunk_idx = 0
    fsm.context["payload"] = {"chunks": fsm._chunks}

    with patch.object(fsm, "_advance_file_or_done", return_value=True):
        seen = _capture_file_progress(fsm._eval_loop_or_done)
    assert seen, "no progress emitted when a file finished and the FSM advanced"
    done, _total = seen[-1]
    assert done >= 1, f"finished file 0 not reflected (done={done})"


def test_injector_bar_total_is_file_count():
    from unittest.mock import patch, PropertyMock
    from rich.console import Console
    from silica.agent.events import ToolStartEvent
    from silica.ui.renderer import make_progress_callback

    orig_mode = CONFIG.tool_progress
    cb = make_progress_callback()
    try:
        CONFIG.tool_progress = "all"
        with patch.object(Console, "is_terminal", new_callable=PropertyMock, return_value=True), \
             patch.object(cb, "_update_live", lambda: None):
            cb(ToolStartEvent(name="silica_run_injector",
                              args={"inbox_files": ["a.md", "b.md", "c.md"]},
                              call_id="1", iteration=1))
        task = cb._inject_progress.tasks[cb._inject_task_id]
        assert task.total == 3  # files, not 16 phases
    finally:
        cb.close()
        CONFIG.tool_progress = orig_mode


def test_run_progress_advances_file_bar():
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn,
    )
    from silica.ui.renderer import make_progress_callback

    cb = make_progress_callback()
    try:
        cb._inject_progress = Progress(
            SpinnerColumn(), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
            auto_refresh=False,
        )
        cb._inject_task_id = cb._inject_progress.add_task("", total=3)
        cb._on_run_progress(2, 3)
        assert cb._inject_progress.tasks[cb._inject_task_id].completed == 2
        cb._on_run_progress(9, 3)  # clamp: never overflow
        assert cb._inject_progress.tasks[cb._inject_task_id].completed == 3
    finally:
        cb.close()


def test_run_progress_updates_inbox_label():
    """Regression: the panel title must follow the document currently processed,
    not stay frozen on the first file."""
    from silica.ui.renderer import make_progress_callback

    cb = make_progress_callback()
    try:
        cb._inject_inbox_label = "a.md"
        cb._on_run_progress(1, 2, label="b.md")
        assert cb._inject_inbox_label == "b.md"
        cb._on_run_progress(2, 2, label="")  # empty label must not clobber
        assert cb._inject_inbox_label == "b.md"
    finally:
        cb.close()


def test_injector_summary_shows_yield(capsys):
    import json
    from unittest.mock import patch, PropertyMock
    from rich.console import Console
    from silica.agent.events import ToolCompleteEvent
    from silica.ui.renderer import make_progress_callback

    orig_mode = CONFIG.tool_progress
    cb = make_progress_callback()
    try:
        CONFIG.tool_progress = "all"
        cb._injector_call_id = "1"
        cb._inject_file_count = 3
        cb._inject_inbox_label = "a.md"
        cb._pipeline_phases = [{"phase": "write", "status": "done", "elapsed": 0.1}]
        with patch.object(Console, "is_terminal", new_callable=PropertyMock, return_value=True):
            cb(ToolCompleteEvent(
                name="silica_run_injector", args={}, call_id="1",
                result=json.dumps({"yield_notes": 7, "yield_links": 12}),
                duration_s=4.2, iteration=1,
            ))
        out = capsys.readouterr().out
        assert "3 file" in out
        assert "7 note" in out
        assert "12 link" in out
    finally:
        cb.close()
        CONFIG.tool_progress = orig_mode


def test_print_banner_styles(capsys):
    from silica.ui.banner import print_banner

    orig_show = CONFIG.show_banner
    try:
        # Banner off → plain one-liner
        CONFIG.show_banner = False
        print_banner()
        captured = capsys.readouterr()
        assert "silica" in captured.out
        assert "Your personal note curator agent" in captured.out

        # Banner on → hand-drawn line-art wordmark (3 rows) + caption
        CONFIG.show_banner = True
        print_banner()
        captured = capsys.readouterr()
        assert "╭─╴" in captured.out  # bespoke rounded line-art, not a figlet font
        assert "Your personal note curator agent" in captured.out
        # wordmark (3 rows) + caption, plus the mascot above it — not the 1-line fallback.
        assert len([ln for ln in captured.out.splitlines() if ln.strip()]) >= 4
    finally:
        CONFIG.show_banner = orig_show


