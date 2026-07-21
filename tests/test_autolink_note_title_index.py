"""Task 2.2: autolink_note reuses the batch-built title index.

fs_backend.autolink_note previously rebuilt its own title index via
build_title_index(self.list_files()) on EVERY call, ignoring the one LINKING
already builds once per chunk. These tests pin:
  1. A passed title_index drives the links (not a rebuild from the vault).
  2. Omitting title_index falls back to the old rebuild-from-vault behavior.
  3. LINKING's handle_autolink threads its title_index through to the call.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from silica.driver.fs_backend import ObsidianFSBackend


def _ref(name):
    return SimpleNamespace(name=name, path=f"{name}.md")


# ---------------------------------------------------------------------------
# fs_backend.autolink_note — passed index vs. fallback
# ---------------------------------------------------------------------------

def test_autolink_note_uses_passed_title_index_not_rebuilt(tmp_path):
    """A phantom title (absent from the vault) still gets linked when it is
    handed in via title_index — proving the passed index drives linking,
    not a rebuild from list_files()."""
    (tmp_path / "Note.md").write_text("Talks about Phantom things.", encoding="utf-8")
    backend = ObsidianFSBackend(str(tmp_path))

    added = backend.autolink_note("Note.md", title_index=["Phantom"])

    body = backend.read_note("Note.md").content
    assert "[[Phantom]]" in body
    assert added == ["Phantom"]


def test_autolink_note_fallback_rebuilds_from_vault_when_no_index_passed(tmp_path):
    """Omitting title_index preserves today's behavior: rebuild from the
    real vault (list_files()) and only link titles that actually exist."""
    (tmp_path / "Phantom.md").write_text("# Phantom\n", encoding="utf-8")
    (tmp_path / "Note.md").write_text("Talks about Phantom things.", encoding="utf-8")
    backend = ObsidianFSBackend(str(tmp_path))

    added = backend.autolink_note("Note.md")

    body = backend.read_note("Note.md").content
    assert "[[Phantom]]" in body
    assert added == ["Phantom"]


def test_autolink_note_fallback_does_not_link_nonexistent_title(tmp_path):
    """Without a passed index, a title with no matching note must NOT be
    linked — the rebuild-from-vault fallback is the authoritative set."""
    (tmp_path / "Note.md").write_text("Talks about Phantom things.", encoding="utf-8")
    backend = ObsidianFSBackend(str(tmp_path))

    added = backend.autolink_note("Note.md")

    body = backend.read_note("Note.md").content
    assert "[[Phantom]]" not in body
    assert added == []


# ---------------------------------------------------------------------------
# LINKING (handle_autolink) — threads its title_index through to the driver
# ---------------------------------------------------------------------------

class _FakeFSM:
    """Minimal stand-in for InjectorFSM covering only what handle_autolink touches."""

    def __init__(self, ops_path):
        self._current_chunk_idx = 0
        self.context: dict = {}
        self._chunk_ctx_dict = {"ops_path": ops_path}

    def _progress_note(self, *args, **kwargs):
        pass

    def _chunk_task_id(self, cap, idx=None):
        return f"{cap}-0"

    def _transition_success(self):
        pass

    @property
    def _chunk_ctx(self):
        return self._chunk_ctx_dict


def test_handle_autolink_passes_title_index_to_driver():
    from silica.kernel.ops import Op, OpType
    from silica.kernel.autolink import build_title_index
    from silica.router.states.linking import handle_autolink

    fsm = _FakeFSM(ops_path="unused.json")
    write_op = Op(
        op=OpType.write,
        heading="Note",
        source_basename="src.md",
        path="Note.md",
    )

    refs = [_ref("Note"), _ref("Other")]
    driver = MagicMock()
    driver.list_files.return_value = refs

    with patch("silica.router.orchestrator.load_ops", return_value=[write_op]), \
         patch("silica.router.orchestrator.DRIVER", driver):
        handle_autolink(fsm)

    driver.autolink_note.assert_called_once()
    _, kwargs = driver.autolink_note.call_args
    assert kwargs.get("title_index") == build_title_index(refs)
