"""Tests for the ad-hoc /dedup and /refine commands and run_subagent_batch."""
import json
from unittest.mock import patch, MagicMock

from silica.agent.subagent import run_subagent_batch, LeashedSubAgent
from silica.planner.workqueue import WorkItem


# --- run_subagent_batch ----------------------------------------------------

def test_run_subagent_batch_aggregates_outcomes():
    items = [WorkItem(kind="dedup", target_path=f"N{i}.md") for i in range(4)]
    with patch.object(LeashedSubAgent, "handle", lambda self, it: {"status": "committed"}):
        res = run_subagent_batch(items, max_workers=2)
    assert res["items"] == 4
    assert res["summary"] == {"committed": 4}
    assert len(res["results"]) == 4


def test_run_subagent_batch_empty():
    assert run_subagent_batch([])["items"] == 0


# --- silica_dedup ----------------------------------------------------------

class _FakeStore:
    def __len__(self):
        return 2

    def paths(self):
        return ["Concepts/A", "Concepts/B"]

    def get_vec(self, p):
        return [1.0, 0.0]

    def cosine_top_k(self, vec, k=1, exclude=None):
        exclude = exclude or set()
        cand = next(x for x in ["Concepts/A", "Concepts/B"] if x not in exclude)
        return [{"path": cand, "score": 0.75, "name": cand}]


def _read_note(path):
    bodies = {"Concepts/A": "short", "Concepts/B": "a much longer note body " * 20}
    return MagicMock(content=bodies.get(path, ""))


def test_silica_dedup_builds_pair_targeting_larger_note():
    from silica.tools.composed import silica_dedup
    with patch("silica.kernel.embed.EmbedStore", _FakeStore), \
         patch("silica.driver.DRIVER.read_note", side_effect=_read_note), \
         patch("silica.agent.subagent.run_subagent_batch", return_value={"items": 1, "summary": {"committed": 1}, "results": []}) as batch:
        res = silica_dedup(folder="Concepts")

    items = batch.call_args.args[0]
    assert len(items) == 1
    # The larger note (B) is the merge target; the smaller (A) is the source.
    assert items[0].target_path == "Concepts/B"
    assert items[0].context["concept"] == "A"
    assert res["pairs_found"] == 1


def test_silica_dedup_requires_index():
    from silica.tools.composed import silica_dedup

    class _Empty(_FakeStore):
        def __len__(self):
            return 0

    with patch("silica.kernel.embed.EmbedStore", _Empty):
        res = silica_dedup(folder="X")
    assert "error" in res


# --- silica_refine ---------------------------------------------------------

def test_silica_refine_requires_folder():
    from silica.tools.composed import silica_refine
    res = silica_refine(folder="")
    assert "error" in res


def test_silica_refine_enqueues_one_item_per_note():
    from silica.tools.composed import silica_refine
    refs = [MagicMock(path="Notes/x.md"), MagicMock(path="Notes/y.md")]
    with patch("silica.driver.DRIVER.list_files", return_value=refs), \
         patch("silica.agent.subagent.run_subagent_batch", return_value={"items": 2, "summary": {"committed": 2}, "results": []}) as batch:
        res = silica_refine(folder="Notes")
    items = batch.call_args.args[0]
    assert len(items) == 2
    assert all(it.kind == "refine" for it in items)
    assert res["notes"] == 2


# --- CLI wiring ------------------------------------------------------------

def test_cli_dedup_shortcut_invokes_tool():
    from silica import cli
    fake_tool = MagicMock()
    fake_tool.run.return_value = json.dumps({"pairs_found": 3, "summary": {"committed": 2, "no_merge": 1}})
    with patch.dict("silica.tools.TOOLS", {"silica_dedup": fake_tool}, clear=False):
        handled = cli._handle_direct_shortcut("/dedup Concepts/ML", [])
    assert handled is True
    fake_tool.run.assert_called_once_with(folder="Concepts/ML")
