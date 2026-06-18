"""Run facade — unified TaskLedger + ProgressLedger + RunManifest lifecycle.

The Run facade is the deepening of the Progress/Task ledger pair flagged by
the CoALA audit: Run.new / Run.resume are the only two ways a run comes into
existence; the resume fallback dance lives behind the interface.
"""
from __future__ import annotations

import pytest

import silica.kernel.progress as prog_mod
from silica.kernel.progress import PlanStep, Run, RunManifestEntry


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """Redirect the runs root to an isolated tmp directory."""
    d = tmp_path / "runs"
    monkeypatch.setattr(prog_mod, "_RUNS_DIR", d)
    return d


def _steps() -> list[PlanStep]:
    return [
        PlanStep(id="recon", kind="mechanical", objective="silica_recon"),
        PlanStep(id="distill", kind="semantic", objective="distiller"),
    ]


def _new_run(**overrides) -> Run:
    kwargs = dict(
        mode="inject",
        user_request="inject Inbox/a.md → Concepts",
        checkpoints=_steps(),
        inputs={"inbox_files": ["Inbox/a.md"]},
    )
    kwargs.update(overrides)
    return Run.new(**kwargs)


# ---------------------------------------------------------------------------
# Run.new
# ---------------------------------------------------------------------------

class TestRunNew:
    def test_creates_and_persists_both_ledgers(self, runs_dir):
        run = _new_run()
        assert run.resumed is False
        assert (runs_dir / run.run_id / "task_ledger.json").exists()
        assert (runs_dir / run.run_id / "ledger.json").exists()

    def test_trio_shares_run_id(self, runs_dir):
        run = _new_run()
        assert run.task_ledger.run_id == run.run_id
        assert run.progress.run_id == run.run_id
        assert run.manifest.run_id == run.run_id

    def test_fields_populated(self, runs_dir):
        run = _new_run()
        assert run.progress.mode == "inject"
        assert run.progress.inputs == {"inbox_files": ["Inbox/a.md"]}
        assert run.task_ledger.user_request == "inject Inbox/a.md → Concepts"
        assert [s.id for s in run.task_ledger.checkpoints] == ["recon", "distill"]
        assert run.manifest.entries == []

    def test_facts_falsy_normalised_to_dict(self, runs_dir):
        # Regression: cli.py used to pass facts=[] into a dict-typed field.
        run = _new_run(facts=[])
        assert run.task_ledger.facts == {}

    def test_run_dir_and_payloads_dir(self, runs_dir):
        run = _new_run()
        assert run.run_dir == runs_dir / run.run_id
        p = run.payloads_dir
        assert p == runs_dir / run.run_id / "payloads"
        assert p.is_dir()

    def test_save_persists_progress_mutations(self, runs_dir):
        run = _new_run()
        run.progress.add_task("recon", task_id="recon")
        run.save()
        from silica.kernel.progress import ProgressLedger
        reloaded = ProgressLedger.load(run.run_id)
        assert [t.id for t in reloaded.tasks] == ["recon"]


# ---------------------------------------------------------------------------
# Run.resume
# ---------------------------------------------------------------------------

class TestRunResume:
    def _resume(self, run_id: str, **overrides) -> Run:
        kwargs = dict(
            mode="inject",
            user_request="resume-args request",
            checkpoints=_steps(),
            inputs={"inbox_files": ["Inbox/b.md"]},
        )
        kwargs.update(overrides)
        return Run.resume(run_id, **kwargs)

    def test_resume_loads_existing_run(self, runs_dir):
        prior = _new_run()
        prior.progress.add_task("recon", task_id="recon")
        prior.save()

        run = self._resume(prior.run_id)
        assert run.resumed is True
        assert run.run_id == prior.run_id
        assert [t.id for t in run.progress.tasks] == ["recon"]
        # Loaded values win over resume-call args
        assert run.progress.inputs == {"inbox_files": ["Inbox/a.md"]}
        assert run.task_ledger.user_request == "inject Inbox/a.md → Concepts"

    def test_resume_missing_run_falls_back_fresh(self, runs_dir):
        run = self._resume("nonexistent_run_xyz")
        assert run.resumed is False
        assert run.run_id != "nonexistent_run_xyz"
        # The fresh fallback uses the caller's args
        assert run.task_ledger.user_request == "resume-args request"

    def test_resume_rebuilds_missing_task_ledger(self, runs_dir):
        prior = _new_run()
        (runs_dir / prior.run_id / "task_ledger.json").unlink()

        run = self._resume(prior.run_id)
        assert run.resumed is True
        assert run.task_ledger.user_request == "resume-args request"
        # Rebuilt ledger is persisted again (write-once semantics restart)
        assert (runs_dir / prior.run_id / "task_ledger.json").exists()

    def test_resume_restores_manifest(self, runs_dir):
        prior = _new_run()
        prior.manifest.record(RunManifestEntry(
            title="Alpha", path="Concepts/Alpha", parent=None,
            cluster_id=0, source_basename="a.md", op="write",
        ))
        prior.manifest.save()

        run = self._resume(prior.run_id)
        assert run.manifest.titles() == ["Alpha"]

    def test_resume_without_manifest_starts_empty(self, runs_dir):
        prior = _new_run()  # Run.new never writes manifest.json
        run = self._resume(prior.run_id)
        assert run.manifest.entries == []
        assert run.manifest.run_id == prior.run_id


# ---------------------------------------------------------------------------
# latest_run_id
# ---------------------------------------------------------------------------

class TestLatestRunId:
    def test_returns_most_recent_run(self, runs_dir):
        import os
        a = _new_run()
        b = _new_run()
        # Make mtimes unambiguous regardless of filesystem resolution
        os.utime(runs_dir / a.run_id, (1_000_000, 1_000_000))
        os.utime(runs_dir / b.run_id, (2_000_000, 2_000_000))
        assert prog_mod.latest_run_id() == b.run_id

    def test_none_when_no_runs(self, runs_dir):
        assert prog_mod.latest_run_id() is None

    def test_none_when_dir_missing(self, runs_dir):
        # The fixture only monkeypatches _RUNS_DIR; the directory is created
        # lazily by save(), so with no run created it must not exist.
        assert not runs_dir.exists()
        assert prog_mod.latest_run_id() is None

    def test_ignores_dirs_without_ledger(self, runs_dir):
        a = _new_run()
        (runs_dir / "junk_dir").mkdir(parents=True)
        assert prog_mod.latest_run_id() == a.run_id
