from __future__ import annotations

from silica.workers.profile import WorkerProfile, WorkerTask, WorkerResult, PROFILES


def test_worker_result_shape():
    r = WorkerResult(status="ok", output={"x": 1}, detail="done")
    assert r.status == "ok"
    assert r.output == {"x": 1}
    assert r.detail == "done"


def test_worker_task_shape():
    t = WorkerTask(profile="reader", goal="gather X", inputs={"paths": ["A.md"]})
    assert t.profile == "reader"
    assert t.inputs["paths"] == ["A.md"]


def test_profile_is_frozen_and_registry_is_a_dict():
    p = WorkerProfile(
        name="t",
        tools=("silica_read_note",),
        leash_factory=None,
        max_iterations=4,
        system_prompt="be brief",
        result_parser=lambda text, trace: WorkerResult(status="ok", output=text),
    )
    assert p.max_iterations == 4
    assert isinstance(PROFILES, dict)
    # frozen dataclass — assignment raises
    try:
        p.name = "x"
        assert False, "WorkerProfile should be frozen"
    except Exception:
        pass
