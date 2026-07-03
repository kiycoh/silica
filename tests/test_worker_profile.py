from __future__ import annotations

from silica.capabilities.profile import WorkerProfile, WorkerResult


def test_worker_result_shape():
    r = WorkerResult(status="ok", output={"x": 1}, detail="done")
    assert r.status == "ok"
    assert r.output == {"x": 1}
    assert r.detail == "done"


def test_profile_is_frozen():
    p = WorkerProfile(
        name="t",
        tools=("silica_read_note",),
        bounds_factory=None,
        max_iterations=4,
        system_prompt="be brief",
        result_parser=lambda text, trace: WorkerResult(status="ok", output=text),
    )
    assert p.max_iterations == 4
    # frozen dataclass — assignment raises
    try:
        p.name = "x"
        assert False, "WorkerProfile should be frozen"
    except Exception:
        pass
