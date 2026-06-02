"""Tests for the WarningLedger (silica/planner/warnings.py)."""
from silica.planner.warnings import WarningLedger


def test_add_and_dedup_by_path_kind():
    wl = WarningLedger()
    wl.add("A.md", "orphan", "first")
    wl.add("A.md", "orphan", "second")  # same (path, kind) → replaces, no dup
    wl.add("B.md", "orphan", "x")
    assert sorted(wl.paths("orphan")) == ["A.md", "B.md"]
    assert len(wl) == 2


def test_paths_filter_by_kind():
    wl = WarningLedger()
    wl.add("A.md", "orphan")
    wl.add("B.md", "other")
    assert wl.paths("orphan") == ["A.md"]
    assert set(wl.paths()) == {"A.md", "B.md"}


def test_empty_path_ignored():
    wl = WarningLedger()
    wl.add("", "orphan")
    assert len(wl) == 0


def test_persistence(tmp_path):
    wl = WarningLedger(run_dir=tmp_path)
    wl.add("A.md", "orphan", "detail")
    assert (tmp_path / "warnings.json").exists()


def test_persist_is_atomic_under_concurrent_add(tmp_path):
    """Concurrent add() calls must not lose entries from warnings.json."""
    import json
    import threading
    ledger = WarningLedger(run_dir=tmp_path)
    barrier = threading.Barrier(2)

    def add_many(paths):
        barrier.wait()
        for p in paths:
            ledger.add(p, kind="orphan")

    t1 = threading.Thread(target=add_many, args=(["a.md", "b.md", "c.md"],))
    t2 = threading.Thread(target=add_many, args=(["d.md", "e.md", "f.md"],))
    t1.start(); t2.start()
    t1.join(); t2.join()

    saved = json.loads((tmp_path / "warnings.json").read_bytes())
    assert len(saved) == 6, f"Expected 6 warnings, got {len(saved)}"
