# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Corrupt state is quarantined (*.corrupt.<stamp>), never clobbered or dropped.

Derived (cooccurrence index) rebuilds from empty after quarantine;
authoritative (provenance ledger) preserves the corrupt bytes so a later
append_record cannot overwrite history.
"""
from __future__ import annotations

import json

from silica.kernel.cooccurrence import CooccurStore
from silica.kernel.paths import quarantine
from silica.kernel.provenance import append_record, read_records


def test_quarantine_never_clobbers(tmp_path):
    a = tmp_path / "state.json"
    a.write_text("junk1")
    dest1 = quarantine(a)
    a.write_text("junk2")
    dest2 = quarantine(a)  # same second: must not overwrite dest1
    assert dest1 != dest2
    assert dest1.read_text() == "junk1"
    assert dest2.read_text() == "junk2"
    assert not a.exists()


def test_quarantine_missing_file_returns_none(tmp_path):
    assert quarantine(tmp_path / "absent.json") is None


def test_corrupt_provenance_quarantined_and_append_does_not_clobber(tmp_path):
    store = tmp_path / "provenance.json"
    store.write_text("{not json", encoding="utf-8")

    assert read_records(vault_path=str(tmp_path)) == []
    kept = list(tmp_path.glob("provenance.json.corrupt.*"))
    assert len(kept) == 1
    assert kept[0].read_text(encoding="utf-8") == "{not json"

    assert append_record("src.md", "sha", "run-1", ["Note A"], vault_path=str(tmp_path))
    assert kept[0].read_text(encoding="utf-8") == "{not json"  # bytes preserved
    assert [r["run_id"] for r in read_records(vault_path=str(tmp_path))] == ["run-1"]


def test_corrupt_provenance_non_array_quarantined(tmp_path):
    (tmp_path / "provenance.json").write_text(json.dumps({"oops": 1}), encoding="utf-8")
    assert read_records(vault_path=str(tmp_path)) == []
    assert len(list(tmp_path.glob("provenance.json.corrupt.*"))) == 1


def test_corrupt_cooccurrence_quarantined_and_loads_empty(tmp_path):
    idx = tmp_path / "cooccurrence.json"
    idx.write_bytes(b"\x00garbage")
    store = CooccurStore(path=idx)
    assert store._notes == {}
    kept = list(tmp_path.glob("cooccurrence.json.corrupt.*"))
    assert len(kept) == 1
    assert kept[0].read_bytes() == b"\x00garbage"
