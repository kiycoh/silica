# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""atomic_write_bytes: overwrite lands, a failed write leaves the old file intact."""
import os

import pytest

from silica.kernel.paths import atomic_write_bytes


def test_write_and_overwrite(tmp_path):
    p = tmp_path / "sub" / "index.json"
    atomic_write_bytes(p, b"v1")  # creates parent dirs
    atomic_write_bytes(p, b"v2")
    assert p.read_bytes() == b"v2"
    assert list(p.parent.iterdir()) == [p]  # no tmp leftovers


def test_failed_write_keeps_previous_content(tmp_path, monkeypatch):
    p = tmp_path / "index.json"
    atomic_write_bytes(p, b"good")

    def boom(fd):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError):
        atomic_write_bytes(p, b"torn")
    assert p.read_bytes() == b"good"
    assert list(p.parent.iterdir()) == [p]
