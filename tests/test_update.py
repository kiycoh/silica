# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""`silica update` against throwaway git repos.

Each test builds a bare remote + an "install" clone tracking origin/main, then
monkeypatches ``silica.update.ROOT`` at the install so update() operates on it.
The rollback test is the one that matters: a pulled syntax error must never
survive on disk.
"""

from __future__ import annotations

import subprocess

import silica.update as upd


def _run(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _make_install(tmp_path):
    """A clone whose origin/main tracks a shared bare remote."""
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "--bare", "-b", "main", str(remote)],
        check=True, capture_output=True,
    )
    seed = tmp_path / "seed"
    seed.mkdir()
    _run(seed, "init", "-b", "main")
    _run(seed, "config", "user.email", "t@t")
    _run(seed, "config", "user.name", "t")
    _run(seed, "remote", "add", "origin", str(remote))
    (seed / "silica").mkdir()
    (seed / "silica" / "__init__.py").write_text("x = 1\n")
    _run(seed, "add", "-A")
    _run(seed, "commit", "-m", "init")
    _run(seed, "push", "-u", "origin", "main")

    install = tmp_path / "install"
    _run(tmp_path, "clone", str(remote), str(install))
    _run(install, "config", "user.email", "t@t")
    _run(install, "config", "user.name", "t")
    return install


def _push(tmp_path, name, rel, content):
    """Land a new commit on the shared remote via a throwaway clone."""
    clone = tmp_path / name
    _run(tmp_path, "clone", str(tmp_path / "remote.git"), str(clone))
    _run(clone, "config", "user.email", "t@t")
    _run(clone, "config", "user.name", "t")
    p = clone / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    _run(clone, "add", "-A")
    _run(clone, "commit", "-m", "change")
    _run(clone, "push", "origin", "main")


def _head(cwd):
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True
    ).stdout.strip()


def test_already_up_to_date(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(upd, "ROOT", _make_install(tmp_path))
    assert upd.update() == 0
    assert "up to date" in capsys.readouterr().out.lower()


def test_dirty_tree_aborts(tmp_path, monkeypatch, capsys):
    install = _make_install(tmp_path)
    _push(tmp_path, "other", "silica/feature.py", "y = 2\n")  # an update exists
    (install / "silica" / "__init__.py").write_text("x = 999\n")  # local edit
    monkeypatch.setattr(upd, "ROOT", install)
    assert upd.update() == 1
    assert "uncommitted" in capsys.readouterr().out.lower()


def test_check_ignores_dirty_tree(tmp_path, monkeypatch, capsys):
    install = _make_install(tmp_path)
    _push(tmp_path, "other", "silica/feature.py", "y = 2\n")
    (install / "silica" / "__init__.py").write_text("x = 999\n")
    monkeypatch.setattr(upd, "ROOT", install)
    assert upd.update(check_only=True) == 0  # pure query, dirty tree irrelevant
    out = capsys.readouterr().out.lower()
    assert "available" in out and "uncommitted" not in out


def test_pulls_and_updates(tmp_path, monkeypatch, capsys):
    install = _make_install(tmp_path)
    _push(tmp_path, "other", "silica/feature.py", "y = 2\n")
    monkeypatch.setattr(upd, "ROOT", install)
    assert upd.update() == 0
    assert (install / "silica" / "feature.py").exists()
    assert "updated" in capsys.readouterr().out.lower()


def test_rolls_back_on_syntax_error(tmp_path, monkeypatch, capsys):
    install = _make_install(tmp_path)
    old = _head(install)
    _push(tmp_path, "other", "silica/broken.py", "def (\n")  # not valid Python
    monkeypatch.setattr(upd, "ROOT", install)
    assert upd.update() == 1
    assert _head(install) == old  # rolled back to pre-pull commit
    assert "syntax error" in capsys.readouterr().out.lower()
