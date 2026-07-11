# tests/test_gitstate_codegraph.py
"""gitstate helpers backing the codegraph index (spec-code-lane §1)."""
import subprocess
from pathlib import Path

from silica.kernel import gitstate


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)


def _commit(path: Path, rel: str, text: str, msg: str) -> str:
    f = path / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text, encoding="utf-8")
    subprocess.run(["git", "add", "--", rel], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", msg, "--", rel], cwd=path, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True
    ).stdout.strip()


def test_list_files_tracked_untracked_not_ignored(tmp_path):
    _init_repo(tmp_path)
    _commit(tmp_path, "a.py", "x = 1\n", "c1")
    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")          # untracked
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("z = 3\n", encoding="utf-8")     # ignored
    files = gitstate.list_files(tmp_path)
    assert files is not None
    assert "a.py" in files and "b.py" in files
    assert "ignored.py" not in files


def test_list_files_none_outside_repo(tmp_path):
    assert gitstate.list_files(tmp_path) is None


def test_show_file_returns_old_content(tmp_path):
    _init_repo(tmp_path)
    ref0 = _commit(tmp_path, "m.py", "v1\n", "c1")
    _commit(tmp_path, "m.py", "v2\n", "c2")
    assert gitstate.show_file(tmp_path, ref0, "m.py") == "v1\n"
    assert gitstate.show_file(tmp_path, "0" * 40, "m.py") is None   # unknown ref
    assert gitstate.show_file(tmp_path, ref0, "nope.py") is None    # unknown path
    assert gitstate.show_file(tmp_path, "", "m.py") is None         # empty ref


def test_changed_paths_worktree_vs_head(tmp_path):
    _init_repo(tmp_path)
    _commit(tmp_path, "m.py", "v1\n", "c1")
    (tmp_path / "m.py").write_text("v2\n", encoding="utf-8")  # uncommitted edit
    assert gitstate.changed_paths(tmp_path) == ["m.py"]


def test_changed_paths_range(tmp_path):
    _init_repo(tmp_path)
    ref0 = _commit(tmp_path, "m.py", "v1\n", "c1")
    ref1 = _commit(tmp_path, "n.py", "w = 1\n", "c2")
    assert gitstate.changed_paths(tmp_path, f"{ref0}..{ref1}") == ["n.py"]
    assert gitstate.changed_paths(tmp_path) == []  # clean worktree
