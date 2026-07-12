"""capabilities/codewiki — contract tests, provider mocked (like enrich)."""
import pytest

from silica.kernel.codewiki import SubsystemDigest
from silica.capabilities.codewiki import (
    generate_overview, generate_subsystem_note, render_digest,
)


def _digest(**over):
    base = dict(
        key="kernel", path="silica/kernel", members=["silica/kernel/util.py"],
        struct_sig="deadbeefdeadbeef",
        public_symbols={"silica/kernel/util.py": [
            {"kind": "function", "name": "helper", "parent": "",
             "signature": "def helper(x: int) -> int", "doc": "Add one.",
             "doc_full": "Add one.\nLonger detail.", "decorators": ["lru_cache"]}]},
        module_docs={"silica/kernel/util.py": "Utility module."},
        module_comments={"silica/kernel/util.py": ["top note"]},
        external_deps=["orjson"],
        collaborators_out=[("router", 2, 3)],
        collaborators_in=[("core", 1, 1)],
        fan_in_hubs=[("silica/kernel/util.py", 4)],
        entry_points=[("silica/kernel/util.py", "__main__ guard")],
        flow_sketches=[["silica/cli.py", "silica/kernel/util.py"]],
        parse_errors=1,
    )
    base.update(over)
    return SubsystemDigest(**base)


class _FakeResp:
    def __init__(self, text):
        self.text = text


class _FakeProvider:
    def __init__(self, text):
        self._text = text
        self.messages = None

    def call_llm(self, messages, tools=None, response_schema=None, max_tokens=0):
        self.messages = messages
        return _FakeResp(self._text)


def test_render_digest_contains_facts_and_residue():
    d = _digest(collaborators_out=[(f"s{i}", 1, 0) for i in range(40)])
    text = render_digest(d)
    assert "def helper(x: int) -> int" in text
    assert "@lru_cache" in text
    assert "Longer detail." in text          # full docstring, not first line
    assert "Utility module." in text
    assert "orjson" in text
    assert "__main__ guard" in text
    assert "silica/cli.py -> silica/kernel/util.py" in text
    assert "and 10 more" in text             # 40 collaborators, cap 30, declared
    assert "1 file(s) not analyzable" in text


def test_generate_subsystem_note_grounds_prompt(monkeypatch):
    fake = _FakeProvider('{"content": "## kernel\\nProse [[router]]"}')
    monkeypatch.setattr("silica.agent.providers.get_provider",
                        lambda config, role: fake)
    d = _digest()
    text = render_digest(d)
    note = generate_subsystem_note(d, text, config=None)
    assert note.content.startswith("## kernel")
    user_msg = fake.messages[1]["content"]
    assert text in user_msg                  # the digest IS the grounding
    assert "kernel" in fake.messages[0]["content"] or "kernel" in user_msg


def test_generate_empty_output_maps_to_empty_content(monkeypatch):
    fake = _FakeProvider("not json at all")
    monkeypatch.setattr("silica.agent.providers.get_provider",
                        lambda config, role: fake)
    note = generate_subsystem_note(_digest(), "digest text", config=None)
    assert note.content == ""


def test_generate_overview_includes_project_info(monkeypatch):
    fake = _FakeProvider('{"content": "# Architecture\\n[[kernel]]"}')
    monkeypatch.setattr("silica.agent.providers.get_provider",
                        lambda config, role: fake)
    note = generate_overview(
        summaries=[("kernel", "does kernel things")],
        edges=[("core", "kernel", 1, 1)],
        flows=[["silica/cli.py", "silica/kernel/util.py"]],
        project_info="name: silica\nscripts: silica = silica.cli:main",
        config=None,
    )
    assert note.content.startswith("# Architecture")
    user_msg = fake.messages[1]["content"]
    assert "name: silica" in user_msg
    assert "does kernel things" in user_msg


# ---------------------------------------------------------------------------
# Task 9: run_wiki pipeline + idempotency gate
# ---------------------------------------------------------------------------

import subprocess

from silica.capabilities.codewiki import run_wiki


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _mkrepo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "pkg").mkdir()
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "core.py").write_text(
        '"""Core module."""\nfrom pkg.util import helper\n\n\n'
        "def main():\n    helper()\n\n\n"
        'if __name__ == "__main__":\n    main()\n', encoding="utf-8")
    (root / "pkg" / "util.py").write_text(
        '"""Util module."""\n\n\ndef helper():\n    pass\n', encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init")
    vault = root / ".silica"
    vault.mkdir()
    return root, vault


@pytest.fixture()
def wiki_env(tmp_path, monkeypatch):
    root, vault = _mkrepo(tmp_path)
    fake = _FakeProvider('{"content": "Behavioral prose about the subsystem."}')
    monkeypatch.setattr("silica.agent.providers.get_provider",
                        lambda config, role: fake)
    # keep the derived index inside the tmp vault
    from silica.kernel import paths as kpaths
    monkeypatch.setattr(kpaths, "index_dir", lambda: vault / ".index")
    (vault / ".index").mkdir(parents=True, exist_ok=True)
    # bind the write channel (DRIVER) at the tmp vault (canonical tmp_vault setup)
    import silica.config
    import silica.driver
    monkeypatch.setattr(silica.config.CONFIG, "backend", "fs")
    monkeypatch.setattr(silica.config.CONFIG, "vault_path", str(vault))
    silica.driver._driver = None
    yield root, vault, fake
    silica.driver._driver = None


def test_first_run_builds_everything(wiki_env):
    root, vault, fake = wiki_env
    result = run_wiki(vault, config=None)
    assert result["status"] == "ok"
    arch = vault / "ARCHITECTURE.md"
    note = vault / "subsystems" / "core.md"
    assert arch.is_file() and note.is_file()
    text = note.read_text(encoding="utf-8")
    assert "wiki_struct_sig:" in text and "code_ref:" in text and "documents:" in text
    assert "```mermaid" in arch.read_text(encoding="utf-8")


def test_second_run_on_still_repo_skips_llm(wiki_env):
    root, vault, fake = wiki_env
    run_wiki(vault, config=None)
    fake.messages = None
    result = run_wiki(vault, config=None)
    assert result["written"] == []
    assert fake.messages is None            # no LLM call on a still repo


def test_body_only_call_change_triggers_regen(wiki_env):
    root, vault, fake = wiki_env
    run_wiki(vault, config=None)
    # body-only edit: remove the imported call (import stays, call goes:
    # import set and signatures identical)
    (root / "pkg" / "core.py").write_text(
        '"""Core module."""\nfrom pkg.util import helper\n\n\n'
        "def main():\n    pass\n\n\n"
        'if __name__ == "__main__":\n    main()\n', encoding="utf-8")
    result = run_wiki(vault, config=None)
    assert any(p.endswith("core.md") for p in result["written"])


def test_no_repo_degrades_soft(tmp_path, monkeypatch):
    from silica.kernel import paths as kpaths
    monkeypatch.setattr(kpaths, "repo_root_for", lambda v: None)
    assert run_wiki(tmp_path, config=None)["status"] == "no_repo"


# ---------------------------------------------------------------------------
# Task 10: conventions.wiki_dir
# ---------------------------------------------------------------------------

def test_conventions_wiki_dir_parsed(tmp_path):
    from silica.kernel.vault_manifest import _parse_conventions
    assert _parse_conventions({"conventions": {"wiki_dir": "docs/wiki"}}).wiki_dir == "docs/wiki"
    assert _parse_conventions({}).wiki_dir == ""
