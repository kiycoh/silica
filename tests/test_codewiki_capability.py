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
