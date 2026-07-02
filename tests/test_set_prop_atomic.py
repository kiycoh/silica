"""set_prop delegates to processFrontMatter — no post-write poll."""
from unittest.mock import patch
from silica.driver.cli_backend import ObsidianCLIBackend
from silica.driver.base import NoteRef


def test_set_prop_uses_process_frontmatter_and_no_wait():
    backend = ObsidianCLIBackend(vault_name="t")
    seen = {}

    def fake_run_cli(*args, **kwargs):
        seen["args"] = args
        return '=> "ok"'  # valid JSON string so _eval succeeds

    with patch.object(backend, "_run_cli", side_effect=fake_run_cli), \
         patch.object(backend, "_wait_for_prop") as wait_mock:
        backend.set_prop(NoteRef(name="N", path="N.md"), "status", "done")

    assert seen["args"][0] == "eval"
    assert "processFrontMatter" in seen["args"][1]
    assert "status" in seen["args"][1]
    wait_mock.assert_not_called()


def test_set_prop_escapes_path_and_name():
    backend = ObsidianCLIBackend(vault_name="t")
    captured = {}
    with patch.object(backend, "_run_cli", side_effect=lambda *a, **k: captured.update(code=a[1]) or '=> "ok"'):
        backend.set_prop(NoteRef(name="N", path="a'b.md"), "my'key", "x'y")
    # Path and property name go through _js_str; value goes through json.dumps.
    assert r"a\'b.md" in captured["code"]
    assert r"my\'key" in captured["code"]


def test_set_prop_falls_back_to_property_set_on_eval_failure():
    """When the eval path raises (any exception), property:set + _wait_for_prop run."""
    backend = ObsidianCLIBackend(vault_name="t")
    cli_calls = []

    def fake_run_cli(*args, **kwargs):
        if args[0] == "eval":
            raise RuntimeError("obsidian down")
        cli_calls.append(args)
        return ""

    with patch.object(backend, "_run_cli", side_effect=fake_run_cli), \
         patch.object(backend, "_wait_for_prop") as wait_mock:
        backend.set_prop(NoteRef(name="N", path="N.md"), "status", "done", type_="text")

    assert any(c[0] == "property:set" for c in cli_calls)
    wait_mock.assert_called_once()


def test_set_prop_falls_back_on_malformed_eval_output():
    """A json.JSONDecodeError (ValueError) from _eval must also trigger the fallback."""
    backend = ObsidianCLIBackend(vault_name="t")
    cli_calls = []

    def fake_run_cli(*args, **kwargs):
        if args[0] == "eval":
            return "=> not-valid-json"  # _eval will raise JSONDecodeError
        cli_calls.append(args)
        return ""

    with patch.object(backend, "_run_cli", side_effect=fake_run_cli), \
         patch.object(backend, "_wait_for_prop") as wait_mock:
        backend.set_prop(NoteRef(name="N", path="N.md"), "status", "done")

    assert any(c[0] == "property:set" for c in cli_calls)
    wait_mock.assert_called_once()


def test_overwrite_uses_vault_process_no_content_poll():
    backend = ObsidianCLIBackend(vault_name="t")
    seen = {}
    with patch.object(backend, "_run_cli", side_effect=lambda *a, **k: seen.update(args=a) or '=> "ok"'), \
         patch.object(backend, "_wait_for_content_reflects") as wait_mock, \
         patch.object(backend, "_patch_graph_add"):
        ref = backend.overwrite("Note.md", "new body")
    assert seen["args"][0] == "eval"
    assert "vault.process" in seen["args"][1]
    assert ref.name == "Note"
    wait_mock.assert_not_called()


def test_append_uses_vault_process_no_contains_poll():
    backend = ObsidianCLIBackend(vault_name="t")
    seen = {}
    with patch.object(backend, "_run_cli", side_effect=lambda *a, **k: seen.update(args=a) or '=> "ok"'), \
         patch.object(backend, "_wait_for_content_contains") as wait_mock:
        backend.append(NoteRef(name="Note", path="Note.md"), "tail text")
    assert seen["args"][0] == "eval"
    assert "vault.process" in seen["args"][1]
    wait_mock.assert_not_called()


def test_overwrite_falls_back_to_verbatim_write_on_eval_failure():
    """On eval failure overwrite must fall back to the verbatim temp-file write,
    never the `obsidian create content=` CLI whose receiver doubled every `\\`."""
    backend = ObsidianCLIBackend(vault_name="t")
    body = r"$\nabla f$" + "\n\\begin{equation*}\ny = \\sum_i x_i\n\\end{equation*}"
    cli = []
    def fake(*a, **k):
        if a[0] == "eval":
            raise RuntimeError("down")
        cli.append(a)
        return ""
    seen = {}
    with patch.object(backend, "_run_cli", side_effect=fake), \
         patch.object(backend, "_write_large_content",
                      side_effect=lambda p, c, append_mode=False: seen.update(content=c)), \
         patch.object(backend, "_wait_for_content_reflects") as wait_mock, \
         patch.object(backend, "_patch_graph_add") as patch_mock:
        backend.overwrite("Note.md", body)
    assert seen["content"] == body                 # verbatim — single backslashes survive
    assert not any(c[0] == "create" for c in cli)  # never the corrupting CLI channel
    wait_mock.assert_called_once()
    patch_mock.assert_called_once()  # graph patched even on the fallback path
