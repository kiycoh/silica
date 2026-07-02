"""Regression: cli backend writes must not backslash-double LaTeX.

Root cause (2026-06-30): `create`/`overwrite`/`append` had a CLI-arg write path
that escaped `\\`→`\\\\` and handed `content=<escaped>` to `obsidian create`. The
Obsidian CLI receiver only reverses `\\n`→newline, NOT `\\\\`→`\\`, so every LaTeX
command landed doubled on disk (`\\begin`, `\\sum`) and `\\n`-commands like
`\\nabla`/`\\neq` got split across a newline. Notes written via `create` (every new
spoke) were uniformly corrupted; patches survived because overwrite/append used the
correct `_js_str` + vault.process eval channel.

The fix routes all writes through a lossless channel (JS string-literal eval, which
decodes `_js_str` cleanly, or the verbatim temp-file write). These tests assert the
raw doubling CLI path is never taken.
"""
from silica.driver.cli_backend import ObsidianCLIBackend, _js_str

# A body exercising both corruption classes: a `\n`-command and a block env.
MATHY = r"$\nabla \cdot f$" + "\n\\begin{equation*}\nx = \\sum_i p_i\n\\end{equation*}"


def _detached_backend():
    """An instance without __init__ (no live Obsidian needed)."""
    return ObsidianCLIBackend.__new__(ObsidianCLIBackend)


def test_create_uses_lossless_eval_channel_not_doubled_cli(monkeypatch):
    be = _detached_backend()
    captured = {}
    monkeypatch.setattr(be, "_reject_hidden", lambda p: None)
    monkeypatch.setattr(be, "_eval", lambda js: captured.__setitem__("js", js) or "ok")
    monkeypatch.setattr(be, "_wait_for_content_reflects", lambda *a, **k: None)
    monkeypatch.setattr(be, "_wait_for_resolved_event", lambda *a, **k: None)
    monkeypatch.setattr(be, "_patch_graph_add", lambda *a, **k: None)

    def fail_raw(*a, **k):
        raise AssertionError(f"create used the raw doubling CLI channel: {a}")

    monkeypatch.setattr(be, "_run_cli", fail_raw)

    be.create("Note.md", MATHY)

    # The eval channel embeds the body via _js_str, which a JS string literal
    # decodes back to the exact original — no surviving backslash doubling.
    assert _js_str(MATHY) in captured["js"]
    # (overwrite/append's verbatim eval-failure fallback is covered by
    #  test_set_prop_atomic.test_overwrite_falls_back_to_verbatim_write_on_eval_failure)
