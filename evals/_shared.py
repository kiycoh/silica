# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Provenance + lever-liveness helpers shared by the benchmark runners.

Provenance answers "which code, which data, which run produced this number":
git SHA (via silica.kernel.gitstate), dataset path + sha256, a timestamped run
id. Liveness answers "was the lever I switched on actually live", so an A/B
cannot silently compare baseline vs baseline (empty lexical index, dead
reranker) or trust an unpinned nondeterministic provider route.
"""
from __future__ import annotations

import datetime
import hashlib
import sys
from pathlib import Path


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_id() -> str:
    """A per-run timestamp id (second resolution) — unique enough to keep two
    reruns from clobbering, and to tie a metrics file to its provenance block."""
    return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")


def git_sha() -> str | None:
    """HEAD sha of the repo this code lives in, or None outside a git repo."""
    from silica.kernel import gitstate

    root = gitstate.find_repo_root(Path(__file__))
    return gitstate.head_ref(root) if root else None


def provenance(data_path: str | Path, *, rid: str | None = None) -> dict:
    """Attribution block: {run_id, git_sha, dataset:{path, sha256}}. The dataset
    sha256 disambiguates positional question ids (conv-26_q0 names different
    questions across data files — the hash says which file)."""
    p = Path(data_path)
    return {
        "run_id": rid or run_id(),
        "git_sha": git_sha(),
        "dataset": {"path": str(p), "sha256": _sha256_file(p)},
    }


def embedding_model(config, live: bool) -> str | None:
    return getattr(config, "embedding_model", None) if live else None


# --- Lever liveness (fail fast, never fake a null A/B) -----------------------

def assert_lexical_live() -> None:
    """--lexical over an empty index is a documented no-op; refuse it rather
    than report a null result that is an artifact. Assumes the vault is bound."""
    from silica.kernel.lexical import get_lexical_store

    if len(get_lexical_store()) == 0:
        raise SystemExit(
            "--lexical set but the lexical index is empty: build it first with "
            "silica_lexical_refresh (the /lexical CLI). Refusing to run a no-op arm.")


def assert_reranker_live(config) -> None:
    """A configured reranker whose server is down silently abstains and fakes
    rerank == embed-only. Probe it once; fail fast if it will not answer. No
    reranker configured is not a lie (config records reranker=None), so pass."""
    from silica.agent.providers import get_reranker

    rr = get_reranker(config)
    if rr is None:
        return
    if rr.scores("ping", ["pong"]) is None:
        raise SystemExit(
            "reranker configured but not responding (server down?): it would "
            "silently abstain and fake rerank == embed-only. Start the reranker "
            "or pass --no-rerank.")


def warn_unpinned_provider(model: str, provider_pin: str | None) -> None:
    """Unpinned openrouter routing is nondeterministic even at temperature=0
    (proven: a byte-identical prompt flipped verdicts). Warn, do not fail —
    local backends legitimately have no provider concept."""
    if provider_pin is None and str(model).startswith("openrouter/"):
        print("WARNING: openrouter model with no provider pin — unpinned routing "
              "is nondeterministic even at temperature=0. Set "
              "SILICA_OPENROUTER_PROVIDER for a comparable A/B.", file=sys.stderr)


if __name__ == "__main__":
    # ponytail: smallest check that the provenance block is well-formed and the
    # dataset hash actually reflects file content.
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "data.json"
        f.write_text(json.dumps({"x": 1}), encoding="utf-8")
        block = provenance(f, rid="RID")
        assert block["run_id"] == "RID"
        assert block["dataset"]["path"] == str(f)
        assert block["dataset"]["sha256"] == _sha256_file(f)
        f.write_text(json.dumps({"x": 2}), encoding="utf-8")
        assert provenance(f)["dataset"]["sha256"] != block["dataset"]["sha256"]
        assert provenance(Path(d) / "missing.json")["dataset"]["sha256"] is None
    print("ok")
