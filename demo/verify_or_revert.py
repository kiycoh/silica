# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia
"""Demo beat: verify-or-revert.

Record it:
    asciinema rec verify-or-revert.cast -c "uv run python demo/verify_or_revert.py"
    agg verify-or-revert.cast verify-or-revert.gif   # cast -> gif

The story, in one beat:
  1. A healthy write lands. LaTeX backslashes survive byte-for-byte.
  2. A backend that corrupts the payload (the real 2026-06-30 backslash-doubling
     bug, injected on purpose) is caught by the post-write read-back gate and the
     write is reverted. The vault is left clean, and Silica says so out loud.

Only the fault is simulated. The gate and the revert are the real code path
(silica.kernel.atomic_write.commit_note_atomic).
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

C = {"dim": "\033[2m", "red": "\033[31m", "grn": "\033[32m", "cyn": "\033[36m",
     "bold": "\033[1m", "off": "\033[0m"}


def cap(text: str, pause: float = 1.6) -> None:
    """One on-screen caption line, with a beat of silence for the GIF."""
    print(f"\n{C['cyn']}# {text}{C['off']}")
    time.sleep(pause)


def show_file(vault: Path, rel: str) -> None:
    p = vault / rel
    if p.exists():
        body = p.read_text()
        latex = next((l for l in body.splitlines() if "\\" in l), "(no backslash line)")
        print(f"{C['grn']}  on disk:{C['off']} {rel}   {C['dim']}{latex.strip()}{C['off']}")
    else:
        print(f"{C['red']}  on disk:{C['off']} {rel} {C['bold']}absent{C['off']} "
              f"{C['dim']}(nothing landed){C['off']}")


def main() -> None:
    import os
    os.environ["SILICA_BACKEND"] = "fs"
    vault = Path(tempfile.mkdtemp(prefix="silica-demo-"))

    from silica.driver import set_driver
    from silica.driver.fs_backend import ObsidianFSBackend
    from silica.driver import get_driver
    from silica.kernel.atomic_write import commit_note_atomic
    from silica.kernel.ops import Op, OpType

    set_driver(ObsidianFSBackend(vault_path=str(vault)))

    snippet = r"The Euler product: $\zeta(s)=\prod_p \frac{1}{1-p^{-s}}$."

    # ── Beat A: a healthy write lands, backslashes intact ────────────────
    cap("silica writes a note with a LaTeX formula")
    op = Op(op=OpType.write, heading="Euler product", source_basename="demo.md",
            path="Euler product.md", hub="Number theory", snippet=snippet)
    res = commit_note_atomic(op, lint=False)
    print(f"  commit -> ok={C['grn']}{res.ok}{C['off']}")
    show_file(vault, "Euler product.md")

    # ── Beat B: a corrupting backend is caught and reverted ──────────────
    cap("now a backend that doubles every backslash (the real 2026-06-30 bug)")
    drv = get_driver()
    healthy_create = drv.create
    drv.create = lambda path, content: healthy_create(path, content.replace("\\", "\\\\"))

    op2 = Op(op=OpType.write, heading="Riemann zeta", source_basename="demo.md",
             path="Riemann zeta.md", hub="Number theory", snippet=snippet)
    res2 = commit_note_atomic(op2, lint=False)
    print(f"  commit -> ok={C['red']}{res2.ok}{C['off']}  "
          f"reverted={C['grn']}{res2.reverted}{C['off']}")
    print(f"  {C['dim']}{res2.error}{C['off']}")
    show_file(vault, "Riemann zeta.md")

    cap("the corrupted note never reached your vault. verify, or revert.", pause=2.2)


if __name__ == "__main__":
    main()
