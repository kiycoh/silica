# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""probe_integrity — differential corruption on write-path transforms.

An absolute lint of the human vault measures the human, not the pipeline (so
those totals are informational). The GATED form is differential: run each real
note body through the pipeline's body-transforming functions and require
``violations(after) ⊆ violations(before)`` — zero NEW violations. Gated at
exactly 1.0; a single introduced violation fails.

Transforms under test:
  T1  frontmatter split → dump round-trip (contract = violation-set inclusion,
      NOT byte equality — dump re-dumps yaml + lstrips body).
  T2  autolink insertion (exercises _build_skip_mask on real bodies).
  T3  fs backend write → read (the channel of the historical double-escaping
      LaTeX bug) — a scratch tempdir, never the real vault.
  T4  sanitize.normalize_ops (distiller post-processing).
"""
from __future__ import annotations

import tempfile

from silica.driver.fs_backend import ObsidianFSBackend
from silica.kernel import frontmatter
from silica.kernel.autolink import autolink, build_title_index
from silica.kernel.sanitize import normalize_ops
from tests.eval.golden import lint


def _t1_frontmatter(text, data, body, stem) -> dict:
    if not isinstance(data, dict):
        return {}  # no frontmatter / YAML error — nothing to round-trip
    return lint.new_violations(text, frontmatter.dump(data, body), stem)


def _t2_autolink(body, title_index, stem) -> dict:
    low = body.casefold()
    cands = [t for t in title_index if t.casefold() in low]
    new_body, _added = autolink(body, title_index, candidates=cands, self_title=stem)
    return lint.new_violations(body, new_body, stem)


def _t3_fs_roundtrip(backend, rel, text, stem) -> dict:
    ref = backend.create(rel, text)
    roundtrip = backend.read_note(ref).content
    return lint.new_violations(text, roundtrip, stem)


def _t4_sanitize(body, stem) -> dict:
    res = normalize_ops([{"content": body}])
    after = res[0].get("content", body) if res else body
    return lint.new_violations(body, after, stem)


def run(vault, *, verbose: bool = False) -> dict:
    from tests.eval.golden.runner import iter_notes

    all_md = iter_notes(vault)
    title_index = build_title_index([p.stem for p in all_md])

    notes = 0
    clean = 0
    vault_structural = 0
    vault_style = 0
    notes_with_structural = 0

    with tempfile.TemporaryDirectory() as scratch:
        backend = ObsidianFSBackend(vault_path=scratch)
        for p in all_md:
            text = p.read_text(encoding="utf-8")
            stem = p.stem
            rel = p.relative_to(vault).as_posix()
            data, _raw, body = frontmatter.split(text)

            # absolute lint (informational)
            structural, style = lint.totals(lint.scan(text, stem))
            vault_structural += structural
            vault_style += style
            if structural:
                notes_with_structural += 1

            # differential across the 4 transforms
            introduced = {
                "T1-frontmatter": _t1_frontmatter(text, data, body, stem),
                "T2-autolink": _t2_autolink(body, title_index, stem),
                "T3-fs-roundtrip": _t3_fs_roundtrip(backend, rel, text, stem),
                "T4-sanitize": _t4_sanitize(body, stem),
            }
            notes += 1
            if any(introduced.values()):
                if verbose:
                    for transform, viols in introduced.items():
                        for name, cnt in viols.items():
                            print(f"  NEW {rel} [{transform}] {name} +{cnt}")
            else:
                clean += 1

    rate = round(clean / notes, 4) if notes else 1.0
    if verbose:
        print(f"\nintegrity: rate {clean}/{notes} = {rate:.3f} | "
              f"vault structural={vault_structural} style={vault_style} "
              f"notes_with_structural={notes_with_structural}")

    return {
        "rate": rate,
        "notes": notes,
        "vault_structural_violations": vault_structural,
        "vault_style_flags": vault_style,
        "vault_notes_with_structural": notes_with_structural,
    }
