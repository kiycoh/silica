# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Offline unit tests for the golden harness probes (bare pytest, no network).

Each test is the smallest check that fails if a probe's masking / metric glue
breaks — not a measure of pipeline quality on the 14-note synthetic vault.
"""
from __future__ import annotations

import pytest

from silica.kernel import frontmatter
from silica.kernel.cooccurrence import CooccurStore, build_contribution
from silica.kernel.health import integrity_probe, lint
from evals.golden import probe_classify, probe_links


def test_probe_links_masked_recall(synthetic_vault):
    """14-note SPEC: [[MissingNote]] is dead, A/Cell + B/Cell are ambiguous
    basenames (excluded from the denominator); the remaining 10 links are
    trivially recoverable, so recall is exactly 1.0."""
    m = probe_links.run(synthetic_vault)
    assert m["links_evaluated"] == 10
    assert m["recall"] == 1.0


def test_probe_classify_taxonomy_and_counting(synthetic_vault, tmp_path):
    # (a) two fake domains sharing the stem "cell" — it appears in BOTH, so the
    #     everywhere-stem is dropped; each domain keeps its distinctive stems.
    store = CooccurStore(path=tmp_path / "co.json", lang="english")
    store.upsert_note("Bio/CellNote",
                      build_contribution("CellNote", "cell mitosis membrane organelle cell", lang="english"))
    store.upsert_note("Net/Router",
                      build_contribution("Router", "cell network router packet cell", lang="english"))

    tax = probe_classify.derive_taxonomy(["Bio", "Net"], store)
    bio = next(r for r in tax.rules if r.folder == "Bio").themes
    net = next(r for r in tax.rules if r.folder == "Net").themes

    bio_stems = set(store.note_nodes("Bio/CellNote"))
    net_stems = set(store.note_nodes("Net/Router"))
    shared = bio_stems & net_stems                    # the "cell" stem
    assert shared, "fixture must share a stem across domains"
    assert shared.isdisjoint(bio) and shared.isdisjoint(net)   # everywhere-stem excluded
    assert (bio_stems - net_stems) & set(bio)         # a Bio-only stem survives
    assert (net_stems - bio_stems) & set(net)

    # (b) counting glue on the synthetic vault: the probe's numbers must equal an
    #     independent recompute with identical args.
    store2 = CooccurStore(path=tmp_path / "syn.json", lang="english")
    for p in synthetic_vault.rglob("*.md"):
        _d, _r, body = frontmatter.split(p.read_text(encoding="utf-8"))
        rel = p.relative_to(synthetic_vault).with_suffix("").as_posix()
        store2.upsert_note(rel, build_contribution(p.stem, body, lang="english"))

    m = probe_classify.run(synthetic_vault, store2)

    from silica.kernel.classify import classify_notes
    domains = probe_classify.vault_domains(synthetic_vault)
    tax2 = probe_classify.derive_taxonomy(domains, store2)
    paths = probe_classify.domain_paths(synthetic_vault, domains)
    res = classify_notes(paths, tax2, cooccur_store=store2, llm_arbiter=False,
                         props_map={p: {} for p in paths})
    agree = sum(1 for c in res if c.target_folder == c.note_path.split("/")[0]) / len(res)
    assert m["notes"] == len(res)
    assert m["agreement"] == pytest.approx(round(agree, 4))


def test_probe_dedup_skips_without_index(tmp_path):
    """No embed index ⇒ empty result, never a crash — the runner SKIPs it."""
    from silica.kernel.cooccurrence import CooccurStore
    from evals.golden import probe_dedup

    out = probe_dedup.run(tmp_path, CooccurStore(path=tmp_path / "c.json"), embed_store=None)
    assert out["fp_pairs_evaluated"] == 0
    assert out["fp_auto_merge_rate"] == 0.0


def test_probe_dedup_fp_counts_mechanical_merge(tmp_path):
    """Two same-domain notes whose names fold to one key (Foo ⇄ Foo 1) at cos≈1
    are auto-routed `patch` — the FP arm must count that as an auto-merge."""
    from silica.kernel.cooccurrence import CooccurStore
    from silica.kernel.embed import EmbedStore
    from evals.golden import probe_dedup

    (tmp_path / "D").mkdir()
    (tmp_path / "D" / "Foo.md").write_text("# Foo\nbody one", encoding="utf-8")
    (tmp_path / "D" / "Foo 1.md").write_text("# Foo 1\nbody two", encoding="utf-8")
    es = EmbedStore(path=tmp_path / "e.json")
    es.upsert("D/Foo", "Foo", [1.0, 0.0, 0.0])
    es.upsert("D/Foo 1", "Foo 1", [1.0, 0.002, 0.0])

    out = probe_dedup.run(tmp_path, CooccurStore(path=tmp_path / "c.json"), embed_store=es)
    assert out["fp_patches"] >= 1
    assert out["fp_auto_merge_rate"] > 0.0


def test_probe_integrity_differential(synthetic_vault):
    # (a) every write-path transform leaves the clean fixture clean.
    assert integrity_probe(synthetic_vault)["rate"] == 1.0
    # (b) an introduced violation is caught.
    assert lint.new_violations("fine", "fine\n```python\nx = 1") == {"unclosed-code-fence": 1}
    # (c) a pre-existing violation diffed against itself never counts.
    assert lint.new_violations("bad```", "bad```") == {}
