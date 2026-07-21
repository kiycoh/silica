# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Pins the LexicalStore save/load roundtrip the write choke-point hook
relies on: a committed note becomes queryable after an upsert+save+load
cycle. The hook's index-presence gating and wiring are verified by
inspection of silica/router/states/write.py and orchestrator.py's
_flush_indexes (no lexical.json on disk => the hook never touches the
store, so non-lexical vaults stay byte-identical)."""
from silica.kernel.lexical import LexicalStore


def test_upsert_then_query_roundtrip(tmp_path):
    # The write hook's job reduced to its core: a committed note becomes queryable.
    idx = tmp_path / "lexical.json"
    s = LexicalStore(idx)
    s.upsert("notes/mars", "Mars rover", "Perseverance landed in Jezero crater.")
    s.save()
    reloaded = LexicalStore.load(idx)
    assert reloaded.rank("Jezero", k=3)[0][0] == "notes/mars"
