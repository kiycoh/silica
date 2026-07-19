"""silica_anneal: mechanical sweep of all deferred bundles + escalation steer."""
import orjson

LONG = (
    "Il pattern publish/subscribe disaccoppia produttori e consumatori tramite "
    "un broker che smista i messaggi per topic su reti inaffidabili. " * 2
)


def _park(monkeypatch, tmp_path):
    """Point the deferred store at a temp dir and return it."""
    from silica.kernel import deferred

    monkeypatch.setattr(deferred, "_store_dir", lambda: tmp_path / "deferred")
    deferred._stores.clear()
    return deferred.get_deferred_store()


def test_anneal_sweeps_all_bundles(tmp_vault, tmp_path, monkeypatch):
    from silica.tools.pipeline import silica_anneal

    tmp_vault.note("Reti/Reti.md", "# Reti\n")
    store = _park(monkeypatch, tmp_path)
    # Bundle 1: write op that passes validation now → written, bundle cleared.
    store.put(
        "aaa1", "inbox/a.md", "Reti", None,
        [{"op": "write", "heading": "PubSub", "source_basename": "a.md",
          "path": "Reti/PubSub.md", "title": "PubSub", "snippet": LONG}],
        rejection_reasons={"Reti/PubSub.md": "lint failed (stale)"},
        phase="VALIDATE",
    )
    # Bundle 2: op still failing (snippet under the 100-char gate).
    store.put(
        "bbb2", "inbox/b.md", "Reti", None,
        [{"op": "write", "heading": "Stub", "source_basename": "b.md",
          "path": "Reti/Stub.md", "title": "Stub", "snippet": "troppo corto"}],
        rejection_reasons={"Reti/Stub.md": "snippet too short"},
        phase="VALIDATE",
    )

    res = silica_anneal()

    assert res["bundles"] == 2
    assert res["written"] == 1
    assert res["still_deferred"] == 1
    assert store.get("aaa1") is None          # cleared
    assert store.get("bbb2") is not None      # still parked


def test_anneal_steer_fixes_with_stamped_reason(tmp_vault, tmp_path, monkeypatch):
    from silica.tools import pipeline

    tmp_vault.note("Reti/Reti.md", "# Reti\n")
    store = _park(monkeypatch, tmp_path)
    store.put(
        "ccc3", "inbox/c.md", "Reti", None,
        [{"op": "write", "heading": "Broker", "source_basename": "c.md",
          "path": "Reti/Broker.md", "title": "Broker", "snippet": "corto"}],
        rejection_reasons={"Reti/Broker.md": "snippet too short"},
        phase="VALIDATE",
    )

    prompts = []

    class _Resp:
        text = orjson.dumps([{
            "op": "write", "heading": "Broker", "source_basename": "c.md",
            "path": "Reti/Broker.md", "title": "Broker", "snippet": LONG,
        }]).decode()

    class _Provider:
        def call_llm(self, messages, tools=None, **kw):
            prompts.append(messages[0]["content"])
            return _Resp()

    monkeypatch.setattr("silica.agent.providers.get_provider", lambda *a, **k: _Provider())

    res = pipeline.silica_anneal(steer=True)

    [row] = res["results"]
    assert row["steer"]["status"] == "committed", row
    assert res["written"] == 1
    assert store.get("ccc3") is None  # written op removed → bundle gone
    # the stamped per-op reason reached the escalation prompt
    assert "snippet too short" in prompts[0]
