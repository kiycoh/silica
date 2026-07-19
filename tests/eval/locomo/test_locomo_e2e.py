# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""Offline harness tests for the LoCoMo e2e leg (fsm ingest + agent answer).

Zero LLM: every product seam is monkeypatched, per the adapter-test pattern.
"""
import json

from tests.eval.locomo import runner


# The one-shot system prompt as shipped today (baseline cell). The e2e leg's
# comparability rule: agent and one-shot prompts differ ONLY in the memory
# delivery sentence, so the judge sees the same contract.
_ONESHOT_SNAPSHOT = (
    "You are a helpful assistant answering questions from your memory of "
    "past conversations between Ann and Bob. Today's "
    "date is 2023-06-01. Use ONLY the memory provided. A 'Personal memory' "
    "section, when present, lists dated facts distilled from those "
    "conversations — treat them as reliable memory on par with the session "
    "transcripts. Answer concisely with only the information asked for. If "
    "the memory does not contain the answer, reply that you do not have "
    "that information — never guess."
)


def test_answer_contract_shared_and_oneshot_unchanged():
    open_ = runner._CONTRACT_OPEN.format(a="Ann", b="Bob", now="2023-06-01")
    oneshot = open_ + runner._ONESHOT_DELIVERY + runner._CONTRACT_CLOSE
    agent = open_ + runner._AGENT_DELIVERY + runner._CONTRACT_CLOSE
    assert oneshot == _ONESHOT_SNAPSHOT
    assert agent != oneshot
    assert agent.startswith(open_) and agent.endswith(runner._CONTRACT_CLOSE)
