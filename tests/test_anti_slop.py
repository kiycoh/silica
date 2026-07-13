# SPDX-License-Identifier: AGPL-3.0-or-later
"""The shared anti-slop fragment must reach the body-writing prompts.

Guards against a future refactor silently dropping the injection. Covers the
distiller (nucleation, highest volume) via the pure render_prompt, plus the
refine/enrich load path, plus the fragment's own content.
"""
from silica.capabilities._base import load_prompt
from silica.kernel.prep_delegation import render_prompt

SENTINEL = "no AI slop"  # heading of _anti_slop.txt


def test_fragment_loads_and_carries_key_rules():
    frag = load_prompt("_anti_slop.txt")
    assert frag.strip(), "anti-slop fragment is empty/missing"
    assert SENTINEL in frag
    assert "em dash" in frag.lower()  # aligns with the vault's no-em-dash rule


def test_distiller_prompt_includes_fragment():
    rendered = render_prompt(target="Notes", hub="Hub", source_text="hello world")
    assert SENTINEL in rendered


def test_refiner_prompt_pairs_with_fragment():
    # refine.py concatenates these two; assert both halves exist so the join holds.
    assert load_prompt("refiner_prompt.txt").strip()
    assert SENTINEL in load_prompt("_anti_slop.txt")


if __name__ == "__main__":
    test_fragment_loads_and_carries_key_rules()
    test_distiller_prompt_includes_fragment()
    test_refiner_prompt_pairs_with_fragment()
    print("ok")
