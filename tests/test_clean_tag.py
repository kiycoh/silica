# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

from silica.kernel.frontmatter import clean_tag


def test_clean_tag_keeps_leading_digit_fused_to_word():
    # A20: a digit that is part of a word must survive.
    assert clean_tag("3d") == "3d"
    assert clean_tag("2fa") == "2fa"
    assert clean_tag("3D-Printing") == "3d-printing"
    assert clean_tag("web3") == "web3"  # trailing digit already safe


def test_clean_tag_strips_list_ordinal():
    # Real numbered-list ordinals (digit + separator + space) are still stripped.
    assert clean_tag("1. Machine Learning") == "machine-learning"
    assert clean_tag("2) Notes") == "notes"
