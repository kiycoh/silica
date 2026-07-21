from silica.kernel import relatedness as R


def test_lexical_leg_folds_into_fusion():
    # Only the lexical leg proposes -> it survives fusion (like recall_rank).
    out = R._fuse(None, None, k=5, lexical_rank=[("notes/x", 3.0), ("notes/y", 1.0)])
    paths = [r.path for r in out]
    assert paths == ["notes/x", "notes/y"]
    assert any(e.startswith("lex:") for e in out[0].evidence)


def test_lexical_none_is_bit_identical_abstain():
    a = R._fuse([("p", "P", 0.9)], None, k=5)
    b = R._fuse([("p", "P", 0.9)], None, k=5, lexical_rank=None)
    assert [r.path for r in a] == [r.path for r in b]
