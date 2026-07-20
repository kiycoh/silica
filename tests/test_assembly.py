from silica.kernel.assembly import Unit, fill_budget


def test_seeds_never_trimmed_even_over_budget():
    seeds = [Unit(path="a", text="x" * 5000, is_seed=True, rank=0)]
    kept, trunc = fill_budget(seeds, [], budget=3000)
    assert [u.path for u in kept] == ["a"]        # protect-seeds invariant
    assert trunc.dropped == []


def test_periphery_fills_by_rank_then_reports_drops():
    seeds = [Unit(path="s", text="x" * 1000, is_seed=True, rank=0)]
    periphery = [
        Unit(path="p1", text="y" * 800, is_seed=False, rank=0),
        Unit(path="p2", text="z" * 800, is_seed=False, rank=1),
        Unit(path="p3", text="w" * 800, is_seed=False, rank=2),
    ]
    kept, trunc = fill_budget(seeds, periphery, budget=2600)  # room for s + p1 + p2
    assert [u.path for u in kept] == ["s", "p1", "p2"]
    assert trunc.dropped == ["p3"]
    assert trunc.kept == 3


from silica.kernel.assembly import relevel_headers, squash, AssembledBlock


def test_relevel_shifts_headings_capped_at_six():
    body = "# Title\n\ntext\n\n## Sub\n"
    assert relevel_headers(body, 1) == "## Title\n\ntext\n\n### Sub\n"
    assert relevel_headers("###### Deep\n", 2) == "###### Deep\n"  # capped
    assert relevel_headers(body, 0) == body


def test_squash_groups_two_units_under_same_hub():
    units = [
        Unit(path="spoke-a", text="# A\naaa", is_seed=True, rank=0),
        Unit(path="spoke-b", text="# B\nbbb", is_seed=True, rank=1),
    ]
    hub_of = {"spoke-a": "Hub", "spoke-b": "Hub"}
    crumb = {"spoke-a": "Hub > A", "spoke-b": "Hub > B"}
    blocks = squash(units, hub_of, crumb)
    assert len(blocks) == 1
    b = blocks[0]
    assert b.hub == "Hub"
    assert b.members == ["spoke-a", "spoke-b"]
    assert b.text.startswith("Hub > A")  # breadcrumb prefix
    assert "# Hub" in b.text          # hub header present
    assert "## A" in b.text and "## B" in b.text  # members re-leveled under it


def test_single_seed_is_not_squashed():
    units = [Unit(path="only", text="# Solo\nx", is_seed=True, rank=0)]
    blocks = squash(units, {"only": "Hub"}, {"only": "Hub > Solo"})
    assert len(blocks) == 1
    assert blocks[0].hub is None            # degenerate: not a squash
    assert blocks[0].members == ["only"]
    assert blocks[0].text.startswith("Hub > Solo")  # breadcrumb prefix
