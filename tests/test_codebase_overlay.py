from silica.router.recipe_parser import load_recipe


def test_codebase_overlay_loads_and_overrides_gate():
    base = load_recipe("injector")
    overlaid = load_recipe("injector", domain="codebase")
    # overlay must apply without falling back to base
    assert overlaid["gates"].get("rejection_rate_max") == 0.05
    # base phases are preserved (overlay may not add/remove)
    assert [p["id"] for p in overlaid["phases"]] == [p["id"] for p in base["phases"]]
