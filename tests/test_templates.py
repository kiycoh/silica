from silica.kernel.templates import template_spoke


def test_spoke_does_not_double_wrap_bracketed_parent_and_related():
    """Distiller often emits parent/related already as [[X]] (hence write.py's
    .strip('[]')). template_spoke must not re-wrap them into [[[[X]]]], which
    Obsidian reads as an unresolved link and trips the graph regression gate."""
    out = template_spoke(
        heading="Modelli Linguistici Generativi (GPT)",
        snippet="testo",
        hub="[[IA Generativa]]",
        related=["[[Reti Neurali Profonde (Deep Learning)]]", "calcolo parallelo"],
        parent="[[Rinascita dell'IA]]",
    )
    assert "[[[[" not in out and "]]]]" not in out, out
    assert 'parent note: "[[Rinascita dell\'IA]]"' in out
    assert '"[[Reti Neurali Profonde (Deep Learning)]]"' in out
    assert '"[[calcolo parallelo]]"' in out  # bare name still wrapped exactly once
    assert '"[[IA Generativa]]"' in out
