from __future__ import annotations

from pydantic import BaseModel

from silica.tools import tool, TOOLS


def test_tool_defaults_to_lazy_collapse_and_no_summarize():
    class _A(BaseModel):
        x: int = 0

    @tool(_A, cls="atomic")
    def silica_dummy_lazy(x: int = 0):
        "doc"
        return {"x": x}

    t = TOOLS["silica_dummy_lazy"]
    assert t.collapse == "lazy"
    assert t.summarize is None


def test_tool_accepts_eager_and_summarize():
    class _B(BaseModel):
        x: int = 0

    def _sum(result: dict) -> str:
        return f"x={result['x']}"

    @tool(_B, cls="composed", collapse="eager", summarize=_sum)
    def silica_dummy_eager(x: int = 0):
        "doc"
        return {"x": x}

    t = TOOLS["silica_dummy_eager"]
    assert t.collapse == "eager"
    assert t.summarize({"x": 5}) == "x=5"
