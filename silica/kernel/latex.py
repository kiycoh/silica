"""Math-aware text substitution for the sanitize pipeline.

`replace_outside_math` lets the distiller post-processor rewrite prose (e.g. turn
double-escaped `\\n` into real newlines) without shredding `\\nabla`/`\\neq` or
splitting inline math. Pure, no LLM — fits the kernel contract.

(The old `normalize_latex` backstop lived here too; removed 2026-06-30 once the
real corruption root — the cli backend write channel doubling `\\`→`\\\\` — was
fixed, leaving the distiller body provably clean. See cli_backend.create.)
"""
from __future__ import annotations

import re

# Inline ($...$) must not cross newlines; block ($$...$$) may.
_MATH = re.compile(r"\$\$.*?\$\$|\$[^\n$]+?\$", re.DOTALL)


def replace_outside_math(text: str, old: str, new: str) -> str:
    """`text.replace(old, new)` everywhere EXCEPT inside `$...$` / `$$...$$` spans.

    Lets the distiller post-processor turn double-escaped prose newlines into real
    ones without shredding `\\nabla`/`\\neq` or splitting inline math.
    """
    out: list[str] = []
    last = 0
    for m in _MATH.finditer(text):
        out.append(text[last:m.start()].replace(old, new))
        out.append(m.group(0))  # math span: verbatim
        last = m.end()
    out.append(text[last:].replace(old, new))
    return "".join(out)
