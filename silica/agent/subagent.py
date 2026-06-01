"""LeashedSubAgent — a small, tightly-bounded worker that runs on the worker model.

A leashed sub-agent consumes one WorkItem at a time and is allowed to write, but
only through its Leash + the commit_ops micro-gate.  It runs on the *worker* model
(role="worker" → small local model via LM Studio), concurrently with the Injector.

v1 implements the **dedup** behaviour: given a borderline pair (an incoming concept
vs. an existing larger note), it decides whether they are the same concept and, if
so, appends only the genuinely-new information into the existing note as a single
`patch`.  The dedup_leash guarantees it can do nothing else.

The decision step is a single structured-output call (cheap, deterministic enough
for an 8B model); the architecture leaves room for a multi-turn explore loop later
(bounded by leash.max_turns / leash.timeout_s).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from silica.config import CONFIG
from silica.agent.commit import commit_ops
from silica.agent.leash import dedup_leash, refiner_leash, orphan_leash
from silica.kernel.ops import Op, OpType
from silica.planner.workqueue import WorkItem

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "workers"


class DedupDecision(BaseModel):
    is_duplicate: bool
    rationale: str = ""
    addition: str = ""


class RefineResult(BaseModel):
    content: str = ""


class OrphanLinkDecision(BaseModel):
    links: list[str] = []
    rationale: str = ""


def _load_prompt(name: str) -> str:
    path = _PROMPT_DIR / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def run_subagent_batch(
    items: list[WorkItem],
    config: Any = CONFIG,
    *,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Run a batch of WorkItems through leashed sub-agents in parallel.

    Used by the ad-hoc /dedup and /refine commands (out of the inject pipeline).
    LeashedSubAgent is stateless beyond its config, so one instance is safely
    shared across threads; commit_ops serialises same-note writes via path_lease.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not items:
        return {"items": 0, "summary": {}, "results": []}

    mw = max(1, int(max_workers or getattr(config, "subagent_max_concurrent", 3)))
    agent = LeashedSubAgent(config)

    with ThreadPoolExecutor(max_workers=mw, thread_name_prefix="subagent") as ex:
        paired = list(ex.map(lambda it: (it, agent.handle(it)), items))

    summary: dict[str, int] = {}
    for _it, res in paired:
        s = res.get("status", "done")
        summary[s] = summary.get(s, 0) + 1
    return {
        "items": len(items),
        "summary": summary,
        "results": [{"target": it.target_path, **res} for it, res in paired],
    }


class LeashedSubAgent:
    """Dispatches a WorkItem to the matching leashed behaviour."""

    def __init__(self, config: Any = CONFIG):
        self.config = config

    # --- public entrypoint ------------------------------------------------

    def handle(self, item: WorkItem) -> dict[str, Any]:
        try:
            if item.kind == "dedup":
                return self._run_dedup(item)
            if item.kind == "refine":
                return self._run_refine(item)
            if item.kind == "enrich":
                return self._run_enrich(item)
            if item.kind == "orphan":
                return self._run_orphan(item)
            return {"status": "skipped", "reason": f"unknown work kind '{item.kind}'"}
        except Exception as e:  # never let a sub-agent crash the pool
            logger.warning("LeashedSubAgent error on %s item %s: %s", item.kind, item.id, e)
            return {"status": "error", "error": str(e)}

    # --- dedup behaviour --------------------------------------------------

    def _run_dedup(self, item: WorkItem) -> dict[str, Any]:
        ctx = item.context
        candidate_path = item.target_path
        budget = 8000

        # Read the existing (larger) note — read-only.
        try:
            from silica.driver import DRIVER
            candidate_body = DRIVER.read_note(candidate_path).content or ""
        except Exception as e:
            return {"status": "skipped", "reason": f"candidate unreadable: {e}"}

        decision = self._decide_dedup(
            concept=ctx.get("concept", ""),
            excerpt=ctx.get("excerpt", ""),
            candidate_name=ctx.get("candidate", candidate_path),
            candidate_body=candidate_body[:budget],
            score=ctx.get("score", 0.0),
            full_score=ctx.get("full_score", ctx.get("score", 0.0)),
            title_score=ctx.get("title_score", 0.0),
        )

        if not decision.is_duplicate or not decision.addition.strip():
            return {
                "status": "no_merge",
                "is_duplicate": decision.is_duplicate,
                "rationale": decision.rationale,
            }

        # Build the single append-only patch and commit it under the dedup leash.
        hub = ctx.get("hub")
        inbox_file = ctx.get("inbox_file", "")
        op = Op(
            op=OpType.patch,
            heading=ctx.get("concept", "") or "merged concept",
            source_basename=os.path.basename(inbox_file) if inbox_file else "dedup",
            path=candidate_path,
            snippet=decision.addition,
            hub=hub,
            reason=f"dedup merge: {decision.rationale[:120]}",
        )
        leash = dedup_leash(candidate_path, hub=hub)
        result = commit_ops(
            [op],
            target_dir=os.path.dirname(candidate_path),
            hub=hub,
            leash=leash,
        )
        result.setdefault("rationale", decision.rationale)
        return result

    # --- refine behaviour -------------------------------------------------

    def _run_refine(self, item: WorkItem) -> dict[str, Any]:
        target_path = item.target_path
        try:
            from silica.driver import DRIVER
            original = DRIVER.read_note(target_path).content or ""
        except Exception as e:
            return {"status": "skipped", "reason": f"target unreadable: {e}"}

        if not original.strip():
            return {"status": "skipped", "reason": "empty note"}

        refined = self._refine_note(target_path, original)
        if not refined.content.strip():
            return {"status": "no_change", "reason": "refiner produced no content"}

        hub = item.context.get("hub")
        op = Op(
            op=OpType.overwrite,
            heading=os.path.splitext(os.path.basename(target_path))[0],
            source_basename=os.path.basename(target_path),
            path=target_path,
            content=refined.content,
            hub=hub,
            reason="stylistic refine",
        )
        # refiner_leash enforces anti-info-loss (wikilinks preserved + length floor).
        leash = refiner_leash(target_path, hub=hub)
        result = commit_ops(
            [op],
            target_dir=os.path.dirname(target_path),
            hub=hub,
            leash=leash,
            read_note=lambda _p: original,
        )
        return result

    # --- enrich behaviour -------------------------------------------------

    def _run_enrich(self, item: WorkItem) -> dict[str, Any]:
        target_path = item.target_path
        try:
            from silica.driver import DRIVER
            original = DRIVER.read_note(target_path).content or ""
        except Exception as e:
            return {"status": "skipped", "reason": f"target unreadable: {e}"}

        hub = item.context.get("hub") or os.path.splitext(os.path.basename(target_path))[0]
        enriched = self._enrich_note(target_path, original, hub)
        if not enriched.content.strip():
            return {"status": "no_change", "reason": "enricher produced no content"}

        op = Op(
            op=OpType.overwrite,
            heading=os.path.splitext(os.path.basename(target_path))[0],
            source_basename=os.path.basename(target_path),
            path=target_path,
            content=enriched.content,
            hub=hub,
            reason="semantic enrichment",
        )
        # We can use refiner_leash as it guarantees anti-info-loss (wikilinks preserved + length floor)
        leash = refiner_leash(target_path, hub=hub)
        result = commit_ops(
            [op],
            target_dir=os.path.dirname(target_path),
            hub=hub,
            leash=leash,
            read_note=lambda _p: original,
        )
        return result

    # --- orphan connector behaviour ---------------------------------------

    def _run_orphan(self, item: WorkItem) -> dict[str, Any]:
        target = item.target_path
        candidates = item.context.get("candidates", [])  # [{"name":..., "path":...}]
        if not candidates:
            return {"status": "no_candidates"}

        try:
            from silica.driver import DRIVER
            body = DRIVER.read_note(target).content or ""
        except Exception as e:
            return {"status": "skipped", "reason": f"orphan unreadable: {e}"}

        decision = self._decide_links(target, body[:8000], candidates)
        # Only keep links that were actually offered as candidates — never let the
        # model invent a target (which would just create another dangling link).
        candidate_names = {c.get("name", "") for c in candidates}
        valid = [n for n in decision.links if n in candidate_names]
        if not valid:
            return {"status": "no_link", "rationale": decision.rationale}

        snippet = "## Related\n\n" + "\n".join(f"- [[{n}]]" for n in valid) + "\n"
        hub = item.context.get("hub")
        op = Op(
            op=OpType.patch,
            heading="Related",
            source_basename="orphan",
            path=target,
            snippet=snippet,
            hub=hub,
            reason=f"orphan connect: {decision.rationale[:120]}",
        )
        leash = orphan_leash(target, hub=hub)
        result = commit_ops([op], target_dir=os.path.dirname(target), hub=hub, leash=leash)
        result.setdefault("linked", valid)
        return result

    # --- worker-model calls (isolated for testability) --------------------

    def _decide_links(
        self,
        target_path: str,
        body: str,
        candidates: list[dict],
    ) -> OrphanLinkDecision:
        from silica.agent.providers import get_provider
        from silica.kernel.sanitize import parse_json

        prompt = _load_prompt("orphan_prompt.txt")
        cand_block = "\n".join(
            f"{i+1}. {c.get('name', c.get('path', '?'))}"
            for i, c in enumerate(candidates)
        )
        user_message = (
            f"{prompt}\n\n---\nORPHAN NOTE ({target_path}):\n{body}\n\n"
            f"---\nCANDIDATES:\n{cand_block}\n"
        )
        provider = get_provider(self.config, role="worker")
        response = provider.call_llm(
            messages=[{"role": "user", "content": user_message}],
            tools=None,
            response_schema=OrphanLinkDecision,
            max_tokens=512,
        )
        raw = response.text or ""
        try:
            parsed, _ = parse_json(raw, strict=False)
            if isinstance(parsed, dict):
                links = parsed.get("links", [])
                return OrphanLinkDecision(
                    links=[str(x) for x in links] if isinstance(links, list) else [],
                    rationale=str(parsed.get("rationale", "")),
                )
        except Exception as e:
            logger.debug("orphan link decision parse failed: %s", e)
        return OrphanLinkDecision(links=[], rationale="unparseable decision")

    def _refine_note(self, target_path: str, original: str) -> RefineResult:
        from silica.agent.providers import get_provider
        from silica.kernel.sanitize import parse_json

        prompt = _load_prompt("refiner_prompt.txt")
        user_message = f"{prompt}\n\n---\nNOTE ({target_path}):\n{original}\n"
        provider = get_provider(self.config, role="worker")
        response = provider.call_llm(
            messages=[{"role": "user", "content": user_message}],
            tools=None,
            response_schema=RefineResult,
            max_tokens=8192,
        )
        raw = response.text or ""
        try:
            parsed, _ = parse_json(raw, strict=False)
            if isinstance(parsed, dict) and "content" in parsed:
                return RefineResult(content=str(parsed["content"]))
        except Exception as e:
            logger.debug("refine parse failed: %s", e)
        return RefineResult(content="")

    def _enrich_note(self, target_path: str, original: str, hub: str) -> RefineResult:
        from silica.agent.providers import get_provider
        from silica.kernel.sanitize import parse_json
        from silica.kernel.context_builder import build_context

        system_prompt = (
            "You are an academic assistant expert in writing and structuring notes in Obsidian Flavored Markdown (OFM) in English.\n"
            "Your task is to enrich the note specified by the target.\n"
            "Fundamental rules:\n"
            "1. Produce a rigorous, complete, and exhaustive academic text in English.\n"
            "2. Preserve all factual information and concepts already present in the note (anti-deletion policy). Do not remove pre-existing information, but expand upon it.\n"
            "3. Perform structuring in Obsidian Flavored Markdown: use callouts (> [!tip], > [!note]), LaTeX equation blocks ($$ ... $$) if appropriate, lists, and bold text.\n"
            f"4. You must include a wikilink [[{hub}]] to the hub/parent note (for example in a final section called '# Relations' or '# Connections').\n"
            "5. Return the result structured in JSON format containing a single key 'content' with the full body of the note (including normalized and updated YAML frontmatter tags, and the enriched body)."
        )

        title = os.path.splitext(os.path.basename(target_path))[0]
        note_payload = f"Title: {title}\nPath: {target_path}\nCurrent content:\n{original}"
        ctx = build_context(checkpoint_id="enrich", payload=note_payload)
        user_message = f"Enrich the following note.\n\n{ctx}"

        provider = get_provider(self.config, role="worker")
        response = provider.call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            tools=None,
            response_schema=RefineResult,
            max_tokens=8192,
        )
        raw = response.text or ""
        try:
            parsed, _ = parse_json(raw, strict=False)
            if isinstance(parsed, dict) and "content" in parsed:
                return RefineResult(content=str(parsed["content"]))
        except Exception as e:
            logger.debug("enrich parse failed: %s", e)
        return RefineResult(content="")

    def _decide_dedup(
        self,
        *,
        concept: str,
        excerpt: str,
        candidate_name: str,
        candidate_body: str,
        score: float = 0.0,
        full_score: float = 0.0,
        title_score: float = 0.0,
    ) -> DedupDecision:
        from silica.agent.providers import get_provider
        from silica.kernel.sanitize import parse_json

        prompt = _load_prompt("dedup_prompt.txt")

        # Build the score block shown to the model.
        # When both metrics are available we surface them separately so the model
        # can interpret the signal correctly: a high title score with a low body
        # score means "topically related but distinct" — very different from a
        # uniformly high score which strongly suggests a true duplicate.
        if title_score > 0.0 and full_score > 0.0:
            score_block = (
                f"SEMANTIC CLOSENESS SCORE: {score:.3f} (effective = max of the two below)\n"
                f"  • Full-note similarity (body + title):  {full_score:.3f}\n"
                f"  • Title-only similarity:                {title_score:.3f}\n"
                f"Interpretation:\n"
                f"  - High full-note score (>0.80): bodies cover the same topic → likely duplicate.\n"
                f"  - High title score with low body score: notes are topically related but\n"
                f"    cover distinct aspects (e.g. 'ROS' vs 'JSON in ROS 2') → prefer linking\n"
                f"    over merging; set is_duplicate=false unless content genuinely overlaps."
            )
        else:
            score_block = (
                f"SEMANTIC CLOSENESS SCORE: {score:.3f} (0.0 to 1.0, where 1.0 is identical)\n"
                f"Use this metric as an indicator. High scores (>0.85) strongly suggest "
                f"duplicates, while lower scores might represent related but distinct topics."
            )

        user_message = (
            f"{prompt}\n\n"
            f"---\n{score_block}\n"
            f"---\nCANDIDATE NOTE ({candidate_name}):\n{candidate_body}\n\n"
            f"---\nINCOMING CONCEPT: {concept}\nEXCERPT:\n{excerpt}\n"
        )
        provider = get_provider(self.config, role="worker")
        response = provider.call_llm(
            messages=[{"role": "user", "content": user_message}],
            tools=None,
            response_schema=DedupDecision,
            max_tokens=1024,
        )
        raw = response.text or ""
        try:
            parsed, _ = parse_json(raw, strict=False)
            if isinstance(parsed, dict):
                return DedupDecision(**{
                    "is_duplicate": bool(parsed.get("is_duplicate", False)),
                    "rationale": str(parsed.get("rationale", "")),
                    "addition": str(parsed.get("addition", "")),
                })
        except Exception as e:
            logger.debug("dedup decision parse failed: %s", e)
        # Conservative default: when in doubt, do not merge.
        return DedupDecision(is_duplicate=False, rationale="unparseable decision")
