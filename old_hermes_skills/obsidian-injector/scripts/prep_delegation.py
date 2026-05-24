"""Prepare delegate_task arguments for the Fact Distiller pipeline.

Reads the protocol template verbatim, substitutes TARGET (and other non-payload placeholders), and
emits a JSON array of task dicts ready to pass directly to
`delegate_task(tasks=...)`. The payload path is passed via per-task context, not substituted into the protocol.

This removes the failure mode where the Router paraphrases the protocol while
inlining it into the `context` field of delegate_task. The Router runs this
script via execute_code, captures its stdout, and passes the JSON verbatim
to delegate_task — no LLM-mediated summarization in between.

Usage:
    python3 prep_delegation.py \\
        --protocol $SKILL_ROOT/prompts/distiller_prompt.txt \\
        --payload /tmp/distiller_payload_0.json \\
        --payload /tmp/distiller_payload_1.json \\
        --payload /tmp/distiller_payload_2.json \\
        --substitute TARGET="$VAULT/.../Agenti autonomi" \\
        --out /tmp/delegation_args.json

Then in the Router's reasoning:
    1. Read /tmp/delegation_args.json (it's an array of task dicts).
    2. Invoke the top-level delegate_task tool directly (not via python code/execute_code) with the tasks array.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def parse_substitution(s: str) -> tuple:
    """Parse 'KEY=VALUE' into (KEY, VALUE). Raises ValueError on malformed input."""
    if "=" not in s:
        raise ValueError(f"--substitute expects KEY=VALUE, got: {s!r}")
    key, _, value = s.partition("=")
    key = key.strip()
    if not key:
        raise ValueError(f"--substitute KEY cannot be empty: {s!r}")
    return key, value


def render_shared_context(protocol_text: str, substitutions: dict) -> str:
    """Substitute all {KEY} placeholders EXCEPT {PAYLOAD_PATH}."""
    body = protocol_text
    for key, value in substitutions.items():
        if key != "PAYLOAD_PATH":
            body = body.replace("{" + key + "}", value)
    return body


def find_unsubstituted_critical(body: str) -> list:
    """Return critical placeholders still present after substitution.

    The protocol contains literal `{` and `}` in JSON schema examples, so we
    can't naively check every `{WORD}` pattern. We only flag placeholders we
    know the pipeline depends on at runtime.
    """
    critical = ["{TARGET}", "{HUB_NAME}", "{LANGUAGE}"]
    return [p for p in critical if p in body]


def build_tasks(protocol_text: str, payload_paths: list, substitutions: dict,
                max_iterations: int, toolset: list, goal_template: str, out_path: Path) -> list:
    
    shared_body = render_shared_context(protocol_text, substitutions)
    leftover = find_unsubstituted_critical(shared_body)
    if leftover:
        sys.stderr.write(
            f"[PREP-DELEGATION] Warning: Shared protocol has unsubstituted "
            f"critical placeholders: {leftover}. The Distiller will see "
            f"these literally in its protocol.\n"
        )
        
    shared_prompt_path = out_path.with_name(f"{out_path.stem}_prompt.txt")
    shared_prompt_path.write_text(shared_body.strip(), encoding="utf-8")
    
    sys.stderr.write(
        f"[PREP-DELEGATION] Wrote shared prompt file ({len(shared_body.strip())} chars) to: {shared_prompt_path.resolve()}\n"
    )

    tasks = []
    for i, p in enumerate(payload_paths):
        payload_path_obj = Path(p)
        
        task_context = (
            f"Prompt: {shared_prompt_path.resolve()}\n"
            f"Payload: {payload_path_obj.resolve()}\n"
            f"Read the Prompt, then process the Payload."
        )
        
        tasks.append({
            "goal": goal_template.format(index=i, path=p),
            "context": task_context,
            "toolsets": toolset,
            "max_iterations": max_iterations,
        })
        
    return tasks


def main():
    ap = argparse.ArgumentParser(
        description="Prepare delegate_task arguments for the Distiller subagent."
    )
    ap.add_argument("--protocol", required=True, type=Path,
                    help="Path to distiller_prompt.txt (protocol template)")
    ap.add_argument("--payload", action="append", required=True, dest="payloads",
                    help="Payload JSON path. Repeat for parallel batches. "
                         "E.g. --payload /tmp/p_0.json --payload /tmp/p_1.json")
    ap.add_argument("--substitute", action="append", default=[],
                    help="Substitute {KEY} placeholder in protocol with VALUE. "
                         "Format: --substitute KEY=VALUE. Repeatable. "
                         "{PAYLOAD_PATH} is left as a literal and passed via context.")
    ap.add_argument("--max-iterations", type=int, default=15,
                    help="Per-subagent iteration cap (default: 15)")
    ap.add_argument("--toolset", default="file",
                    help="Comma-separated toolsets for the subagent (default: 'file')")
    ap.add_argument("--goal-template",
                    default="Distill batch {index} of inbox-vs-vault concepts",
                    help="Format string for per-task goal. Vars: {index}, {path}")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output path for the JSON task array")
    args = ap.parse_args()

    if not args.protocol.exists():
        print(json.dumps({"error": f"protocol file {args.protocol} not found"}))
        sys.exit(1)

    missing = [p for p in args.payloads if not Path(p).exists()]
    if missing:
        print(json.dumps({"error": f"payload file(s) not found: {missing}"}))
        sys.exit(1)

    try:
        substitutions = dict(parse_substitution(s) for s in args.substitute)
    except ValueError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    protocol_text = args.protocol.read_text(encoding="utf-8")
    toolset_list = [t.strip() for t in args.toolset.split(",") if t.strip()]

    tasks = build_tasks(
        protocol_text=protocol_text,
        payload_paths=args.payloads,
        substitutions=substitutions,
        max_iterations=args.max_iterations,
        toolset=toolset_list,
        goal_template=args.goal_template,
        out_path=args.out
    )

    rendered = json.dumps(tasks, ensure_ascii=False, indent=2)
    checksum = hashlib.sha256(rendered.encode("utf-8")).hexdigest()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    checksum_path = args.out.with_suffix(".checksum")
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    checksum_path.write_text(checksum, encoding="utf-8")
    sys.stderr.write(f"[PREP-DELEGATION] Wrote checksum to: {checksum_path.resolve()}\n")

    total_chars = sum(len(t["context"]) for t in tasks)
    avg_chars = total_chars // len(tasks) if tasks else 0
    sys.stderr.write(
        f"[PREP-DELEGATION] Prepared {len(tasks)} task(s), "
        f"context body avg {avg_chars} chars, total {total_chars} chars\n"
    )


if __name__ == "__main__":
    main()
