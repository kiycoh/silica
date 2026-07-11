# Security Policy

Silica's whole thesis is that an LLM should not be trusted with your vault without a guardrail. Security reports are taken in that spirit.

## Reporting a vulnerability

**Do not open a public issue for a security problem.** Report it privately through GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
(the **Security** tab → *Report a vulnerability*). If that is unavailable, email the maintainer.

Please include: what you did, what you expected, what happened, and, if it touches the vault, whether it could **corrupt, orphan, or leak** notes without passing the Injector FSM. A minimal reproduction helps most.

Expect an acknowledgement within a few days. This is a solo, pre-1.0 project; fixes are best-effort, prioritized by blast radius.

## What counts as a vulnerability

In scope is anything that lets a write reach the vault **outside the FSM's contracts**:

- a side channel that mutates vault files without passing the Injector FSM;
- an edit that orphans a note or breaks a link and is **not** caught by the verify gate or graph-safe move;
- external content (web search, ingested files) reaching the vault without the zero-trust `Inbox/` staging;
- a rollback (`/undo`, `/revert`, git safety net) that silently fails to restore prior state;
- secret leakage (API keys from `.env`) into notes, logs, or exports.

## What is a known limitation, not a vulnerability

Stated plainly in the README's *Design contracts* honesty note:

- The contracts are enforced control flow, **not yet crash-verified**. A process killed mid-write may leave partial state; the chaos harness that would prove otherwise is in progress. Reports that "a `kill -9` mid-write left the vault dirty" are known, not new.
- "Coherence" is a heuristic, not a theorem. A semantically wrong-but-well-formed edit is a quality bug, not a security hole.

## Supported versions

Only the latest commit on the default branch is supported. There are no backported security fixes.
