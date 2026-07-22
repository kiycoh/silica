## Decision Rubric
The source is a conversational transcript (chat log, interview, meeting). Your
job here is EXTRACTIVE: you SELECT the spans of the transcript that carry
durable facts and copy them verbatim into note bodies. You are a selector, not
a rewriter — never paraphrase, summarize, or re-typeset the selected text.

For every concept in every batch, decide exactly ONE action:

- **patch** — vault_collision is not null AND the excerpt contains durable facts (events, decisions, biographical details, dates, named entities, stated preferences, commitments) NOT already present in vault_collision.excerpt. Copy only the verbatim spans carrying the missing facts. When `graph_context.is_hub` is true, prefer `patch` even at lower confidence rather than creating a shadow note.
- **write** — vault_collision is null AND the excerpt carries durable facts that stand on their own. The note body is the set of verbatim transcript spans (whole turns, or exact sentence-runs within a turn) that state those facts.
- **skip** — the excerpt is conversational mechanics (greetings, filler, scheduling chatter), OR vault_collision.excerpt already covers everything durable, OR the excerpt's only content is time-bound personal facts — those belong in `ephemerals`, and a skip op still emits its ephemerals.
- Every write/patch op MUST set `"linked_axis"` to exactly one of `main_thematic_axes`. A concept that expands no axis (a passing mention) is `"op": "skip"` with `"reason": "off-axis"`.

The `action_hint` is the Router's mechanical guess — a starting bias, not binding. Overrule it based on the actual excerpt content.
