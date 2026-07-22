## Content Quality Requirements (EXTRACTIVE — enforced)
The body is built by COPYING spans from the excerpt, not by writing about it. A
mechanical validator checks every body line against the source and REJECTS the
op (costing the attempt) if any line is not a verbatim span. Follow these or the
op is rejected:

- **Copy, never reword.** Each body line MUST be an exact substring of the concept's `inbox_excerpt` — the same words, in the same order. Do not paraphrase, condense, translate, "clean up", or merge two turns into one sentence. Selecting is allowed (drop turns that carry no durable fact); rewriting is not.
- **Keep the transcript verbatim, dates included.** Do NOT resolve relative dates in the BODY — leave "May 20th", "last Tuesday", "two months ago" exactly as written. Absolute-date resolution happens ONLY in the `ephemerals` section (per the Ephemeral Facts rules), never in the note body.
- **Keep speaker attribution as written.** If the excerpt reads `Elena: ...`, copy the `Elena:` prefix with the line. Never strip or invent attribution.
- **No added prose.** Do not write connective sentences, summaries, or headers like "In this session ...". The body is spans and nothing else. A descriptive or summarizing body is distill-loss and is worse than no note.
- **No wikilinks in the body.** Do NOT insert `[[...]]` into copied spans — the autolink phase adds links mechanically after the write. Added `[[...]]` would corrupt the verbatim span.
- **Select whole turns, not minimal fragments.** When a turn carries a durable fact, copy the WHOLE turn (its full `Speaker: ...` line), not just the one sentence — the surrounding wording is context the reader needs and keeps the note above the placeholder-length floor. Only drop a turn entirely when it carries no durable fact. Never split mid-sentence in a way that drops words from the middle.
- **Note Title Elegance**: `title` controls the filename and H1; derive it from the subject of the facts ("Elena's pottery class", "Sam's job change"), grounded in the excerpt. The `heading` MUST still equal the payload concept name exactly (traceability anchor). The `path` MUST be `{TARGET}/<title>.md`. The title is the ONLY field you author freely — the body is copied.
