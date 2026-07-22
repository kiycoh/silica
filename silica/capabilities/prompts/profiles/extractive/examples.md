## Few-Shot Example

### Example Input Payload:
{
  "schema_version": 1,
  "batches": [
    {
      "inbox_file": "/abs/path/to/inbox/session_2026-05-07.md",
      "concepts": [
        {
          "name": "pottery class",
          "action_hint": "create",
          "inbox_excerpt": "Elena: I finally signed up for the pottery class at the community center! It starts May 20th, every Tuesday evening.\nSam: That's great! Is that the one your sister teaches?\nElena: No, she teaches the advanced one. Mine is the beginners course with Mr. Alvarez.",
          "vault_collision": null
        },
        {
          "name": "greetings",
          "action_hint": "likely_skip",
          "inbox_excerpt": "Sam: Hey Elena! How have you been?\nElena: Good, good. Busy week!",
          "vault_collision": null
        }
      ]
    }
  ]
}

### Example Output (NO prose, NO markdown fences — JSON, then the Body Appendix):
{
  "main_thematic_axes": ["Elena's hobbies and classes", "family relationships", "conversational logistics"],
  "updates": [
    {
      "heading": "pottery class",
      "title": "Elena's pottery class",
      "op": "write",
      "path": "{TARGET}/Elena's pottery class.md",
      "source_basename": "session_2026-05-07.md",
      "hub": "{HUB_NAME}",
      "linked_axis": "Elena's hobbies and classes",
      "concepts": ["pottery class", "community center", "beginners course"],
      "snippet_ref": 1
    },
    {
      "heading": "greetings",
      "op": "skip",
      "source_basename": "session_2026-05-07.md",
      "reason": "conversational mechanics — no durable facts"
    }
  ],
  "ephemerals": [
    {"key": "elena.pottery_class.start_date", "text": "Elena's pottery class starts on 2026-05-20 (\"May 20th\"), Tuesday evenings"},
    {"key": "elena.pottery_class.teacher", "text": "Elena's beginners pottery class is taught by Mr. Alvarez"}
  ]
}
===SILICA-BODY 1===
Elena: I finally signed up for the pottery class at the community center! It starts May 20th, every Tuesday evening.
Elena: No, she teaches the advanced one. Mine is the beginners course with Mr. Alvarez.

Note what this body IS and IS NOT:
- Each line is an EXACT substring of the excerpt — copied, not reworded. Sam's question was dropped (no durable fact); the two Elena turns were kept whole.
- The body keeps "May 20th" verbatim; the absolute date 2026-05-20 appears ONLY in `ephemerals`, never in the body.
- No `[[wikilinks]]`, no summary line, no attribution stripped. A body like "Elena signed up for a beginners pottery class starting May 20th" would be REJECTED — it rewords instead of copying.
