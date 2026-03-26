# ProjectDocs — guide for assistant (Claude)

Purpose
This file explains the structure and authoritative sources inside the repository `docs/`. Use it as the canonical map when generating text, code, or extracting claims.

Repository path: `docs/`

Directory layout (important files only)
- `.quarto/` : Quarto build internals (ignore for content extraction).
- `claims/`  : Claim artifacts (per-claim notes). **Authoritative for claim-level intent.**
- `figures/` : Figures and diagrams (images) used in reports. Prefer figure files when diagram detail matters.
- `styles/`  : Quarto/CSS styles (ignore for research content).

Top-level documents (authoritative priority)
1. `task_definition.qmd` — primary statement of research goal and scope (use this first).
2. `task_definition_method.qmd` — authoritative method description (use this for implementation details).
3. `claims.qmd` — condensed claims and hypotheses (use for claim extraction / RQ creation).
4. `literature_review.qmd` — background and citation context.
5. `index.qmd` and `references.bib` — site index and bibliographic sources.

How to use this doc when responding
- Priority rule: for implementation use `task_definition_method.qmd` > `claims.qmd` > figures.
- If any two sources conflict, flag the conflict and ask the user which source is authoritative.
- Always quote file name and (short) excerpt when you assert a fact from the docs.

Prompt snippet examples (copy into conversation)
- “Use `task_definition_method.qmd` as canonical. Summarize its model architecture in 4 bullets.”
- “Compare claim X in `claims/q1.md` to the method in `task_definition_method.qmd`. List 3 differences.”

Note on attachments
Some previously uploaded auxiliary files have expired. Re-upload any missing PDFs or images you want the assistant to read.