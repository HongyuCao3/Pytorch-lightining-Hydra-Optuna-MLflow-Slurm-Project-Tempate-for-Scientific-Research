# docs.md — Documentation rules for Claude

## Purpose
Canonical map of `docs/` for content generation, editing, and claim extraction.

---

## Site Structure

Built with **Quarto website** (`docs/_quarto.yml`). Render with:
```bash
cd docs && quarto render
# or preview live:
cd docs && quarto preview
```

Output: `docs/_site/`. Dark mode theme (`darkly`). Sidebar navigation + prev/next buttons via `page-navigation: true`.

---

## Document Authority (priority order)

1. `task_definition.qmd` — primary statement of research goal and scope
2. `claims.qmd` — testable hypotheses; use for claim extraction and RQ creation
3. `method/overview.qmd` — authoritative method description; use for implementation details
4. `method/component_a.qmd`, `method/component_b.qmd` — per-component specs
5. `literature_review.qmd` — background and citation context
6. `experiments/*.qmd` — experimental results (fill in after runs)
7. `index.qmd` — dashboard; not authoritative for content, only for navigation

If two sources conflict, flag the conflict and ask the user which is authoritative.

---

## Page Order (sidebar / prev-next sequence)

```
index.qmd
task_definition.qmd
literature_review.qmd
claims.qmd
method/overview.qmd
method/component_a.qmd
method/component_b.qmd
experiments/overview.qmd
experiments/main.qmd
experiments/ablation.qmd
experiments/hyperparam.qmd
```

---

## Writing Style Rules

- **Language**: English only
- **Format**: bullet points for descriptions; numbered steps for procedures
- **Sentence length**: ≤ 15 words per bullet; one idea per bullet
- **No formulas** in text — reference equations in `references.bib` or code
- **Tense**: present tense for method descriptions; past tense for completed experiments

---

## Graphviz Diagram Rules (CRITICAL)

Every H2 subsection must have one `{dot}` diagram showing the logic of that subsection.

**Layout constraints** (strictly enforced):
- `rankdir=TB` or `rankdir=LR`; choose based on flow direction
- **Max 3 nodes per row** (use `{rank=same; A; B; C}`)
- **Max 4 rows** total
- Symmetric shape: prefer diamond, pyramid, or balanced tree

**Required graph attributes**:
```dot
bgcolor="transparent"
graph [ranksep=0.5, nodesep=0.5]
node [style=filled, fillcolor="#2c3035", color="#4a9eda",
      fontcolor="#f8f9fa", fontname="Arial", fontsize=11,
      shape=box, width=1.55, height=0.42]
edge [color="#6c757d", penwidth=1.4]
```

**Semantic colour palette** (dark-mode compatible):
| Role | fillcolor | color (border) |
|------|-----------|----------------|
| Input / source | `#375a7f` | `#4a9eda` |
| Process / neutral | `#2c3035` | `#4a9eda` |
| Output / success | `#1a4a3a` | `#00bc8c` |
| Problem / gap / failure | `#4a2040` | `#e74c3c` |

**Node labels**: 1–3 words only; no full sentences; no special characters or formulas.
**Edge labels**: ≤ 3 words if needed; use `fontcolor="#adb5bd", fontsize=9`.

**Figure options** (always include):
```
//| label: fig-<unique-name>
//| fig-cap: "Short descriptive caption"
//| fig-width: 7
//| fig-height: 2.4   (or 2.8 / 3.2 / 3.5 depending on rows)
```

---

## Updating Experiment Pages

When experiment runs complete:
1. Open the corresponding `experiments/*.qmd`
2. Fill results into the Markdown table (mean ± std)
3. Update the Claim Status Tracker in `claims.qmd`
4. Update the Key Results metrics in `index.qmd`

Never overwrite the graphviz diagrams or section structure when updating results.

---

## Adding a New Method Component

1. Copy `method/component_a.qmd` → `method/component_<name>.qmd`
2. Add the new page to the sidebar in `_quarto.yml` under the `Method` section
3. Add a row to the Components table in `method/overview.qmd`
4. Update `method/overview.qmd` architecture diagram to include the new component

---

## Adding a New Experiment

1. Create `experiments/<name>.qmd` following the structure of `experiments/main.qmd`
2. Add the new page to `_quarto.yml` under the `Experiments` section
3. Add a nav-card to `experiments/overview.qmd`
4. Link the new experiment to a claim in `claims.qmd`

---

## References

- Citation keys live in `docs/references.bib`
- CSL style: `docs/styles/ieee.csl`
- Cite in text: `[@key]` for author-year or `[-@key]` to suppress author
