# CLAUDE.md — Project rule router

This repository uses a layered Claude rule system. Always obey rule priority:

1) .claude/global.md
2) .claude/architecture.md
3) module-level rules: .claude/data.md, .claude/methods.md, .claude/inference.md, .claude/train.md, .claude/configs.md

When editing code:
- If you modify files under src/data → read .claude/data.md
- If you modify files under src/methods → read .claude/methods.md
- If you modify training loop / trainer usage → read .claude/train.md
- If you change configs/** → read .claude/configs.md

When editing docs:
- read .claude/docs.md

For long-running experiments:
- Before generating or changing experiment code, provide a best-effort runtime estimate.
- State the assumptions used for the estimate (dataset size, epochs, batch size, device, number of steps).
- If exact runtime is unknown, give an order-of-magnitude estimate and identify the missing information.

For experiment notifications:
- Do not assume Claude can send email directly.
- If requested, generate or update a wrapper script that sends email on normal exit or failure.
- Prefer a shell or Python wrapper in scripts/ or tools/ with email credentials supplied via environment variables.

Do not infer architecture from single example files.
If rules conflict, follow .claude/global.md.

Before modifying files, suggest running:
git pull.