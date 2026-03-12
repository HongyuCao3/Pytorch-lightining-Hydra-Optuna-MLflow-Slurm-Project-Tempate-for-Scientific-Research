# CLAUDE.md — Project rule router

This repository uses a layered Claude rule system.  Always obey rule priority:

1) .claude/global.md
2) .claude/architecture.md
3) module-level rules: .claude/data.md, .claude/methods.md, .claude/inference.md, .claude/train.md, .claude/configs.md

When editing code:
- If you modify files under src/data → read .claude/data.md
- If you modify files under src/methods → read .claude/methods.md
- If you modify training loop / trainer usage → read .claude/train.md
- If you change configs/** → read .claude/configs.md

Do not infer architecture from single example files.
If rules conflict, follow .claude/global.md.