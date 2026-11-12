# Project-wide AI agent rules

Canonical instructions for any agent are in [`CLAUDE.md`](CLAUDE.md). Follow the same standards whether the agent is Claude, Codex, or another tool.

Read `docs/00_START_HERE.md`, `docs/01_MASTER_CONTEXT.md`, `docs/02_CURRENT_STATE.md`, `docs/03_CNN_IMPLEMENTATION.md`, `docs/04_DATA_AND_EVALUATION.md`, `docs/09_TASKS_AND_MILESTONES.md`, and `docs/11_REPOSITORY_BOUNDARIES.md` before implementing.

**Priority:** train a small U-Net on real expert-labelled retinal images for university; then demonstrate and evaluate; then consider research and commercial value. No medical diagnosis claims. Never overwrite upstream `MouhssineElBoumshouli/medskel` experiments. New work belongs here (`medskel-ai`) on feature branches with issue/PR review, not on `main`. Never merge, incur costs, deploy, publish or contact third parties without specific approval.
