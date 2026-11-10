# Repository architecture and boundaries

## Decision on 9 November 2026

The user explicitly requested **a dedicated GitHub repository for MedSkel AI** that Claude can inspect, distinct from the original `medskel` repository. This supersedes the previously documented preference for a single repository.

### Repository responsibilities

| | Original `MouhssineElBoumshouli/medskel` | New `MouhssineElBoumshouli/medskel-ai` |
|---|---|---|
| Purpose | Preserve and evolve original bisector/polygon/Voronoi research | Build CNN segmentation, class demo, reproducible experiments, later validated integration |
| Existing code | Classical image processing, geometric skeletonizers, tests, CHASE experiments | Day-1 U-Net notebook and documentation only at handoff creation |
| Existing evidence | Measured inter-observer reproducibility and centerline agreement tradeoff | No trained retinal U-Net or test metrics |
| Changes by Claude | **Read-only unless separately authorized** | Dedicated branch + PR under owner-approved scope |
| Releases/data | Original historical sources and existing result files | Code and metadata only. FIVES/CHASE source images, checkpoints and private medical data stay outside Git |

### How to integrate later

Once trained U-Net and per-image segmentation evaluation work, run pixel-thinning as the first centerline baseline. Then consider one of:

1. A pinned dependency on a released version of the original geometry code (only if packaging and licenses permit).
2. A reproducible integration adapter invoking the original package in a controlled environment.
3. Carefully extracted shared geometry package **only if independently tested against original outputs and approved**.

Do **not** blindly copy old modules or claim the old experiments ran under the new implementation. Document pinned SHA, mask-coordinate conventions, scale changes, runtime and side-by-side regression tests.

### Git rules

- Initialize the new repository once with its documentation on `main`; this is repository bootstrap, not feature work.
- For **all further changes**, open a scoped issue and implement on a feature branch; request review via PR; do not merge without approval.
- Never rewrite the original repository's Git history or results. Do not assume that code in upstream must be moved merely because the new project has a similar name.

### What Claude should work on first

`M1 — FIVES train-only image/mask loader + small PyTorch U-Net + real training smoke test`, with tests and a simple notebook/CLI to show image and expert mask. Official test images stay sealed. See [`../prompts/CLAUDE_START.md`](../prompts/CLAUDE_START.md).
