# MedSkel AI — Decisions, hypotheses and open loops
Updated: 5 November 2026

## Confirmed from user
- A CNN course project must be grounded in a dataset or original research paper, coded, and mathematically explained.
- Preferred direction: retinal blood vessel analysis based on existing MedSkel rather than road crack detection.
- Goal is genuine long-term research/product potential, not university-only work.
- Codex is available for implementation.

## Evidence-based technical facts
- Existing public MedSkel repository: geometry-based skeletonization, image processing and metrics; not CNN training.
- Its CHASE_DB1 two-observer median relative total-length change 0.032 (bisector ε=4) vs 0.093 (thinning); centerline agreement 0.716 vs 0.809 respectively; no independent clinical measurement accuracy established.
- Current research software competitors already do much of the intended pipeline; especially AutoMorph and VascX.

## Proposed (not yet approved or independently proven)
- Research-use software positioning focused on auditable, reproducible retinal vessel morphometry.
- Initial machine-learning baseline small U-Net using public annotated fundus images.
- University question: effect of vessel segmentation method on downstream geometry measurement.
- First customers: retinal research labs, imaging researchers, academic clinical-study teams.

## Explicitly unresolved
- Which exact customer workflow and measurements are insufficient in current tools?
- Does MedSkel offer measurable advantage over AutoMorph/VascX for real images, quality variation, or repeat scans?
- How to structure new product code vs existing shipped research repository? No change made yet.
- Availability of hardware, medical collaborators, repeat-camera images and physical resolution metadata.
- What exact rights would third-party model/code/dataset licenses permit for eventual commercial application?
- What clinical/regulatory classification would a later intended use require?

## Decisions not yet taken
- No new GitHub repo, Codex branch, code changes, data processing, payment, outreach, or clinical product claim is authorized merely by this document.

## Change protocol
For every new technical/product decision: date, proposed change, evidence, alternatives rejected, risk, human approval if needed, and authoritative GitHub issue/PR link.
