# 10 — Decisions, chronology and unresolved questions

**Rules:** A suggestion in a research brief is not an approved implementation. When a new decision is made, record date, alternatives considered, evidence, owner approval and GitHub issue/PR reference. Avoid contradicting original repository findings.

## Chronology

| Date | Event / decision | Status |
|---|---|---|
| 2024 / original previous project | Saidou et al. method motivates the original skeletonization research; repo implemented Voronoi/polygon approach and a partial separate wavefront experiment. | Existing code, prior results |
| 2 Nov 2026 | CNN professor's course assignment: choose paper or dataset, code and explain mathematics; retinal vessel project selected as preferred direction based on existing MedSkel. | Confirmed course direction |
| Early Nov 2026 | Investigated measurement and possible researcher use case rather than unrelated image classifiers; FIVES + U-Net proposed. | Technical direction proposed |
| 5 Nov 2026 | Original charter, experiment protocol, resources and decision log created and uploaded to ChatGPT project. | Supporting docs supplied |
| 7–8 Nov 2026 | Live repo/Notion read-only reconciliation, original measured results and limitations; old suggestion to start with exhaustive research audit. | Read-only scientific groundwork |
| 8 Nov 2026 | Owner corrected priorities: **must first have working CNN to demonstrate for class**; long-term product research follows. | **Current priority rule** |
| 8–9 Nov 2026 | Day-1 Jupyter notebook produced with PyTorch small U-Net and synthetic gradient test; owner opened it in VS Code. | Initial artifact, no real training |
| 9 Nov 2026 | Owner confirmed PyTorch `2.11.0+cpu`, sample retina image, U-Net output shapes and 29,761 parameters in VS Code. Asked for simple explanation of encoder, decoder, weights; step-by-step teaching needed. | Confirmed observations |
| 9 Nov 2026 | Original Yusufu 2025 and Poplin 2018 papers discussed to clarify medical motivation and limits of BP/diabetes claims. | Research context |
| 9 Nov 2026 | Detailed agent handoff package generated locally. Original geometry repo not modified. | Completed artifact |
| 9 Nov 2026 | Owner requested a **separate dedicated GitHub repository for MedSkel AI** to hand to Claude. This supersedes the earlier one-repo preference; original MedSkel remains upstream/reference only. | Current repository strategy; creation/push not yet completed |

## Established strategic choices

- Retinal blood vessel segmentation using a small, understandable U-Net is the immediate university direction.
- PyTorch/Python first; CPU-only environment currently reported. Start with FIVES expert segmentation masks and honest train/validation/test separation.
- Reuse original geometry research when practical, but do not make that a prerequisite for getting a valid CNN demo.
- First audience is university professor; longer-term potential users are **researchers**, not consumers getting diabetes/BP readings.
- GitHub code and experiment runs are authoritative; Notion carries memory and decisions; this handoff is a snapshot.
- No disease diagnosis or clinical measurement claims; raw pixel units unless proven calibration.
- New `medskel-ai` repository owns CNN work and documentation; original `medskel` repository remains the verified reference for the historical geometry research. Never modify either repository `main` directly; use focused issues/branches/PRs after initialization. The handoff package creation has not modified any GitHub repository.

## Major unconfirmed assumptions

1. The owner can download the full original FIVES archive onto the local machine and has enough disk/memory to inspect it.
2. CPU training a suitable model at useful vessel resolution can complete within an acceptable duration. Need benchmark.
3. FIVES release contains enough identifier metadata to establish patient-disjoint validation; paper says 573 patients but available public IDs may not map to patients. Need verify.
4. A small CNN trained on FIVES will improve downstream measurement quality vs Frangi. **Hypothesis** only.
5. Existing MedSkel benefits once segmentation improves. Original algorithm may lose fine vessels or be too slow. **Hypothesis** only.
6. Research users have a need not satisfied by AutoMorph or VascX. **Hypothesis** only.
7. A future commercial license/product is practical. **Hypothesis** only.

## Questions for the owner or agent to resolve in order

### Must resolve for implementation

- Where are the FIVES original archive files stored (and is exact dataset download/usage permitted)? If not downloaded, obtain only after owner approval.
- What hardware does VS Code's actual running kernel see? CPU is confirmed, but benchmark memory and training throughput; GPU availability might change later.
- Confirm the new **medskel-ai** GitHub repository is created, connected to Claude, and the user approves a scoped M1 implementation in its own feature branch. The original `medskel` is read-only unless separately authorized.
- What is the next concrete class check-in/final presentation date? It was unknown at the time of handoff; do not block the initial build for the answer.

### Must resolve for scientific claims

- Is subject identity recoverable for FIVES train/test? How to disclose leakage uncertainty if not?
- What exact centerline/width ground truth (if any) is available outside observer masks?
- How to preserve resolution and thin vessels during training and inference?
- What output graph quality and runtime make the existing Voronoi method worthwhile relative to fairly pruned thinning?
- What measurement region/FOV, pixels vs physical units and QC gates define valid downstream outputs?

### Must resolve for product direction

- Which exact research workflows remain unsolved by AutoMorph/VascX?
- Will a potential research customer validate the outputs, try pilot data, or pay?
- What are exact licensing rights for datasets, models and packages in the chosen technical stack?
- Which jurisdiction and intended use would apply to any external clinical deployment?

## Source and quality caveats

- Original prior experiments are described accurately from checked-in result files, but their tests/results were not rerun during this handoff compilation.
- The Day-1 notebook was executed in an earlier environment for synthetic testing; on the owner's computer we only have self-reported setup/image/shape results, not a successful FIVES training run.
- Dates or dataset availability could change; use live primary sources and repository HEAD before implementation.
- Being thorough does not mean deferring the working model. **Ship the smallest scientifically honest CNN slice first.**
