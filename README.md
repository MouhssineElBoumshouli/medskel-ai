# MedSkel AI

**Research-first retinal blood vessel segmentation and measurement.** The immediate deliverable is a small, understandable **PyTorch U-Net** for a university CNN project. The long-term goal is to test whether retinal vessel geometry can be measured accurately and reproducibly enough for research workflows.

> **Current status (9 November 2026): starter notebook only — no trained FIVES model, no clinical validation.**

## What this repository is

This is the **dedicated MedSkel AI development repository**. It holds the new CNN code as it is implemented, experiments, documentation, and the eventual class demonstration. It is **not** the original geometry repository and must not overwrite its findings.

| Repo | Purpose | Rules |
| --- | --- | --- |
| **This repository: `medskel-ai`** | New CNN, FIVES pipeline, segmentation evaluation, demo, future carefully tested integration | Feature branches + PR review; new results must be reproducible |
| [Original `medskel`](https://github.com/MouhssineElBoumshouli/medskel) | Previous polygon/Voronoi-based skeletonization, CHASE results, tests | Reference/upstream. Read-only unless separately approved |

## What the project will demonstrate

```text
Color retinal image
       ↓
Trained U-Net (predicts vessel probabilities)
       ↓
Binary vessel segmentation + overlay
       ↓
Pixel thinning baseline (initial centerlines)
       ↓
Research-only vessel length / width / branch summaries (pixel units)
       ↓
Later: controlled comparison against original MedSkel geometry
```

**Do not confuse:** a synthetic gradient check, a model trained on real annotated fundus images, a validated segmentation model, reliable physical vessel measurements, and a clinical diagnostic tool. These are different milestones.

## What is available now

- [`notebooks/MedSkel_AI_Day1.ipynb`](notebooks/MedSkel_AI_Day1.ipynb): genuine retinal example image, small U-Net network, shape checks, synthetic backprop test, optional unexecuted FIVES training section. The owner has reported PyTorch `2.11.0+cpu` in VS Code and has successfully run the sample retina and model-shape sections.
- [`CLAUDE.md`](CLAUDE.md): how Claude should work here.
- [`docs/00_START_HERE.md`](docs/00_START_HERE.md): project guide and source hierarchy.
- [`docs/01_MASTER_CONTEXT.md`](docs/01_MASTER_CONTEXT.md): detailed scientific/product context.
- [`docs/09_TASKS_AND_MILESTONES.md`](docs/09_TASKS_AND_MILESTONES.md): executable priorities and acceptance tests.
- [`prompts/CLAUDE_START.md`](prompts/CLAUDE_START.md): first prompt for Claude.

**There is currently no trained checkpoint, confirmed FIVES download, measured Dice score, or production inference service.** Code will be added in focused pull requests, not promised in documentation.

## Read the original research

- Original MedSkel geometry paper, experiments, and tests: [original GitHub repository](https://github.com/MouhssineElBoumshouli/medskel)
- U-Net, Ronneberger et al. (2015): https://arxiv.org/abs/1505.04597
- FIVES dataset, original peer-reviewed paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC9352679/
- CHASE_DB1, Kingston University: https://researchinnovation.kingston.ac.uk/en/datasets/chasedb1-retinal-vessel-reference-dataset-4/
- Relevant established alternatives: AutoMorph, VascX (see [`docs/06_RESEARCH_AND_COMPETITION.md`](docs/06_RESEARCH_AND_COMPETITION.md))

## First engineering milestone

**M1:** Load actual expert-annotated FIVES training image/mask pairs, verify pixel alignment, run a tiny real-data training step, and show a real image + expert mask. Use official FIVES `train` data only. The official test set remains untouched. The code must be testable with synthetic fixtures *without pretending fixture training is training on FIVES*.

Do not push datasets, patient data, secrets, large checkpoints, downloaded models, or unverified performance numbers to this repository. See [project boundaries](docs/11_REPOSITORY_BOUNDARIES.md) and [data protocol](docs/04_DATA_AND_EVALUATION.md).

## For AI agents

Start with [`CLAUDE.md`](CLAUDE.md) and [`docs/00_START_HERE.md`](docs/00_START_HERE.md). Inspect actual code and paper sources before choosing an approach. Keep explanations simple enough that the student can defend the CNN architecture and mathematics to the professor.

**Status is not success:** The current repository contents document a plan and a starter notebook. Successful M1 requires actual code, tests, and honest evidence.
