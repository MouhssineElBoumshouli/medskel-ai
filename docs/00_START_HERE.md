# Start here — project map

**Scope:** This GitHub repository is the new MedSkel AI CNN project, *separate from* the existing historical `medskel` geometry project.

## What a top research engineer should establish

- **Exact starting artifact:** the unchanged Day-1 notebook demonstrates the *shape and gradient plumbing* of a tiny U-Net, not retinal vessel training or validated predictions.
- **Raw evidence:** official FIVES files and annotations; original U-Net publication; original Saidou paper and the original repo code/experiments; separate original 2025/2018 papers on retinal features and cardiovascular risk (without conflating association with diagnosis).
- **Methods/tools:** Python/PyTorch, controlled train/validation/test split, mask pairing tests, deterministic preprocessing, model and checkpoint logging, held-out Dice/IoU, independent analysis of geometry, CI tests and clear user-facing overlay.
- **Independent proof:** real annotated training logs, per-image held-out metrics, tests, reproducible code and dataset manifests, and later external scientific review. An AI claim, pretty image or synthetic optimization is not enough.

## Current reality

- Historical original MedSkel source and experimental results are located in [`MouhssineElBoumshouli/medskel`](https://github.com/MouhssineElBoumshouli/medskel) — read only, no trained CNN.
- New U-Net and training code are **not yet implemented in this repository**.
- Existing Day-1 notebook is a teaching/proof-of-plumbing artifact only.
- Owner has confirmed their local VS Code kernel reports `torch 2.11.0+cpu`, shows the retina image, and produces `(1,3,128,128)` to `(1,1,128,128)` with 29,761 parameters. They are learning encoder/decoder, parameters and backprop step by step.
- No verified local FIVES archive/annotations or real training run yet. The final class presentation date remains unknown.

## Detailed reading map

- **Project mission:** [`01_MASTER_CONTEXT.md`](01_MASTER_CONTEXT.md)
- **Verified and unverified status:** [`02_CURRENT_STATE.md`](02_CURRENT_STATE.md)
- **Model and math:** [`03_CNN_IMPLEMENTATION.md`](03_CNN_IMPLEMENTATION.md)
- **Official datasets and split policies:** [`04_DATA_AND_EVALUATION.md`](04_DATA_AND_EVALUATION.md)
- **Pixel measurements and skeletons:** [`05_MEASUREMENTS.md`](05_MEASUREMENTS.md)
- **Primary research and competing tools:** [`06_RESEARCH_AND_COMPETITION.md`](06_RESEARCH_AND_COMPETITION.md)
- **Product hypotheses and boundaries:** [`07_PRODUCT_AND_RISK.md`](07_PRODUCT_AND_RISK.md)
- **Class presentation:** [`08_UNIVERSITY_PRESENTATION.md`](08_UNIVERSITY_PRESENTATION.md)
- **Implementation milestones:** [`09_TASKS_AND_MILESTONES.md`](09_TASKS_AND_MILESTONES.md)
- **Decision log:** [`10_DECISIONS_AND_OPEN_QUESTIONS.md`](10_DECISIONS_AND_OPEN_QUESTIONS.md)
- **Two-repository strategy:** [`11_REPOSITORY_BOUNDARIES.md`](11_REPOSITORY_BOUNDARIES.md)
- **Original user-supplied charter and protocol:** [`source/`](source/)

## Source of truth precedence

1. Actual source code, GitHub commits/PRs, original data, experiment logs and original papers.
2. Explicit agreed project decisions in this repository.
3. This handoff and Notion memory as context, never a substitute for verification.

If directions in inherited documents say 'modify existing medskel', that is **superseded** by the 9 November 2026 decision to build the CNN in `medskel-ai`. Never merge the two repositories without a separately approved plan. Do not silently edit the upstream code.

## First task

See [`../prompts/CLAUDE_START.md`](../prompts/CLAUDE_START.md), then implement M1 if explicitly authorized. A genuine milestone requires working data-loading tests plus actual real-data supervision, not an attractive web interface.
