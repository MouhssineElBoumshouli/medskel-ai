# 09 — Actionable tasks for the AI agent (prioritized)

**This is the implementation queue. Scope only; no task in this handoff was performed on GitHub.** Each engineering task requires agreement on a branch/issue/PR workflow and must be executed in a focused scope. Do not infer that every proposed task is authorized merely because it appears here.

## M0 — Reconcile environment and repo (small, read-only)

**Goal:** know what really exists so new code doesn't break old work.

Inputs: current `medskel-ai` repository, upstream original `medskel` (read-only), `notebooks/MedSkel_AI_Day1.ipynb`, and `docs/02_CURRENT_STATE.md`.

Agent procedure:

1. Inspect this new CNN repository HEAD, working tree, existing issues/PRs/tests; separately inspect original `medskel` read-only for geometry methods and reproducibility context.
2. Check OS, Python, PyTorch version, CPU/GPU, disk/RAM and if official FIVES is installed locally. **Do not inspect or upload private files or health data without consent.**
3. Read actual notebook and original README/geometry modules, do not assume snippets in handoff override current source.
4. Propose a branch/issue scoped to minimal real-data training. Ask user to approve code modifications before remote edits.

Acceptance: short current-state note with factual status; no changed tracked files; proposed next action takes priority over lengthy audit report. Do not spend hours on academic survey before usable CNN training.

## M1 — Official FIVES train image/mask pipeline (P0)

**Problem:** current notebook has no confirmed real fundus/mask training pairs.

**Raw sources:** official FIVES Figshare archive, FIVES paper, official data layout; local dataset location supplied/approved by owner.

Implementation steps:

1. Obtain dataset by a legally permitted method. Prefer owner-driven download and place it **outside Git**; verify license and data provenance. If downloading requires network/data transfer, confirm with user first.
2. Enumerate official training pairs by exact matching names. Assert expected train/test counts against actual archive, rather than fabricating.
3. Define clean PyTorch dataset yielding `(image, mask, image_id)`, with image `[3,H,W]`, mask `[1,H,W]` `float32` labels exactly 0/1.
4. Check image/mask dimensions and alignment; resize image with suitable interpolation and mask with nearest neighbor. Show a verified original/mask overlay, document class imbalance.
5. Create split manifest and audit repeat subjects/checksums. Keep official test untouched.
6. Test loader on at least two **real FIVES training pairs** and on constructed fixtures that test mask inversion and dimensions.

Acceptance: real matched pairs loaded and visualized; correct mask fraction/shape; files under correct splits; no raw dataset committed; unit tests pass. Report if subject mapping is unavailable. Source control: small focused PR.

## M2 — Tiny real-data optimization and teachable baseline (P0)

**Goal:** prove the actual U-Net can learn *something from real annotations*, not just synthetic lines.

1. Move/refactor the tiny notebook `SmallUNet` to a documented Python module without altering original MedSkel geometry.
2. Keep shape tests for input/output logits; set seed, record parameter count, CPU-only support.
3. Run a small real-data training batch/mini overfit experiment from FIVES **training** images. Confirm finite loss, gradients and meaningful loss reduction over repeated steps.
4. Explain to owner how weights changed, what BCE loss measures, and why a tiny-batch win is not generalization.
5. Save run config, train log and a sample output; avoid giant checkpoints in git.

Acceptance: test on real masks, reproducible command, nonzero grads, finite and reduced loss; documented limitations, clean branch. If optimizer fails to learn tiny batch, debug before training all images.

## M3 — First honest train/validation run (P0/P1)

1. Create training-only validation split with grouping/metadata disclosure, deterministic preprocessing.
2. Implement train/evaluate loops: BCE baseline (optionally BCE+soft Dice after validating), small augmentation, Adam, best validation checkpoint, early stop if justified.
3. Record epochs, LR, resolution, hardware, seed, checkpoint, validation threshold (locked before test).
4. Compute **per-image** validation Dice, IoU, precision and recall; plot losses and qualitative examples.
5. Compare 128px to a higher resolution or patches when feasible, to quantify if tiny vessels disappear.
6. Create an explicit inference command `one RGB image -> probabilities -> mask -> overlay`.

Acceptance: genuine training and validation outputs saved, plots from raw logs, test set not consulted; ability to demo unseen validation predictions. No invented benchmark threshold.

## M4 — Class-ready demonstration (P1)

1. Add a small CLI/notebook or local read-only page that takes a FIVES validation image, loads checkpoint, displays input/predicted mask/overlay and computes known evaluation metrics with expert mask.
2. Add skeleton visualization via **known thinning** as a fallback, and label approximate pixel measures if shown. Existing MedSkel integration is optional if time permits.
3. Write teacher-friendly explanation for each step: conv, pool, skip, sigmoid, gradients, Dice. Have student walk through the code.
4. Build brief slides grounded in actual runs and papers; generate no fake charts, scores or patients.

Acceptance: local demo reproducible in fresh environment, one command to run, 6–8 clear slides, owner can explain encoder/decoder and what was trained. **This is the university delivery gate.**

## M5 — Integrate old MedSkel geometry properly (P2)

1. Define shared mask API and coordinate conventions. Map U-Net inference image/pixel scale accurately to geometry.
2. Run both thinning and MedSkel on identical masks. Preserve separate branches/outputs and original code.
3. Unit-test all-black, disconnected components, narrow vessels, junctions, loops, masks with holes, x/y ordering, realistic large mask runtime.
4. Report lengths/calibres in **pixels**, not micrometres; characterize resampling and image quality.
5. Add a comparison overlay: source → expert/predicted mask → centerline A/B → per-image measurements.

Acceptance: actual outputs and tests; no previous experiments broken; differences not presented as clinical superiority.

## M6 — Research-grade controlled evaluation (P2)

1. Freeze selected model, threshold and segmentation config; evaluate once on official FIVES test per protocol.
2. External CHASE test (assuming never used to train/tune), paired eyes grouped by child for inference.
3. Run factorial Frangi/U-Net/expert masks × thinning/MedSkel. Same mask for skeletonizer comparisons, same skeletonizer for segmentation comparisons.
4. Report per-image Dice/IoU, thin-vessel recall, graph topology, length/width/tortuosity, centerline agreement and coverage.
5. Reanalyze original 28 CHASE cases with subject-aware paired tests and CIs, and fair tuned/pruned thinning comparison.
6. Document failure modes, runtime, parameter sensitivity, fair incumbent comparison when legally permitted.

Acceptance: independent reproducible evidence, honest negative findings, no falsely independent 28-child statistics, no diagnosis claim.

## M7 — Research user discovery (P3)

After a functional class demo, propose a small number of **permission-based** interviews with retinal imaging researchers. Ask what existing tools they use, their actual projects, errors, file formats, QC and review time; show the prototype and solicit critique. Never send messages or upload data without approval. Test real unmet needs rather than building features from imagination.

Acceptance: documented interviews and observed task performance or negative results; competition benchmark. Product direction updated on evidence, not preference.

## First recommended Claude assignment

Use the exact starting prompt in [`prompts/CLAUDE_START.md`](../prompts/CLAUDE_START.md). It asks Claude to inspect this dedicated CNN repository and the original MedSkel geometry repository, then implement **M1: official FIVES pair loader plus real-annotation smoke test** on a branch in this new repository. **Do not modify the original `medskel` repository**. If real FIVES files are unavailable, implement and test the loader with temporary fixtures and report the data-dependent run as blocked; don't invent real-data results.

## After each task, report this brief format

```text
Task:
Status: COMPLETE / PARTIAL / BLOCKED
Source repo/branch/commit:
Source data and split:
Changes:
Exact commands run:
Observed logs/outputs:
Tests run and result:
What has NOT been proven:
Next smallest step:
Open questions / approval needed:
```
