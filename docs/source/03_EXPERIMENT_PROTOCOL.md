# MedSkel AI — Experimental protocol v0 (must be frozen before running final experiments)

## University CNN research question
Does a simple U-Net vessel segmentation improve *generalization and the reliability of downstream vessel measurements* compared with the existing Frangi-based segmentation, when paired with the same skeletonization method?

Secondary: Does MedSkel's skeletonization yield a useful reproducibility/measurement benefit compared with thinning on masks of equal quality? Separate this from the impact of the CNN.

## Design: avoid conflating two algorithms
A factorial comparison, where feasible:
- Input segmentation: expert annotation / Frangi baseline / new U-Net / an established publicly available model (subject to license).
- Skeletonization: pixel thinning / existing MedSkel.
Run both skeletonizers on the **same** mask, and run each segmentation method through a fixed skeletonizer. This isolates effects instead of attributing every change to U-Net or MedSkel indiscriminately.

## Raw data to inspect
- FIVES: 800 fundus images + pixel-level vessel masks + quality labels; canonical 600/200 train/test split, verify participants and duplicate/subject IDs before additional splitting.
- CHASE_DB1: 28 images from 14 children, 2 eyes each and two independent observer masks; no overlap of subject pairs between internal train/validation/test if used for training; preferably held out as a distinct external dataset after FIVES development.
- Retain origin, copyright/license/attribution, resolution, camera/fov metadata when provided, annotation rules, checksums and case IDs. Never commit raw clinical or large training data into the public repo.

## Model (first version)
- PyTorch, small 2D U-Net encoder/decoder with skip connections.
- Explicitly derive image tensors, convolution output dimensions, receptive field intuition, ReLU, max pooling, interpolation/transposed convolution, logits and sigmoid.
- Pixelwise BCEWithLogits + soft Dice loss; explain why vessel pixels are scarce and test an alternative if justified.
- On modest hardware use deterministic resize/patch sampling and limited capacity; preserve the original field of view and test thin-vessel loss.
- Training-only augmentation: rotations/flips and modest photometric variation, applied consistently to image and mask for geometric operations.
- Log seed, optimizer, LR, epochs, checkpoint selection criterion, loss and per-image validation; save exact config and code SHA.

## Segmentation test metrics
Per-image Dice, IoU, precision, recall; report median/mean, distributions, confidence intervals and qualitative failures. Evaluate within field-of-view masks, use pre-declared thresholds selected only on validation. Assess thin-vessel recall and vessel connectivity/topology if feasible.

## Downstream geometry tests
- Existing MedSkel total vessel length, bifurcations, median pixel calibre, simple arc/chord tortuosity and skeleton positional agreement.
- Distinguish accuracy to an *independent* annotated vessel geometry from agreement between two observers.
- Check connectivity loss and graph fragmentation (existing implementation can fragment).
- Do not label existing median width as CRAE/CRVE/AVR.
- If repeated scans across cameras become available, use ICC/Bland–Altman and stratify by device, after specifying study conditions; CHASE two observers cannot substitute for this.

## Split discipline / uncertainty
- Never split patches from an image across train and test; all patches inherit the parent image split.
- Group related images by subject where metadata permits. Examine the original FIVES split for repeated subjects; do not claim patient independence when unverified.
- Keep official FIVES test set untouched until final evaluation; validate on a held-out subset of training data.
- CHASE subject IDs are the 14 original children; preserve both eyes together.
- Pre-register final evaluation and compare paired per-image differences, confidence intervals (cluster by subject when appropriate) and individual failure cases.

## Milestones and pass/fail gates
1. Audit original repo + run reproducible tests without modifying `main`.
2. Build dataset manifest, legally acquired images, train/val/test integrity checks and initial manual samples.
3. Implement a small U-Net and deliberately overfit a tiny training batch to verify training code, then train a clean baseline.
4. Freeze checkpoint and threshold by validation; run locked FIVES and CHASE evaluations.
5. Factorial comparison and error audit; report if CNN changes masks but not measurements, or if MedSkel loses.
6. External review with researcher to identify whether the measurements matter and whether the failure modes are acceptable.

## Independent evidence of success
Original labels/raw cases, reproducible scripts and config, participant-level holdouts, external datasets, measured comparison with credible baselines, and a researcher's blinded assessment. A high training score or visually attractive overlay is not enough.
