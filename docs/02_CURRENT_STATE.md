# 02 — Current state, implementation inventory and evidence

**Snapshot: 9 November 2026. This is not the future plan. It is the inventory of things that actually exist.**

## Verified live GitHub status

- **Original geometry repository (upstream/reference):** https://github.com/MouhssineElBoumshouli/medskel (public, `main`). **New CNN project:** `MouhssineElBoumshouli/medskel-ai` (separate repository, not yet created on GitHub at packaging time).
- Live `main` commit checked: `1a838a84d7c6b7ff5c97154c58a665950916d2cb`; its most recent commit message concerns clarifying how an earlier counting bug was found. **Re-fetch before writing**.
- Root includes `README.md`, `LICENSE`, `requirements.txt`, `medskel/`, `data/`, `docs/`, `experiments/`, `results/`, `tests/`, `figures/`, `legacy/`.
- `requirements.txt`: `numpy>=1.24`, `scipy>=1.10`, `scikit-image>=0.22`, `opencv-python>=4.8`, `networkx>=3.0`, `matplotlib>=3.7`, `pooch>=1.7`. No torch.
- Main algorithms: `medskel/polygon.py`, `medskel/voronoi.py`, `medskel/bisector.py`, `medskel/baseline.py`.
- Image handling: `medskel/data.py` (sample-data demo segmentation), `medskel/chase.py` (manual CHASE input).
- Measurement: `medskel/metrics.py` and methods on `Skeleton` class.
- Test entry point advertised in tests: `python -m pytest tests -q`. Repo documents **28 tests**. **Not re-executed as part of this package generation.**
- Experimental code: 9 experiments and their checked-in results/figures. The detailed numbers in this package were read from stored JSON, **not independently reproduced during this package generation**.
- Existing `docs/saidou-et-al-2024-a-novel-method-of-skeletonization-of-complex-shapes-based-on-bisectors.pdf` is the original method paper; read it before saying the implementation faithfully follows every theoretical step.

## Code truths that matter when extending

1. `medskel/data.py` uses sample `skimage.data.retina()`, **green channel**, adaptive histogram equalization, Frangi vesselness, hysteresis thresholding and morphological closing. It is not a general multi-image data loader; it may remove small disconnected vessels and favor a largest component.
2. `medskel/voronoi.py` operates on polygon boundaries and returns a graph (fine and branch-reduced), more than a binary skeleton image. Check coordinate order: many geometric functions use `(x,y)` while NumPy images use `(row,col)`/`(y,x)`.
3. `medskel/bisector.py` wavefront construction stops at split events; don't use it as a complete arbitrary-concave vessel-tree algorithm.
4. Existing metrics were carefully corrected after earlier counting/storage bugs; preserve fair comparison conditions. Thinning pixel staircase affects absolute lengths. `vessel_summary` uses distance transform widths and arc/chord tortuosity, **not** standardized clinical equivalents.
5. `CHASE_DB1` loader looks for official `CHASEDB1.zip` or extracted `Image_XXL/R` and `1stHO`/`2ndHO` annotations; it does not download the dataset automatically.
6. A new CNN should not depend on `skimage.data.retina()` or the original fixed Frangi mask to create training labels. Use official expert masks.

## Exact Day-1 notebook inventory

Included unchanged at `notebooks/MedSkel_AI_Day1.ipynb`. Notebook consists of **14 cells** (markdown + Python). The user downloaded this file and opened it in VS Code and personally ran the first cells.

### Observed by user in VS Code

- `PyTorch version: 2.11.0+cpu`; `Available accelerator: CPU`.
- A sample retinal image displayed successfully.
- CNN cell output:

```text
Input tensor: (1, 3, 128, 128)
Output tensor: (1, 1, 128, 128)
Trainable parameters: 29761
```

**The user has not yet reported running the synthetic backprop cell or the optional FIVES loader on their own computer.** Do not claim they did.

### Observed during notebook creation in an earlier execution environment [REPORTED]

- PyTorch `2.10.0+cpu` (different environment than owner's machine).
- U-Net input/output tensor shapes as above.
- A synthetic drawn branching line was treated as both the image data and the target mask; one Adam optimizer step using BCEWithLogitsLoss returned PASS and sample loss ~`0.7329`.
- No FIVES pairs were present there. Train-only loader found zero pairs; optional real-data training was skipped.
- **Loss 0.7329 was a synthetic software test, not an FIVES performance score.**

### Notebook architecture (code-confirmed)

```text
RGB input (3 channels)
  ConvBlock(3 -> 8)     [128x128] enc1, save a
  MaxPool(2)           [64x64]
  ConvBlock(8 -> 16)   [64x64] enc2, save b
  MaxPool(2)           [32x32]
  ConvBlock(16 -> 32)  [32x32] bridge
  Bilinear upsample    [64x64]
  Concatenate with b   [32+16=48 channels]
  ConvBlock(48 -> 16)  [64x64]
  Bilinear upsample    [128x128]
  Concatenate with a   [16+8=24 channels]
  ConvBlock(24 -> 8)   [128x128]
  1x1 Conv (8 -> 1)    [128x128 logits]
```

`ConvBlock` is two 3×3 padded convolutions and two ReLUs. Network has **29,761 trainable weights and biases**. This is a simplified U-Net suitable for learning—not the full original 2015 architecture, and not a clinically trained model. `torch.sigmoid()` is not applied inside `forward`, because the notebook's `BCEWithLogitsLoss` handles logits numerically stably.

### Notebook data interface

```text
FIVES_ROOT = Path('FIVES')
FIVES/train/Original/<name>.png
FIVES/train/Ground Truth/<same name>.png
```

The notebook pairs images by identical filename, resizes **images with bilinear** and **masks with nearest-neighbor** to `128x128`, binarizes at `>127`, and optionally trains only 2 training images for 4 passes. This is a smoke test, **not** a valid evaluation. Original FIVES comes from 573 patients, so patient-aware split needs attention; official release may not include directly actionable patient mapping.

## Existing scientific results: source-of-truth files

- [interobserver_summary.json](https://github.com/MouhssineElBoumshouli/medskel/blob/main/results/interobserver_summary.json) — original median observer-change and agreement measures.
- [robustness.json](https://github.com/MouhssineElBoumshouli/medskel/blob/main/results/robustness.json) — sign tests, leave-one-out, tolerance sensitivity. Notably length wins 26/28, bifurcation wins 18/28 (sign p~0.185), positional agreement wins 0/28.
- [08_interobserver.py](https://github.com/MouhssineElBoumshouli/medskel/blob/main/experiments/08_interobserver.py) and [09_robustness.py](https://github.com/MouhssineElBoumshouli/medskel/blob/main/experiments/09_robustness.py) — how reported metrics were generated.
- Original README explicitly says the bisector method is **~66× slower than pixel thinning** on one retina example, and says its synthetic-noise branch-count advantage does not transfer in a single-real-image experiment.

## Critical status distinction

| Component | Actual state | Never claim |
|---|---|---|
| Existing geometric MedSkel | Implemented, experiment results and tests in repo | It is a CNN; it proved clinical validity |
| PyTorch dependency | Works on user's Windows VS Code environment (CPU) | CUDA is available |
| Tiny U-Net | Defined, forward shape checked by user | Trained to identify retinal vessels |
| Synthetic backward step | Executed in earlier notebook environment; user hasn't reported their run | Real annotated training |
| FIVES raw images | Acquisition described; not confirmed present on user's computer | Downloaded, audited, patient-disjoint |
| FIVES CNN training | **Not yet confirmed / not completed** | Full model ready |
| Validation, held-out test | Not implemented or executed | Dice/IoU scores exist |
| U-Net → MedSkel integration | Planned | End-to-end metrics working |
| User interface / product | Concept | Production clinical product |

## Immediate evidence to collect after agent implementation

- `git rev-parse HEAD` + `git status --short` before and after.
- Local OS/Python/PyTorch/torchvision info, CPU/GPU and memory.
- Official FIVES dataset provenance and image/mask checksums; number of train/test pairs.
- Example paired PNGs, input/output shape assertions and label-value histogram.
- Training command and seed, loss history, checkpoint and real prediction visualization.
- Validation split manifest and per-image metrics; full final test set untouched until model selection is fixed.
- Branch name, issue, PR and actual test logs. If no data or compute available, mark progress precisely without inventing finished work.
