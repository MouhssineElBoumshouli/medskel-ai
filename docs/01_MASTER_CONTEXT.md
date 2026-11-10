# 01 — MedSkel AI: project master context

**As of 9 November 2026. Distinguishes established results from goals.**

## 1. Why this project exists

The owner has a university course in convolutional neural networks. The professor expects a project grounded in a research paper or dataset, implemented and understood mathematically, with a scientific presentation. **There is no confirmed final presentation date.** On 9 November the owner said the professor needed evidence that the work had at least begun; the Day-1 notebook provides that honest start.

The owner already built [`medskel`](https://github.com/MouhssineElBoumshouli/medskel), implementing and testing a geometric skeletonization approach inspired by Saidou et al. (2024). They want to **extend**, not abandon, that prior effort. Long term they want to learn whether retinal vessel measurements can be made accurate, reproducible and useful to researchers, and whether that becomes a viable product. But **working CNN first, research/product depth next** is the agreed priority correction.

### Working project title

**MedSkel AI — Retinal Blood Vessel Segmentation and Morphometry Using U-Net.**

### Immediate question for class

> Can a small U-Net learn to segment retinal blood vessels from annotated fundus photographs, and how well do its predictions match held-out expert masks?

### Follow-on research question (not a class gate)

> Does replacing traditional Frangi-based segmentation with CNN segmentation improve downstream vessel measurements, and can MedSkel's geometry offer a reproducibility advantage over pixel thinning without discarding meaningful vessel structure?

### Product question (hypothesis)

> Do retinal imaging labs have a measurable unmet need for trustworthy, auditable, batch vessel measurements with transparent failure inspection, beyond the tools they already use?

## 2. Target pipeline and boundaries

```text
[Public de-identified color fundus image + metadata]
              │
              v
[Image integrity / quality / field-of-view assessment]      <- later research feature
              │
              v
[U-Net vessel segmentation: probability map / binary mask] <- immediate CNN goal
              │
              v
[Comparable centerline methods: thinning | MedSkel]       <- existing geometry + integration
              │
              v
[Pixel-space vessel graph and measurement table]           <- initial study outputs
              │
              v
[Annotated overlay, measurement warnings, export CSV/JSON] <- usable research demo
```

This is a research analysis pipeline, **not** a scanner, diagnostic system or blood-pressure device. An uploaded photograph can support experimental measures; it does not supply actual clinical blood pressure or diabetes status.

### First in-scope software features

- Upload/select public retinal images and load with correct RGB dimensions.
- Display image, expert vessel mask (when available), predicted vessel probability map, binary segmentation and overlay.
- Train an understandable U-Net with documented train/validation/test protocol.
- Compute Dice, IoU, precision and recall on held-out annotated images.
- Extract centerline with a simple trusted baseline (scikit-image thinning); integrate MedSkel's Voronoi-based approach when stable and fair.
- Show candidate length (px), approximate calibre/width (px), branches, bifurcations and tortuosity with explicit definition/limitations.
- Save model/checkpoint metadata, per-image measurements and provenance.

### Explicitly out-of-scope at first

- Disease classification/diabetes detection or blood-pressure inference.
- CRAE, CRVE, AVR without trained artery/vein labels and validated optic-disc-centered protocol.
- Physical micrometre estimates from plain pixel measurements without reliable calibration.
- Clinical application, patient-report generation, interpretation or regulatory claims.
- Expensive GPUs, image acquisition hardware, enterprise UI, automated patient-data processing.

## 3. Who could use it later? [HYPOTHESIS]

Initial potential users: retinal imaging researchers, ophthalmology researchers, clinical-study quantitative imaging groups, academic labs. Their candidate needs include consistent masks, transparent QC, per-vessel measurements, high-volume batch analysis, reviewer correction and traceable outputs. These are **not confirmed customer needs**; interviews and side-by-side tests are required.

Do not confuse 'retinal features correlate with blood pressure or HbA1c in published population studies' with 'our system can predict those values accurately or make clinical decisions'. See `06_RESEARCH_AND_COMPETITION.md`.

## 4. Existing method: geometric MedSkel [VERIFIED CODE / REPORTED RESULT]

Original Python package uses scikit-image sample images, Frangi vesselness + thresholding to get rough masks, then a boundary-to-polygon and dense-boundary Voronoi pipeline with separation-angle filtering and graph reduction. It also implements a separate wavefront/straight-skeleton demonstration. The Voronoi/medial-axis and wavefront/straight-skeleton definitions are not interchangeable for every concave shape.

- [`medskel/data.py`](https://github.com/MouhssineElBoumshouli/medskel/blob/main/medskel/data.py) — demonstrative preprocessing; **not a robust learned or clinically evaluated segmenter**.
- [`medskel/polygon.py`](https://github.com/MouhssineElBoumshouli/medskel/blob/main/medskel/polygon.py) — mask boundaries and polygon simplification.
- [`medskel/voronoi.py`](https://github.com/MouhssineElBoumshouli/medskel/blob/main/medskel/voronoi.py) — geometric centerline graph and per-branch measurements.
- [`medskel/bisector.py`](https://github.com/MouhssineElBoumshouli/medskel/blob/main/medskel/bisector.py) — independent partial wavefront/straight-skeleton check; split-event handling is **not implemented**.
- [`medskel/baseline.py`](https://github.com/MouhssineElBoumshouli/medskel/blob/main/medskel/baseline.py) — pixel skeletonization/branch extraction.
- [`medskel/metrics.py`](https://github.com/MouhssineElBoumshouli/medskel/blob/main/medskel/metrics.py) — length, width, tortuosity, position/reproducibility and comparison metrics.
- [`medskel/chase.py`](https://github.com/MouhssineElBoumshouli/medskel/blob/main/medskel/chase.py) — official CHASE dataset masks loader, manual download.
- `experiments/01_pipeline.py` through `experiments/09_robustness.py`; `tests/test_medskel.py`, `tests/test_measurements.py`, `results/interobserver_summary.json`, `results/robustness.json`.

The original repo has no PyTorch dependency in `requirements.txt`. **Do not call existing medskel a CNN.**

### Existing reported evidence from CHASE_DB1

Original study compared 28 pairs of two-observer masks (two eyes for each of 14 children) processed through the same geometric method. Saved results show:

| Metric | Pixel thinning | MedSkel bisector ε=4 | Honest interpretation |
|---|---:|---:|---|
| Median relative change in total length between observers | 0.09287 | 0.03215 | MedSkel more stable for this global measure, **not necessarily more accurate**. |
| Median relative bifurcation-count change | 0.21648 | 0.12222 | Wilcoxon p~0.00266, but only 18/28 cases favor it; sign test p~0.185; qualify. |
| Median relative width change | 0.17462 | 0.21372 | No clear improvement (Wilcoxon p~0.227). |
| Median relative tortuosity change | 0.00364 | 0.00654 | Thinning slightly more stable; tiny changes. |
| Fraction with cross-observer centerline agreement at 2px | 0.80915 | 0.71627 | **Thinning better**. |
| Agreement on thick/common vessels | 0.91639 | 0.79078 | **Thinning better**. |

For total-length stability the bisector favors 26/28 images. Positional-agreement direction favors thinning 28/28. Single-retina example in README: ~0.66s MedSkel vs ~0.0099s thinning (~66× slower). Phantom tests show comparable centerline error on clean shapes. High-noise phantom benefit does **not** clearly transfer to real vessel-mask branch counting in experiment 07.

**Methodological warning:** the 28 images are not 28 independent children. Prior per-image Wilcoxon p-values need subject-grouped sensitivity analysis. A stable but over-simplified vessel graph can appear more reproducible precisely because it lost thin vessels. Any claim about improved measurements must test retained coverage, bias and agreement with an independent reference.

## 5. New CNN concept [PLANNED]

Use PyTorch and a **small understandable U-Net**. It takes a color RGB photograph `(N,3,H,W)`, produces a single vessel *logit* for each pixel `(N,1,H,W)`, then applies `sigmoid` to obtain vessel probabilities. Learn from real fundus/mask pairs in FIVES. Binary classification at every pixel is a **segmentation** task, not generic image-level classification.

An existing local `MedSkel_AI_Day1.ipynb` proves basic PyTorch installation, sample image visualization, network shape check and a gradient step on **synthetic** data only. No tested retinal trained model yet. `02_CURRENT_STATE.md` records the exact shape and limitations.

Use the train-only loader as an initial guide, but implement a clean validated dataset class before claiming generalization. CPU-only PyTorch 2.11.0 was reported by the owner in VS Code; avoid assuming CUDA hardware. Reduced resolution is okay for proving training but may erase tiny vessels. Preserve an upgrade path to patches/full-resolution inference.

## 6. Source-of-truth priority

1. Current verified source code, experiment outputs, tests and git history on GitHub.
2. Original peer-reviewed papers and original datasets plus licensing/label definitions.
3. Raw logs/checkpoints/data splits and reproducible experiment specifications.
4. Existing Notion history and the four uploaded project documents.
5. This handoff snapshot and conversation summaries, which can go stale.

When conflicting, identify the conflict, check the higher-quality source, and record the reason for the resolution. Do not silently rewrite history.

## 7. How success will be assessed

### University minimum
- Clearly explained model grounded in U-Net 2015 and FIVES 2022.
- Actual real-data loader and successful training; no synthetic masks substituted for FIVES.
- Reproducible proof of learned weights, loss curve and stable checkpoint.
- Side-by-side images and computed holdout Dice/IoU (not invented; no test leakage).
- Live demo or recorded local run with original image → prediction → optional centerline/measurements.
- Student understands and can defend every major mathematical operation.

### Research standard (later)
- Independent CHASE evaluation respecting subject clustering.
- Paired comparisons on identical masks; strong thinner-vessel and topology error review.
- Calibrated physical/biological interpretation only where justified by metadata.
- Baseline AutoMorph/VascX comparison and independent expert review.
- Reproducible scripts, manifest, locked evaluation, confidence intervals, failure cases.

### Product standard (much later)
- Documented unmet researcher workflow need, measurable time/quality improvement, approved rights to software/datasets/models and practical path through legal and privacy requirements.

## 8. What to do now

Run the concrete checklist in `09_TASKS_AND_MILESTONES.md`. The **first implementation must be very small**: open official FIVES *train* pairs, validate masks, show a sample, run an optimizer step and save a non-misleading result. Obtain permission before editing the remote repository. Then finish a real training/validation loop and clear presentation demo. Do not wait for comprehensive product research to produce a CNN result.
