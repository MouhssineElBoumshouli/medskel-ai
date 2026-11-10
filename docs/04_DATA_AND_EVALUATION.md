# 04 — Datasets, controls and scientific evaluation protocol

## Original data sources

| Source | Exact research purpose | What is known | What must be checked before use |
|---|---|---|---|
| [FIVES original article](https://doi.org/10.1038/s41597-022-01564-3) + [Figshare data](https://doi.org/10.6084/m9.figshare.19688169) | CNN supervised retinal vessel segmentation | 800 images, manual binary vessel masks, quality assessments; authors report 573 patients and recommended 600 train / 200 test | Actual archive license, official split/checksums, folder naming, duplicate/participant linkage, annotations, FOV; do not assume image independence |
| [CHASE_DB1 Kingston](https://researchinnovation.kingston.ac.uk/en/datasets/chasedb1-retinal-vessel-reference-dataset-4/) | External evaluation, two-observer variability, geometry comparisons | 28 images, both eyes of 14 children, two independent vessel annotations, CC BY 4.0 on Kingston dataset record | Acquisition differences, subject IDs and paired eyes, attribution, mask conventions, no test-train overlap |
| scikit-image `data.retina()` | One simple visual demonstration and legacy Frangi path | Sample retinal photograph, no paired label in Day-1 notebook | **Do not** train supervised retina CNN against fake or unrelated annotation |

### FIVES official folder structure (original publication)

```text
FIVES/
  train/
    Original/       # matching .png retinal photographs
    Ground Truth/   # same filenames for binary masks
  test/
    Original/
    Ground Truth/
  Quality Assessment.xlsx
```

The scientific paper says 800 images from **573 subjects**, not 800 independent people. The official release contains disease categories (`A` AMD, `D` diabetic retinopathy, `G` glaucoma, `N` normal) *to describe image origins*, **not** a mandate to train a disease classifier. Dataset includes 200 images each across categories. Class labels may serve stratification; using masks for segmentation remains the first supervised task.

**Dataset law:** Article's CC BY license may not automatically describe every underlying data file or downstream use. Record the actual Figshare data license and attribution on access, and respect any restriction attached to individual data/models. Never redistribute raw images by default.

### Data manifest schema [PROPOSED]

Create `data_manifest.csv` with at least:

```text
image_id,source,source_split,subject_id,eye,diagnosis_group,image_relpath,mask_relpath,sha256_image,sha256_mask,width,height,fov,quality_label,license_url
```

`subject_id` may be `UNKNOWN`, not invented. Include pair integrity checks: image/mask stem equality; decoded dimensions; nonempty mask; binary encoding after loading; duplicate SHA across all splits; near-duplicate/paired-eye inspection as possible; category balance and quality distribution. Keep raw manifest/data outside public repo when redistribution or privacy uncertain, but allow safe de-identified split indexes if rights permit.

## Split and leakage rules

1. FIVES **official test** must never drive training, validation, augmentation choices, architecture choice or post hoc threshold selection. Do not repeatedly check it during development.
2. Use only official `train` to make a train/val split. **Patient-level grouping is preferred and required when subject IDs are available**; verify public metadata actually allows it. If not, disclose residual subject-leakage uncertainty and avoid unsupported 'patient-independent' claims. Consider alternative dataset with explicit subject IDs or controlled label acquisition if this is a hard research gate.
3. CHASE is intended as an external set after FIVES training. If used in training, it is no longer a completely unseen external dataset; any alternative split must keep both eyes of the same child grouped.
4. Augmented patches and crops stay in the split of their parent image; no patches across train and val/test. The same raw photograph must not be in two splits due to naming, resizing or mirror copies.
5. Repeated images/eyes from one person are correlated. For confidence intervals, resample subjects when possible, not individual image files as if independent.
6. Hold out external/locked test sets and predefine evaluation code and thresholds before final runs.

## Segmentation metric definitions (binary vessel mask)

With TP true-positive vessel pixels, FP false-positive vessel pixels, FN missed vessel pixels and TN correctly rejected background:

- **Precision** = `TP/(TP+FP)`; low precision means excessive fake vessels.
- **Recall / sensitivity** = `TP/(TP+FN)`; low recall means missed vessels.
- **Dice / F1** = `2TP/(2TP+FP+FN)`; overlap, often useful for rare foreground vessels.
- **IoU / Jaccard** = `TP/(TP+FP+FN)`.
- **Specificity** = `TN/(TN+FP)` if needed; background-rich images can make it misleadingly high.

Always define FOV mask (if available/appropriate), threshold, handling of empty foreground, image size, pixel inclusion and interpolation. Compute **per-image** metrics first and summarize mean/median/distribution; don't equate pooled pixel Dice with average per-image Dice. Include illustrative false positives, tiny-vessel misses, optic disc artifacts, image-quality failure modes.

### Evaluation design

A scientifically fair study separates segmentation and skeletonization variables:

| Segmentation input | Skeletonizer | Purpose |
|---|---|---|
| FIVES/CHASE expert mask | Pixel thinning | Reference upper input quality, classical centerline |
| Same expert mask | MedSkel | Geometry difference holding segmentation fixed |
| Existing Frangi mask | Pixel thinning | Classical end-to-end pipeline |
| Same Frangi mask | MedSkel | Geometry effect on hand-crafted mask |
| U-Net predicted mask | Pixel thinning | CNN + conventional graph |
| Same U-Net mask | MedSkel | CNN + existing geometrical innovation |

This **factorial layout** prevents attributing better measurements to the CNN when only the skeletonizer changed or vice versa. Not every arm is necessary for the first class demo; make the minimal U-Net baseline work before full benchmarking. Better segmentation Dice alone does not establish better branch/width measurements.

## Measurement comparison design

- Compare methods on **identical binary masks** if asking what skeletonization changes.
- Compare segmentation methods through the **same fixed skeletonizer** if asking what CNN changes.
- Use independent reference geometry for accuracy if possible; human observer A vs B is *interobserver agreement* (not absolute truth).
- Report length, width, tortuosity, bifurcations, centerline positional agreement and **coverage**, so discarding thin branches cannot masquerade as improved quality.
- Apply fair baseline pruning/simplification, fixed parameter selection on validation, multiple epsilon values, report runtime on same hardware.
- Include per-image paired differences, confidence intervals and negative results. CHASE paired eyes require 14-child group-aware analysis.
- Report physical-unit uncertainty if imaging magnification/camera scale unavailable.

## Required experiment record

For each run store (e.g. `runs/<run_id>/` outside git if large):

```text
config.json                  # parameters, seed, code commit, split ID
manifest.csv                 # subject/image IDs and canonical split
training_history.csv         # epoch train/validation loss and Dice
best_model.pt                # checkpoint; big files kept out of Git
validation_metrics.csv       # per-image not cherry-picked
validation_predictions/      # selected documented qualitative examples
hardware_environment.txt    # OS, Python, torch, CPU/GPU, libraries
README.txt                   # commands, limitations and failures
```

Test results should include all final per-image scores, not just successful example images.

## Model selection and stopping rule

- Select best epoch by predefined validation criterion (e.g. validation Dice) and tune probability threshold using validation only.
- Do not invent target Dice thresholds or report others' published scores as ours. A minimal demo succeeds when there is *real supervised training and genuine prediction*, even if early accuracy is modest; a research success gate is stricter.
- If FIVES is not legally accessible, subject grouping is unresolvable or CPU training takes too long, state the blocker and run a safe subset proof without claiming a completed evaluation. Identify the smallest practical correction.

## Risks specific to available sources

- FIVES images are 2048×2048; 128×128 masks may lose fine vessels. Compare visual preservation before interpreting width/branch counts.
- Retinal disease categories and camera settings may cause domain shift; a model trained on FIVES may degrade on pediatric CHASE.
- Annotators disagree; training labels are not perfect biological truth.
- Different imaging magnification, field of view and crop can distort apparent calibre/density and prevent absolute biological comparisons.
- AutoMorph/VascX may have seen some public datasets in their training; confirm training provenance before labeling cross-tool performance 'fully external'.
