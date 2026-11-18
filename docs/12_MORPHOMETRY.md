# 12 — Measurement study (M5): CNN mask → skeleton → measurements

This connects the CNN (versions 1 and 2) back to the original MedSkel idea:
a better vessel drawing should give better vessel *measurements*.

## The pipeline

```text
retina photo ──U-Net──> vessel mask ──skeleton──> centre lines ──> 6 measurements (pixels)
                                        │
                                        ├── thinning (standard, scikit-image)
                                        └── MedSkel  (original repo, bisectors/Voronoi, epsilon 4)
```

Measurements: vessel density, fractal dimension, total length, number of bifurcations,
median width, median tortuosity. Definitions are in `src/medskel_ai/morphometry.py`.
All in pixels: there is no validated pixel-to-millimetre scale.

## The three questions (fixed before running)

1. **CHASE_DB1 (two humans per image).** If the CNN mask replaces a human mask, how
   much does each measurement move? Compare with how much it moves between the two
   humans. Does MedSkel move less than thinning? This repeats the original MedSkel
   inter-observer experiment (3.2% vs 9.3% for total length) and adds the CNN.
2. **FIVES test.** The same question against the FIVES expert mask (200 images).
3. **FIVES disease groups.** Do the measurements differ between normal, AMD, diabetic
   retinopathy and glaucoma eyes? Can a simple logistic regression guess the group
   from the 6 numbers? This **repeats a known kind of result** (e.g. Yusufu et al.
   2025, UK Biobank). It is an association in one dataset. It is not a diagnosis.

## Honest notes

- **Bug found in the original MedSkel width.** In `medskel/voronoi.py`
  (`_reduce_to_branches`), each branch keeps its radii only at the vertices that
  survive polyline simplification. On straight vessels those are mostly the tips,
  so the width is far too small (a 12 px test vessel gives 1.3 px). The original
  repository is not edited. Here, width is measured the same way for both methods
  (distance transform sampled along the centre line). The original repository's
  *width* results (not its length results) should be read with this in mind.
- **The FIVES test set is used again, but nothing is chosen on it.** The CNN, its
  threshold, the skeleton settings and the measures are all fixed before this step.
  The test images are only used to report results, and the classifier is trained on
  training images only.
- Left out of question 3: the two empty training masks (447_G, 448_G) and train/174_D
  (its expert mask does not match the photo). Identical photographs count once.
- CHASE: 28 images from 14 children; intervals resample children, not images.
- FIVES has no patient ids, so some patients may appear in both train and test.

## How to run (on the training laptop)

See the README section "Measurement study". Outputs go to `results/morphometry/`:
`measurements.csv` (one row per mask and method), `analysis.json`, `summary.md`,
`chase_agreement.png`, `groups_expert.png`. CNN masks are saved (git-ignored) in
`runs/morphometry/masks/`.
