## What this is

A small **U-Net** (7.85 million parameters, trained from scratch, no pretrained weights) that marks blood vessels in colour photographs of the retina. It was trained for 150 minutes on a laptop GPU on 510 images of the public **FIVES** dataset. University CNN class project.

**Research prototype. Not a medical device. It does not diagnose anything.**

## How it works

1. The photograph is cut into overlapping 512 x 512 tiles.
2. The U-Net gives every pixel a vessel probability.
3. Tiles are stitched; overlapping areas are averaged. Optionally the photo is also predicted mirrored (4 passes) and the results are averaged.
4. Pixels with probability at or above the threshold are called vessel. The default threshold was chosen on validation images.

## Real results

All numbers were computed by us. The value in brackets is a 95% bootstrap confidence interval.

| Test | Images | Dice | clDice (connectivity) | Sensitivity | Specificity |
|---|---|---|---|---|---|
| FIVES official test set | 200 | 0.889 (0.873 – 0.903) | 0.897 (0.881 – 0.911) | 0.874 | 0.994 |
| CHASE_DB1, external, vs human observer 1 | 28 (14 children) | 0.766 (0.750 – 0.782) | 0.835 (0.825 – 0.845) | 0.908 | 0.949 |
| For comparison: human observer 2 vs observer 1 on CHASE_DB1 | 28 | 0.777 (0.766 – 0.788) | 0.835 (0.822 – 0.847) | 0.769 | 0.978 |

PRETRAINED_ABOUT

## Known weaknesses

- **Thin vessels:** only 46% of thin-vessel pixels (1 to 4 pixels wide) are found on the FIVES test set, against 88% for thicker vessels.
- **Other cameras:** quality drops on images unlike FIVES (Dice 0.89 → 0.77 on CHASE_DB1), and the model marks too many pixels as vessel there.
- **Dark or blurred photographs** can give an almost empty result.
- FIVES has 800 images from 573 patients without patient identifiers, so FIVES scores may be optimistic. The FIVES test set was used twice during the project.

## Sources

- Code and full report: https://github.com/MouhssineElBoumshouli/medskel-ai
- FIVES dataset: Jin et al., *Scientific Data* 9, 475 (2022), https://doi.org/10.1038/s41597-022-01564-3 (CC BY 4.0)
- CHASE_DB1: Kingston University, https://researchinnovation.kingston.ac.uk/en/datasets/chasedb1-retinal-vessel-reference-dataset-4/ (CC BY 4.0)
- U-Net: Ronneberger et al., 2015. clDice: Shit et al., 2021.
