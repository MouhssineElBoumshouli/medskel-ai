# MedSkel AI

**Research-first retinal blood vessel segmentation and measurement.** The immediate deliverable is a small, understandable **PyTorch U-Net** for a university CNN project. The long-term goal is to test whether retinal vessel geometry can be measured accurately and reproducibly enough for research workflows.

> **Current status (November 2026): a U-Net trained on real FIVES expert masks, validated with connectivity and thin-vessel metrics, tested externally on CHASE_DB1 with a human-agreement comparison, with a local demo.** Research prototype for a class project. No clinical validation. Not a medical device.

## Results

Our model: a plain U-Net, 7,849,601 parameters, trained from scratch for 150 minutes on an RTX 3050 (4 GB) on 510 FIVES training images (random 512 x 512 crops at full resolution). Predictions are averaged over 4 mirrored passes. Threshold 0.30 chosen on 88 validation images. Brackets: 95% bootstrap interval.

| Test | Dice | clDice | Sensitivity | Specificity |
| --- | --- | --- | --- | --- |
| FIVES official test set, 200 images | **0.889** (0.873 – 0.903) | 0.897 | 0.874 | 0.994 |
| CHASE_DB1 (external, 28 images, 14 children) vs observer 1 | 0.766 (0.750 – 0.782) | 0.835 | 0.908 | 0.949 |
| Human observer 2 vs observer 1 on CHASE_DB1 | 0.777 (0.766 – 0.788) | 0.835 | 0.769 | 0.978 |

- Version 1 baseline (pull request #4, 50 minutes of training): FIVES test Dice 0.875.
- A reference U-Net with an ImageNet-pretrained ResNet34 encoder, same data and time, is better on FIVES (Dice 0.907) and equal on CHASE_DB1 (0.770).
- **Main weakness:** only 46% of thin-vessel pixels (1 to 4 px wide) are found.
- Limits: FIVES has 800 images from 573 patients and no patient ids; the FIVES test set was used twice; one training run per setting; the model marks too many vessel pixels on CHASE_DB1; dark, blurred photos fail.

Full details in [`report.md`](report.md), raw numbers in [`results/`](results/) and [`results/v2/`](results/v2/).

## Quick start (Windows, PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -e ".[dev,demo]"
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"   # must print True
```

Put the FIVES dataset **outside this repository**, at `C:\data\FIVES` (or pass `--data <folder>`):

```text
C:\data\FIVES\train\Original\1_A.png ...        600 photographs
C:\data\FIVES\train\Ground truth\1_A.png ...    600 expert masks (same file names)
C:\data\FIVES\test\Original\...                 200 photographs
C:\data\FIVES\test\Ground truth\...             200 expert masks
```

| Step | Command |
| --- | --- |
| Run the tests (synthetic fixtures, no FIVES needed) | `.\.venv\Scripts\python.exe -m pytest` |
| Print model shapes and parameter count | `.\.venv\Scripts\python.exe -m medskel_ai.model` |
| Check the data files, write the split | `.\.venv\Scripts\python.exe -m medskel_ai.data` |
| Train the final model (150 minutes on an RTX 3050 4 GB) | `.\.venv\Scripts\python.exe -m medskel_ai.train --minutes 150 --out runs/e2_long` |
| Choose the threshold on validation images | `.\.venv\Scripts\python.exe -m medskel_ai.evaluate --run runs/e2_long --split val --flips` |
| Test evaluation (**once** per model) | `.\.venv\Scripts\python.exe -m medskel_ai.evaluate --run runs/e2_long --split test --flips` |
| External test on CHASE_DB1 (put it in `C:\data\CHASEDB1`) | `.\.venv\Scripts\python.exe -m medskel_ai.chase --run runs/e2_long --flips` |
| Segment one photograph | `.\.venv\Scripts\python.exe -m medskel_ai.predict --image photo.png --checkpoint runs/e2_long/best.pt --flips` |
| **Launch the demo** | `.\.venv\Scripts\python.exe app.py` then open http://127.0.0.1:7860 |

Training output (checkpoint, logs, example figures) goes to `runs/unet_fives/`, which Git ignores. Small result files (numbers only) are copied to [`results/`](results/).

## Where to read

| File | What it is |
| --- | --- |
| [`docs/explanation.md`](docs/explanation.md) | The CNN and its mathematics in simple English, with worked examples and 10 likely exam questions |
| [`report.md`](report.md) | Short project report with the real numbers |
| [`demo.ipynb`](demo.ipynb) | Notebook walk-through (see the note in the report about image outputs) |
| [`src/medskel_ai/model.py`](src/medskel_ai/model.py) | The U-Net, with the full tensor-shape table |
| [`src/medskel_ai/data.py`](src/medskel_ai/data.py) | File pairing, split, crops, augmentation, data audit |
| [`src/medskel_ai/train.py`](src/medskel_ai/train.py) | Loss, training loop, validation |
| [`src/medskel_ai/evaluate.py`](src/medskel_ai/evaluate.py) | Metrics, threshold choice, test evaluation |
| [`src/medskel_ai/predict.py`](src/medskel_ai/predict.py) | Whole-image prediction by tiling and stitching, optional flips |
| [`src/medskel_ai/topology.py`](src/medskel_ai/topology.py) | clDice and thin-vessel recall |
| [`src/medskel_ai/chase.py`](src/medskel_ai/chase.py) | External test on CHASE_DB1, model vs human observers |
| [`PROGRESS.md`](PROGRESS.md) | Step list so a new session can resume |

## What this repository is

This is the **dedicated MedSkel AI development repository**. It holds the new CNN code, experiments, documentation, and the class demonstration. It is **not** the original geometry repository and must not overwrite its findings.

| Repo | Purpose | Rules |
| --- | --- | --- |
| **This repository: `medskel-ai`** | New CNN, FIVES pipeline, segmentation evaluation, demo, future carefully tested integration | Feature branches + PR review; new results must be reproducible |
| [Original `medskel`](https://github.com/MouhssineElBoumshouli/medskel) | Previous polygon/Voronoi-based skeletonization, CHASE results, tests | Reference/upstream. Read-only unless separately approved |

**Do not confuse:** a synthetic gradient check, a model trained on real annotated fundus images, a validated segmentation model, reliable physical vessel measurements, and a clinical diagnostic tool. These are different milestones. This repository has reached the second one. Nothing here measures vessel geometry yet, and nothing here improves on the results of the original `medskel` repository.

## Not done yet

- No centerlines, widths or lengths (milestone M5). CHASE_DB1 was used as an external segmentation test only.
- No patient-level split: FIVES publishes no patient ids (800 images, 573 patients).
- One training run per setting. One comparison model (pretrained ResNet34 U-Net); no comparison with published numbers.
- No online demo: a free Hugging Face Gradio Space could not be created (the site asked for a paid plan). Files for it are in `space/`.

Do not push datasets, patient data, secrets, large checkpoints, downloaded models, or unverified performance numbers to this repository. See [project boundaries](docs/11_REPOSITORY_BOUNDARIES.md) and [data protocol](docs/04_DATA_AND_EVALUATION.md).

## Sources

- U-Net, Ronneberger et al. (2015): https://arxiv.org/abs/1505.04597
- FIVES dataset paper, Jin et al. (2022): https://doi.org/10.1038/s41597-022-01564-3 — data: https://doi.org/10.6084/m9.figshare.19688169
- Original MedSkel geometry work: [original GitHub repository](https://github.com/MouhssineElBoumshouli/medskel)
- CHASE_DB1, Kingston University: https://researchinnovation.kingston.ac.uk/en/datasets/chasedb1-retinal-vessel-reference-dataset-4/
- Established alternatives: AutoMorph, VascX (see [`docs/06_RESEARCH_AND_COMPETITION.md`](docs/06_RESEARCH_AND_COMPETITION.md))

## For AI agents

Start with [`CLAUDE.md`](CLAUDE.md), [`PROGRESS.md`](PROGRESS.md) and [`docs/00_START_HERE.md`](docs/00_START_HERE.md). Inspect actual code and paper sources before choosing an approach. Keep explanations simple enough that the student can defend the CNN architecture and mathematics to the professor.
