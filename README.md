# MedSkel AI

**Research-first retinal blood vessel segmentation and measurement.** The immediate deliverable is a small, understandable **PyTorch U-Net** for a university CNN project. The long-term goal is to test whether retinal vessel geometry can be measured accurately and reproducibly enough for research workflows.

> **Current status (14 November 2026): a U-Net trained on real FIVES expert masks, evaluated once on the official test set, with a local demo.** Research prototype for a class project. No clinical validation. Not a medical device.

## Result of the first real run

Plain U-Net, 7,849,601 parameters, trained 51 minutes on an RTX 3050 (4 GB) on 510 FIVES training images (random 512 x 512 crops at full resolution). Threshold 0.40 chosen on 88 validation images. Official test set (200 images) evaluated **once**:

| Dice | IoU | Precision | Recall | Specificity | Accuracy |
| --- | --- | --- | --- | --- | --- |
| **0.875** | 0.791 | 0.909 | 0.856 | 0.993 | 0.986 |

Mean over images. Median Dice 0.907, worst image 0.031. By group: A 0.908, D 0.876, G 0.828, N 0.887. An all-black prediction already has accuracy 0.929, so accuracy is not a useful score here.

Read these numbers with their limits: FIVES has 800 images from 573 patients and no patient ids, so patient overlap between train, validation and test cannot be ruled out; one run with one seed; no external dataset. Full details in [`report.md`](report.md), raw numbers in [`results/`](results/).

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
| Train (about 50 minutes on an RTX 3050 4 GB) | `.\.venv\Scripts\python.exe -m medskel_ai.train --minutes 50` |
| Choose the threshold on validation images | `.\.venv\Scripts\python.exe -m medskel_ai.evaluate --split val` |
| Final test evaluation (runs **once**) | `.\.venv\Scripts\python.exe -m medskel_ai.evaluate --split test` |
| Segment one photograph | `.\.venv\Scripts\python.exe -m medskel_ai.predict --image photo.png` |
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
| [`src/medskel_ai/predict.py`](src/medskel_ai/predict.py) | Whole-image prediction by tiling and stitching |
| [`PROGRESS.md`](PROGRESS.md) | Step list so a new session can resume |

## What this repository is

This is the **dedicated MedSkel AI development repository**. It holds the new CNN code, experiments, documentation, and the class demonstration. It is **not** the original geometry repository and must not overwrite its findings.

| Repo | Purpose | Rules |
| --- | --- | --- |
| **This repository: `medskel-ai`** | New CNN, FIVES pipeline, segmentation evaluation, demo, future carefully tested integration | Feature branches + PR review; new results must be reproducible |
| [Original `medskel`](https://github.com/MouhssineElBoumshouli/medskel) | Previous polygon/Voronoi-based skeletonization, CHASE results, tests | Reference/upstream. Read-only unless separately approved |

**Do not confuse:** a synthetic gradient check, a model trained on real annotated fundus images, a validated segmentation model, reliable physical vessel measurements, and a clinical diagnostic tool. These are different milestones. This repository has reached the second one. Nothing here measures vessel geometry yet, and nothing here improves on the results of the original `medskel` repository.

## Not done yet

- No centerlines, widths or lengths (milestone M5), no CHASE_DB1 external test (M6).
- No patient-level split: FIVES publishes no patient ids (800 images, 573 patients).
- One training run with one seed. No comparison against other models.

Do not push datasets, patient data, secrets, large checkpoints, downloaded models, or unverified performance numbers to this repository. See [project boundaries](docs/11_REPOSITORY_BOUNDARIES.md) and [data protocol](docs/04_DATA_AND_EVALUATION.md).

## Sources

- U-Net, Ronneberger et al. (2015): https://arxiv.org/abs/1505.04597
- FIVES dataset paper, Jin et al. (2022): https://doi.org/10.1038/s41597-022-01564-3 — data: https://doi.org/10.6084/m9.figshare.19688169
- Original MedSkel geometry work: [original GitHub repository](https://github.com/MouhssineElBoumshouli/medskel)
- CHASE_DB1, Kingston University: https://researchinnovation.kingston.ac.uk/en/datasets/chasedb1-retinal-vessel-reference-dataset-4/
- Established alternatives: AutoMorph, VascX (see [`docs/06_RESEARCH_AND_COMPETITION.md`](docs/06_RESEARCH_AND_COMPETITION.md))

## For AI agents

Start with [`CLAUDE.md`](CLAUDE.md), [`PROGRESS.md`](PROGRESS.md) and [`docs/00_START_HERE.md`](docs/00_START_HERE.md). Inspect actual code and paper sources before choosing an approach. Keep explanations simple enough that the student can defend the CNN architecture and mathematics to the professor.
