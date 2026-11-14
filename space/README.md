---
title: MedSkel AI - Retinal Vessel Segmentation
emoji: 👁️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Small U-Net that marks blood vessels in retina photos
---

# MedSkel AI — retinal vessel segmentation demo

A small U-Net (7.85 million parameters), trained from scratch on the FIVES dataset, marks the blood vessels in a colour photograph of the retina. University class project.

**Research prototype. Not a medical device. No diagnosis.**

- Runs on the free CPU tier. One photograph takes about one to two minutes (25 tiles of 512 x 512). The "4 mirrored passes" option is about 4 times slower.
- Results, limits and method: see the **About** tab and the code repository https://github.com/MouhssineElBoumshouli/medskel-ai
- Training data: FIVES, Jin et al., *Scientific Data* 9, 475 (2022), https://doi.org/10.1038/s41597-022-01564-3, licence CC BY 4.0. The model weights in `model/` were trained on this dataset.

## Files in this Space

| File | Purpose |
| --- | --- |
| `app.py` | the Gradio demo (same file as in the code repository) |
| `ABOUT.md` | text of the About tab, with the real results |
| `src/medskel_ai/` | model and prediction code |
| `model/best.pt`, `model/threshold*.json` | trained weights and the validation-chosen thresholds |
| `requirements.txt` | CPU build of PyTorch |

This folder is assembled by `scripts/build_space.ps1` in the code repository.
