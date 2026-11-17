# PROGRESS - resume file for a new session

Read `CLAUDE.md` first. Never merge. Version 1 = PR #4 (merged by the owner). Version 2 = branch `feature/v2-validation`.

Setup facts (checked on this machine):

- FIVES at `C:\data\FIVES` (600 train + 200 test pairs). CHASE_DB1 at `C:\data\CHASEDB1\images` (28 images, 56 masks; downloaded by hand by the owner, the site blocks scripts).
- `.venv` = Python 3.11, torch 2.14.1+cu126, torchvision 0.29.1+cu126, segmentation-models-pytorch 0.6.0, scikit-image, scipy, gradio 6.32. GPU: RTX 3050 Laptop 4 GB.
- Run folders (git-ignored): `runs/unet_fives` (v1 baseline), `runs/e3_cldice`, `runs/e2_long` (FINAL plain U-Net), `runs/pretrained_r34` (comparison). Numbers are copied to `results/` and `results/v2/`.

## Version 2 - all done

- [x] 1. Weakness measured (validation, baseline): clDice 0.905, thin-vessel recall 0.459.
- [x] 2. E1 flips: kept. E2 150 min: kept. E3 soft clDice loss: worse, dropped. E4 = E2 + flips = final (val Dice 0.910, threshold 0.30).
- [x] 3. Pretrained ResNet34 U-Net (150 min): val Dice 0.924 with flips, threshold 0.40.
- [x] 4. CHASE_DB1 external: final 0.766, pretrained 0.770, baseline 0.768 vs observer 1; human observer 2 vs 1 = 0.777.
- [x] 5. Bootstrap 95% intervals everywhere (children resampled on CHASE).
- [x] 6. Worst images viewed: train/174_D = mismatched expert mask; test/122_G, 123_G = dark blurred photos.
- [x] 7. FIVES test, once per final model: plain 0.889, pretrained 0.907. DO NOT RUN TEST AGAIN (the script refuses).
- [x] 8. report.md, docs/explanation.md (Part 2), README, ABOUT.md, 34 tests pass, PR opened.
- [x] 9. app.py redesigned (Gradio Blocks, slider, About tab, model selector). Local app works.
- [ ] Hugging Face Space: BLOCKED - Hugging Face asks for a paid PRO plan for Gradio Spaces. Nothing was created. `space/` + `scripts/build_space.ps1` are ready.

## Open points

- `demo.ipynb` still shows version 1 (baseline model). Not updated in v2.
- train/174_D (wrong mask) is still in the validation set, on purpose, so all experiments share one validation set. Remove it before any new round of experiments.
- M5 measurement study: code written on branch `feature/measurements` (stacked on v2). See docs/12_MORPHOMETRY.md.
  - [x] src/medskel_ai/morphometry.py, src/medskel_ai/study.py, tests/test_morphometry.py (16 pass with the medskel clone; synthetic data only)
  - [x] Found and worked around a width bug in the original MedSkel code (documented in docs/12)
  - [x] Run on the laptop (2026-11-17): 1081 masks measured; human-vs-human length check reproduces 0.093 / 0.032
  - [x] results/morphometry/ committed; report.md section 6 written with the real numbers
- Ideas for thin vessels, untested: width-weighted loss, training at higher zoom, several seeds.

## Notes

- Compare experiments: `python scripts/compare.py unet_fives unet_fives:flips e3_cldice e2_long e2_long:flips pretrained_r34 pretrained_r34:flips`
- Read logs with tail only. Interrupted training: add `--resume`.