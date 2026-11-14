# PROGRESS - resume file for a new session

Branch: `feature/unet-fives`. Scope: M1-M4 of `docs/09`. Issue #3. Never merge. Read `CLAUDE.md` first.

Setup facts (checked on this machine):

- FIVES at `C:\data\FIVES`: `train|test / Original|Ground truth / <n>_<A|D|G|N>.png`, 600 + 200 pairs.
- `.venv` = Python 3.11, torch 2.14.1+cu126, CUDA works. GPU: RTX 3050 Laptop 4 GB.
- Run output is in `runs/unet_fives/` (git-ignored): `best.pt`, logs, `test_examples/` figures.
- Small result files (numbers and one curve plot) are copied to `results/` (tracked).

## Steps

- [x] 1. Inspect repo and FIVES folder
- [x] 2. venv + CUDA PyTorch verified
- [x] 3. Code: `src/medskel_ai/{data,model,train,evaluate,predict}.py`, `app.py`
- [x] 4. Tests pass (`pytest`, 22 passed)
- [x] 5. Data audit -> `results/data_audit.json`, `results/split_seed42.json` (510 train / 88 val, 2 empty masks excluded, identical photos kept together)
- [x] 6. Training: 38 epochs, 51.2 min, best validation Dice 0.878 (fixed crops) at epoch 32
- [x] 7. Threshold on validation: 0.40 (whole-image validation Dice 0.899)
- [x] 8. Test evaluation done ONCE: mean Dice 0.875 over 200 images. Do not run again.
- [x] 9. `demo.ipynb` executed (pictures stripped from the committed copy; full copy in `outputs/`)
- [x] 10. `docs/explanation.md`, `report.md`
- [x] 11. README, tests, push, PR opened (not merged)

## Open points for a later session

- Look at the worst images by eye (`test/122_G`, `test/123_G`, `train/174_D`) to see if the photo or the label is the problem.
- Repeat training with other seeds for error bars.
- M5/M6 (geometry, CHASE_DB1) are NOT approved yet.
- M4 asked for a thinning skeleton view and slides; these were not part of the approved task list and are not done.

## Notes

- If a training is interrupted: `python -m medskel_ai.train --minutes 50 --resume` continues from `runs/unet_fives/last.pt`.
- `--split test` refuses to run if `runs/unet_fives/test_metrics.json` exists.
- Read logs with tail only (`Get-Content runs\unet_fives\train.log -Tail 5`).

## v2 (branch feature/v2-validation) - session stopped early by usage limit, 2026-11-14

Done and committed: topology.py (clDice, thin-vessel recall), flips in predict.py, bootstrap CIs + new evaluate.py (--run, --flips), soft clDice loss + --arch resnet34 in train.py, chase.py, scripts/worst_cases.py, 34 tests pass.
Running when the session stopped: baseline validation eval without and with flips -> runs/unet_fives/val_metrics.json, val_flips_metrics.json (read the RESULT line: Get-Content runs\unet_fives\eval_val_flips.log -Tail 1).
BLOCKED: CHASE_DB1 download (Kingston site has a bot check). Owner must download CHASEDB1.zip by hand and unzip into C:\data\CHASEDB1\images.
Not started: worst-case figures (python scripts/worst_cases.py train/174_D test/122_G test/123_G), E2 (train --minutes 150 --out runs/e2_long), E3 (train --minutes 50 --cldice-weight 0.5 --out runs/e3_cldice), E4, pretrained baseline (needs pip install torchvision for cu126 + segmentation-models-pytorch), CHASE eval, final test, report/explanation update, PR, demo redesign, HF Space.
Nothing in v2 has touched the FIVES test set yet.
