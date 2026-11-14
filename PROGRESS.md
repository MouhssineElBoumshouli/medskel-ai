# PROGRESS - resume file for a new session

Branch: `feature/unet-fives`. Scope: M1-M4 of `docs/09`. Never merge. Read `CLAUDE.md` first.

Setup facts (checked on this machine):

- FIVES at `C:\data\FIVES`: `train|test / Original|Ground truth / <n>_<A|D|G|N>.png`, 600 + 200 pairs.
- `.venv` = Python 3.11, torch 2.14.1+cu126, CUDA works. GPU: RTX 3050 Laptop 4 GB.
- Run output goes to `runs/unet_fives/` (git-ignored). Small result files are copied to `results/` (tracked).

## Steps

- [x] 1. Inspect repo and FIVES folder
- [x] 2. venv + CUDA PyTorch verified
- [x] 3. Code: `src/medskel_ai/{data,model,train,evaluate,predict}.py`, `app.py`
- [x] 4. Tests pass (`pytest`, 20 passed)
- [ ] 5. Data audit (`python -m medskel_ai.data`) -> `results/data_audit.json`, `results/split_seed42.json`
- [ ] 6. Train (`python -m medskel_ai.train --minutes 50`) -> `runs/unet_fives/`  (started 2026-11-14, background)
- [ ] 7. Threshold on validation (`python -m medskel_ai.evaluate --split val`)
- [ ] 8. Test evaluation ONCE (`python -m medskel_ai.evaluate --split test`)
- [ ] 9. Executed `demo.ipynb`
- [ ] 10. `docs/explanation.md` sections 10-12 (results, limits, Q&A), `report.md`
- [ ] 11. README, all tests, push, issue + PR (no merge)

## Notes for the next session

- If training was interrupted: `python -m medskel_ai.train --minutes 50 --resume` continues from `runs/unet_fives/last.pt`.
- Do not run `--split test` twice. The script refuses if `runs/unet_fives/test_metrics.json` exists.
- Read logs with tail only (`Get-Content runs\unet_fives\train.log -Tail 5`).
- `docs/explanation.md` still has RESULTS_PLACEHOLDER / LIMITS_PLACEHOLDER / QA_PLACEHOLDER to fill with real numbers.
