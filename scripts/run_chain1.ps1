# Chain 1: E3 (clDice loss, 50 min) then E2 (long training, 150 min), each followed by validation evaluation.
# Validation only. The test set is not touched.
$py = ".\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force runs\e3_cldice, runs\e2_long | Out-Null
& $py -m medskel_ai.train --minutes 50 --cldice-weight 0.5 --out runs/e3_cldice *> runs\e3_cldice\train.log
& $py -m medskel_ai.evaluate --run runs/e3_cldice --split val *> runs\e3_cldice\eval_val.log
& $py -m medskel_ai.evaluate --run runs/e3_cldice --split val --flips *> runs\e3_cldice\eval_val_flips.log
& $py -m medskel_ai.train --minutes 150 --out runs/e2_long *> runs\e2_long\train.log
& $py -m medskel_ai.evaluate --run runs/e2_long --split val *> runs\e2_long\eval_val.log
& $py -m medskel_ai.evaluate --run runs/e2_long --split val --flips *> runs\e2_long\eval_val_flips.log
Get-Content runs\e3_cldice\train.log -Tail 1
Get-Content runs\e3_cldice\eval_val.log -Tail 1
Get-Content runs\e3_cldice\eval_val_flips.log -Tail 1
Get-Content runs\e2_long\train.log -Tail 1
Get-Content runs\e2_long\eval_val.log -Tail 1
Get-Content runs\e2_long\eval_val_flips.log -Tail 1
