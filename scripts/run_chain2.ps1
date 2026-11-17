# Chain 2. Decisions were fixed on validation before this script was written:
#   final plain U-Net = runs/e2_long (150 min) with flips, threshold from threshold_flips.json.
# 1) final model: CHASE_DB1 (external) and the official FIVES test set, once.
# 2) pretrained ResNet34 U-Net: same data, same 150 min budget, same loss; validation evaluation.
$py = ".\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force runs\pretrained_r34 | Out-Null
& $py -m medskel_ai.chase --run runs/e2_long --flips *> runs\e2_long\chase_flips.log
& $py -m medskel_ai.evaluate --run runs/e2_long --split test --flips *> runs\e2_long\eval_test_flips.log
& $py -m medskel_ai.train --minutes 150 --arch resnet34 --out runs/pretrained_r34 *> runs\pretrained_r34\train.log
& $py -m medskel_ai.evaluate --run runs/pretrained_r34 --split val *> runs\pretrained_r34\eval_val.log
& $py -m medskel_ai.evaluate --run runs/pretrained_r34 --split val --flips *> runs\pretrained_r34\eval_val_flips.log
Get-Content runs\e2_long\chase_flips.log -Tail 6
Get-Content runs\e2_long\eval_test_flips.log -Tail 1
Get-Content runs\pretrained_r34\train.log -Tail 1
Get-Content runs\pretrained_r34\eval_val.log -Tail 1
Get-Content runs\pretrained_r34\eval_val_flips.log -Tail 1
