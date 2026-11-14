"""Compare experiments on the SAME validation images (paired comparison).

    python scripts/compare.py unet_fives unet_fives:flips e2_long e3_cldice

The first entry is the reference. For every other entry we print the mean of
each metric and the mean per-image difference to the reference with a 95%
bootstrap interval. If the interval does not contain 0, the difference is
unlikely to be chance (for these images and this single training run).
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

from medskel_ai.evaluate import bootstrap_ci

KEYS = ["dice", "cldice", "thin_recall", "precision", "recall"]


def load(entry: str, split: str = "val"):
    name, _, mode = entry.partition(":")
    suffix = "_flips" if mode == "flips" else ""
    run = Path("runs") / name
    rows = {r["id"]: r for r in csv.DictReader(open(run / f"{split}{suffix}_per_image.csv"))}
    summary = json.loads((run / f"{split}{suffix}_metrics.json").read_text())
    return rows, summary


reference_rows, _ = load(sys.argv[1])
ids = sorted(reference_rows)
out = {}
for entry in sys.argv[1:]:
    rows, summary = load(entry)
    line = {"threshold": summary["threshold"]}
    for k in KEYS:
        values = np.array([float(rows[i][k]) for i in ids])
        base = np.array([float(reference_rows[i][k]) for i in ids])
        low, high = bootstrap_ci(values - base)
        line[k] = round(float(np.nanmean(values)), 4)
        line[k + "_diff"] = [round(float(np.nanmean(values - base)), 4), round(low, 4), round(high, 4)]
    out[entry] = line
    print(entry, "t=%.2f" % line["threshold"], " ".join(f"{k}={line[k]:.4f}" for k in KEYS))
    if entry != sys.argv[1]:
        print("   diff vs reference [mean, low, high]:", {k: line[k + "_diff"] for k in ("dice", "cldice", "thin_recall")})
Path("results/v2").mkdir(parents=True, exist_ok=True)
Path("results/v2/validation_comparison.json").write_text(json.dumps(out, indent=1))
