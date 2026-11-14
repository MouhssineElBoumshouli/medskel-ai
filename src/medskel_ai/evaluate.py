"""Metrics and evaluation on whole images.

Two uses:

    # 1) choose the threshold on VALIDATION images (never on test)
    python -m medskel_ai.evaluate --split val

    # 2) final score on the official TEST set. Runs once; a second run is refused.
    python -m medskel_ai.evaluate --split test

Every pixel falls in one of four boxes:
    TP  vessel in the expert mask, vessel in our prediction      (found)
    FP  background in the expert mask, vessel in our prediction  (false alarm)
    FN  vessel in the expert mask, background in our prediction  (missed)
    TN  background in both                                       (correctly ignored)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .data import list_pairs, load_image, load_mask, make_split, resize_image_bilinear, resize_mask_nearest
from .predict import DEFAULT_CHECKPOINT, load_model, make_overlay, predict_probability

METRICS = ["dice", "iou", "precision", "recall", "specificity", "accuracy"]


def confusion_counts(pred: torch.Tensor, target: torch.Tensor) -> tuple[int, int, int, int]:
    """Count TP, FP, FN, TN. Both inputs are boolean tensors of the same shape."""
    tp = int((pred & target).sum())
    fp = int((pred & ~target).sum())
    fn = int((~pred & target).sum())
    tn = int((~pred & ~target).sum())
    return tp, fp, fn, tn


def _divide(top: float, bottom: float, if_empty: float) -> float:
    return top / bottom if bottom > 0 else if_empty


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    """All metrics from the four counts.

    Empty cases: if there is no vessel in the truth AND none predicted, the
    prediction is perfect, so Dice/IoU/precision/recall are defined as 1.
    """
    return {
        "dice": _divide(2 * tp, 2 * tp + fp + fn, 1.0),  # overlap; same as F1
        "iou": _divide(tp, tp + fp + fn, 1.0),  # overlap, stricter than Dice
        "precision": _divide(tp, tp + fp, 1.0 if fn == 0 else 0.0),  # of predicted vessel pixels, how many are real
        "recall": _divide(tp, tp + fn, 1.0),  # of real vessel pixels, how many we found
        "specificity": _divide(tn, tn + fp, 1.0),  # of real background pixels, how many we left alone
        "accuracy": _divide(tp + tn, tp + fp + fn + tn, 1.0),  # all correct pixels; misleading, see report
    }


def _mean_std(rows: list[dict]) -> dict:
    return {m: {"mean": float(np.mean([r[m] for r in rows])), "std": float(np.std([r[m] for r in rows]))} for m in METRICS}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(path: Path, pair: dict, mask: np.ndarray, pred: np.ndarray, dice: float, size: int = 512) -> None:
    """One row: image | expert mask | prediction | overlay (shrunk for display only)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = load_image(pair["image"])
    overlay = make_overlay(image, pred)
    panels = [
        ("image", resize_image_bilinear(image, size, size)),
        ("expert mask", resize_mask_nearest(mask, size, size) * 255),
        ("prediction", resize_mask_nearest(pred.astype(np.uint8), size, size) * 255),
        ("overlay (prediction in green)", resize_image_bilinear(overlay, size, size)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, (title, panel) in zip(axes, panels):
        ax.imshow(panel, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{pair['id']}   Dice = {dice:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained U-Net on whole images.")
    parser.add_argument("--data", default=r"C:\data\FIVES")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="use only the first N images (smoke test, val only)")
    parser.add_argument("--force", action="store_true", help="allow a second test evaluation (do not use casually)")
    args = parser.parse_args()

    run_dir = Path(args.checkpoint).parent
    model, device, _ = load_model(args.checkpoint)

    if args.split == "val":
        _, pairs = make_split(list_pairs(args.data, "train"), seed=args.seed)
        thresholds = [round(0.05 * k, 2) for k in range(2, 19)]  # 0.10, 0.15, ... 0.90
    else:
        if args.limit:
            raise SystemExit("--limit is not allowed on the test set.")
        result_file = run_dir / "test_metrics.json"
        if result_file.exists() and not args.force:
            raise SystemExit(f"{result_file} already exists. The test set is evaluated ONCE. Refusing to run again.")
        threshold_file = run_dir / "threshold.json"
        if not threshold_file.exists():
            raise SystemExit("Run '--split val' first: the threshold must be fixed on validation before the test.")
        thresholds = [json.loads(threshold_file.read_text())["threshold"]]
        pairs = list_pairs(args.data, "test")
    if args.limit:
        pairs = pairs[: args.limit]

    # counts[i][j] = (tp, fp, fn, tn) of image i at threshold j
    counts = []
    for number, pair in enumerate(pairs, 1):
        prob = torch.from_numpy(predict_probability(model, load_image(pair["image"]), device)).to(device)
        target = torch.from_numpy(load_mask(pair["mask"])).to(device).bool()
        counts.append([confusion_counts(prob >= t, target) for t in thresholds])
        if number % 20 == 0 or number == len(pairs):
            print(f"{args.split}: {number}/{len(pairs)} images", flush=True)

    # Mean per-image Dice for each threshold; the best one wins (val only has more than one).
    mean_dice = [float(np.mean([metrics_from_counts(*c[j])["dice"] for c in counts])) for j in range(len(thresholds))]
    best = int(np.argmax(mean_dice))
    threshold = thresholds[best]

    rows = []
    for pair, c in zip(pairs, counts):
        tp, fp, fn, tn = c[best]
        row = {"id": pair["id"], "group": pair["group"], "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        row.update(metrics_from_counts(tp, fp, fn, tn))
        rows.append(row)
    _write_csv(run_dir / f"{args.split}_per_image.csv", rows)

    # "Predict nothing" baseline: an all-black mask. Its accuracy is simply the background share.
    background_share = [(r["tn"] + r["fp"]) / (r["tp"] + r["fp"] + r["fn"] + r["tn"]) for r in rows]
    summary = {
        "split": args.split,
        "images": len(rows),
        "threshold": threshold,
        "per_image": _mean_std(rows),
        "median_dice": float(np.median([r["dice"] for r in rows])),
        "min_dice": float(np.min([r["dice"] for r in rows])),
        "max_dice": float(np.max([r["dice"] for r in rows])),
        "pooled_over_all_pixels": metrics_from_counts(*[sum(r[k] for r in rows) for k in ("tp", "fp", "fn", "tn")]),
        "by_group": {g: {"images": len(rs), **_mean_std(rs)} for g in sorted({r["group"] for r in rows}) for rs in [[r for r in rows if r["group"] == g]]},
        "all_background_baseline": {"accuracy": float(np.mean(background_share)), "dice": 0.0, "recall": 0.0},
        "note": "Metrics use every pixel of the image (no field-of-view mask). Mean and std are over images.",
    }

    if args.split == "val":
        summary["threshold_sweep"] = {f"{t:.2f}": d for t, d in zip(thresholds, mean_dice)}
        if not args.limit:
            (run_dir / "threshold.json").write_text(json.dumps({"threshold": threshold, "chosen_on": "validation", "mean_val_dice": mean_dice[best]}, indent=1))
        (run_dir / "val_metrics.json").write_text(json.dumps(summary, indent=1))
    else:
        (run_dir / "test_metrics.json").write_text(json.dumps(summary, indent=1))
        # Six examples picked by a fixed rule (not by taste): worst, 20/40/60/80th percentile, best.
        order = sorted(range(len(rows)), key=lambda i: rows[i]["dice"])
        picks = [order[int(round(q * (len(order) - 1)))] for q in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
        figure_dir = run_dir / "test_examples"
        figure_dir.mkdir(exist_ok=True)
        for rank, i in enumerate(picks, 1):
            mask = load_mask(pairs[i]["mask"])
            pred = predict_probability(model, load_image(pairs[i]["image"]), device) >= threshold
            name = f"{rank}_{pairs[i]['id'].replace('/', '_')}.png"
            _save_figure(figure_dir / name, pairs[i], mask, pred, rows[i]["dice"])

    print(json.dumps({k: summary[k] for k in ("split", "images", "threshold", "per_image", "by_group")}, indent=1))


if __name__ == "__main__":
    main()
