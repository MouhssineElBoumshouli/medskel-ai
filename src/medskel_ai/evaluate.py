"""Metrics and evaluation on whole FIVES images.

    # 1) choose the threshold on VALIDATION images (never on test)
    python -m medskel_ai.evaluate --run runs/unet_fives --split val [--flips]

    # 2) final score on the official TEST set. One run per model; a second run is refused.
    python -m medskel_ai.evaluate --run runs/unet_fives --split test [--flips]

Every pixel falls in one of four boxes:
    TP  vessel in the expert mask, vessel in our prediction      (found)
    FP  background in the expert mask, vessel in our prediction  (false alarm)
    FN  vessel in the expert mask, background in our prediction  (missed)
    TN  background in both                                       (correctly ignored)

Besides the pixel metrics we report clDice (connectivity) and thin-vessel
recall (see topology.py), and a 95% bootstrap confidence interval for every mean.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .data import list_pairs, load_image, load_mask, make_split, resize_image_bilinear, resize_mask_nearest
from .predict import load_model, make_overlay, predict_probability
from .topology import skeleton, thin_vessel_mask, topology_metrics

METRICS = ["dice", "iou", "precision", "recall", "specificity", "accuracy", "cldice", "thin_recall"]
CACHE_DIR = Path("runs/cache")  # skeleton and thin-vessel mask of each expert mask (they never change)


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
    """All pixel metrics from the four counts.

    Empty cases: if there is no vessel in the truth AND none predicted, the
    prediction is perfect, so Dice/IoU/precision/recall are defined as 1.
    """
    return {
        "dice": _divide(2 * tp, 2 * tp + fp + fn, 1.0),  # overlap; same as F1
        "iou": _divide(tp, tp + fp + fn, 1.0),  # overlap, stricter than Dice
        "precision": _divide(tp, tp + fp, 1.0 if fn == 0 else 0.0),  # of predicted vessel pixels, how many are real
        "recall": _divide(tp, tp + fn, 1.0),  # of real vessel pixels, how many we found (= sensitivity)
        "specificity": _divide(tn, tn + fp, 1.0),  # of real background pixels, how many we left alone
        "accuracy": _divide(tp + tn, tp + fp + fn + tn, 1.0),  # all correct pixels; misleading, see report
    }


def bootstrap_ci(values, clusters=None, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    """95% confidence interval of a mean by the bootstrap.

    Idea: our images are one random sample. To see how much the mean could
    change with another sample, we draw a new sample of the same size FROM our
    own images, with replacement (some images twice, some not at all), and
    compute the mean. We do this 10,000 times. The interval that holds the
    middle 95% of these means is the confidence interval.

    clusters: if images are not independent (the two eyes of one child), give
    one label per image. Then whole clusters are drawn together.
    NaN values (metric not defined for that image) are skipped.
    """
    v = np.asarray(values, dtype=float)
    labels = np.arange(len(v)) if clusters is None else np.asarray(clusters)
    ids, inverse = np.unique(labels, return_inverse=True)
    valid = ~np.isnan(v)
    sums = np.bincount(inverse, weights=np.where(valid, v, 0.0), minlength=len(ids))
    counts = np.bincount(inverse, weights=valid.astype(float), minlength=len(ids))
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    low, high = np.nanpercentile(means, [2.5, 97.5])
    return float(low), float(high)


def summarize(rows: list[dict], metrics=METRICS, clusters=None) -> dict:
    """Mean, spread and 95% bootstrap interval of each metric over images."""
    out = {}
    for m in metrics:
        values = np.array([r[m] for r in rows], dtype=float)
        low, high = bootstrap_ci(values, clusters)
        out[m] = {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values)), "ci95": [low, high], "n": int((~np.isnan(values)).sum())}
    return out


def expert_structures(key: str, truth: np.ndarray):
    """Skeleton and thin-vessel mask of an expert mask, cached on disk because they are slow and never change."""
    path = CACHE_DIR / (key.replace("/", "_") + ".npz")
    if path.exists():
        saved = np.load(path)
        return saved["skeleton"], saved["thin"]
    skel = skeleton(truth)
    thin = thin_vessel_mask(truth, skel)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, skeleton=skel, thin=thin)
    return skel, thin


def image_metrics(pred: np.ndarray, truth: np.ndarray, cache_key: str | None = None, region: np.ndarray | None = None) -> dict:
    """All metrics of one image. pred and truth are boolean arrays.

    region: optional boolean mask (field of view). Pixels outside it are ignored.
    """
    pred, truth = pred.astype(bool), truth.astype(bool)
    if region is not None:
        pred, truth = pred & region, truth & region
        counts = confusion_counts(torch.from_numpy(pred[region]), torch.from_numpy(truth[region]))
    else:
        counts = confusion_counts(torch.from_numpy(pred), torch.from_numpy(truth))
    skel, thin = expert_structures(cache_key, truth) if cache_key else (None, None)
    row = dict(zip(("tp", "fp", "fn", "tn"), counts))
    row.update(metrics_from_counts(*counts))
    row.update(topology_metrics(pred, truth, skel, thin))
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(path: Path, image: np.ndarray, mask: np.ndarray, pred: np.ndarray, title: str, size: int = 512, overlay: bool = True) -> None:
    """One row: image | expert mask | prediction [| overlay] (shrunk for display only)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    height, width = mask.shape
    w, h = size, round(size * height / width)
    panels = [
        ("image", resize_image_bilinear(image, w, h)),
        ("expert mask", resize_mask_nearest(mask, w, h) * 255),
        ("prediction", resize_mask_nearest(pred.astype(np.uint8), w, h) * 255),
    ]
    if overlay:
        panels.append(("overlay (prediction in green)", resize_image_bilinear(make_overlay(image, pred), w, h)))
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.4))
    for ax, (name, panel) in zip(axes, panels):
        ax.imshow(panel, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110 if overlay else 75)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on whole FIVES images.")
    parser.add_argument("--data", default=r"C:\data\FIVES")
    parser.add_argument("--run", default="runs/unet_fives", help="folder that contains best.pt")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--flips", action="store_true", help="average predictions over mirrored versions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="use only the first N images (smoke test, val only)")
    args = parser.parse_args()

    run_dir = Path(args.run)
    suffix = "_flips" if args.flips else ""
    model, device, _ = load_model(run_dir / "best.pt")

    if args.split == "val":
        _, pairs = make_split(list_pairs(args.data, "train"), seed=args.seed)
        thresholds = [round(0.05 * k, 2) for k in range(2, 19)]  # 0.10, 0.15, ... 0.90
    else:
        if args.limit:
            raise SystemExit("--limit is not allowed on the test set.")
        if list(run_dir.glob("test*_metrics.json")):
            raise SystemExit(f"{run_dir} already has a test result. One test evaluation per model. Refusing to run again.")
        threshold_file = run_dir / f"threshold{suffix}.json"
        if not threshold_file.exists():
            raise SystemExit("Run '--split val' first: the threshold must be fixed on validation before the test.")
        thresholds = [json.loads(threshold_file.read_text())["threshold"]]
        pairs = list_pairs(args.data, "test")
    if args.limit:
        pairs = pairs[: args.limit]

    # Pass 1: predict every image; count TP/FP/FN/TN for every candidate threshold.
    counts, stored = [], []
    for number, pair in enumerate(pairs, 1):
        prob = predict_probability(model, load_image(pair["image"]), device, flips=args.flips, as_tensor=True).half()
        target = torch.from_numpy(load_mask(pair["mask"])).to(device).bool()
        counts.append([confusion_counts(prob >= t, target) for t in thresholds])
        stored.append(prob.cpu() if len(thresholds) > 1 else (prob >= thresholds[0]).cpu().numpy())
        if number % 20 == 0 or number == len(pairs):
            print(f"{args.split}{suffix}: predicted {number}/{len(pairs)}", flush=True)

    # The threshold with the best mean per-image Dice wins (on test there is only one candidate).
    mean_dice = [float(np.mean([metrics_from_counts(*c[j])["dice"] for c in counts])) for j in range(len(thresholds))]
    best = int(np.argmax(mean_dice))
    threshold = thresholds[best]

    # Pass 2: all metrics (including the slow skeleton-based ones) at the chosen threshold.
    rows = []
    for number, (pair, item) in enumerate(zip(pairs, stored), 1):
        pred = item if isinstance(item, np.ndarray) else (item >= threshold).numpy()
        row = {"id": pair["id"], "group": pair["group"]}
        row.update(image_metrics(pred, load_mask(pair["mask"]), cache_key=pair["id"]))
        rows.append(row)
        if number % 40 == 0:
            print(f"{args.split}{suffix}: measured {number}/{len(pairs)}", flush=True)
    write_csv(run_dir / f"{args.split}{suffix}_per_image.csv", rows)

    # "Predict nothing" baseline: an all-black mask. Its accuracy is simply the background share.
    background_share = [(r["tn"] + r["fp"]) / (r["tp"] + r["fp"] + r["fn"] + r["tn"]) for r in rows]
    groups = sorted({r["group"] for r in rows})
    summary = {
        "split": args.split,
        "flips": args.flips,
        "images": len(rows),
        "threshold": threshold,
        "per_image": summarize(rows),
        "median_dice": float(np.median([r["dice"] for r in rows])),
        "min_dice": float(np.min([r["dice"] for r in rows])),
        "max_dice": float(np.max([r["dice"] for r in rows])),
        "thin_share_of_vessel_pixels_mean": float(np.nanmean([r["thin_share_of_vessel_pixels"] for r in rows])),
        "thick_recall_mean": float(np.nanmean([r["thick_recall"] for r in rows])),
        "pooled_over_all_pixels": metrics_from_counts(*[sum(r[k] for r in rows) for k in ("tp", "fp", "fn", "tn")]),
        "by_group": {g: {"images": sum(r["group"] == g for r in rows), **summarize([r for r in rows if r["group"] == g])} for g in groups},
        "all_background_baseline": {"accuracy": float(np.mean(background_share)), "dice": 0.0, "recall": 0.0},
        "note": "Every pixel of the image is used (no field-of-view mask). Mean, std and 95% bootstrap interval are over images.",
    }

    if args.split == "val":
        summary["threshold_sweep"] = {f"{t:.2f}": d for t, d in zip(thresholds, mean_dice)}
        if not args.limit:
            chosen = {"threshold": threshold, "chosen_on": "validation", "flips": args.flips, "mean_val_dice": mean_dice[best]}
            (run_dir / f"threshold{suffix}.json").write_text(json.dumps(chosen, indent=1))
    else:
        # Six examples picked by a fixed rule (not by taste): worst, 20/40/60/80th percentile, best.
        order = sorted(range(len(rows)), key=lambda i: rows[i]["dice"])
        picks = [order[int(round(q * (len(order) - 1)))] for q in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
        figure_dir = run_dir / "test_examples"
        figure_dir.mkdir(exist_ok=True)
        for rank, i in enumerate(picks, 1):
            name = f"{rank}_{pairs[i]['id'].replace('/', '_')}.png"
            title = f"{pairs[i]['id']}   Dice = {rows[i]['dice']:.3f}"
            save_figure(figure_dir / name, load_image(pairs[i]["image"]), load_mask(pairs[i]["mask"]), stored[i], title)
    (run_dir / f"{args.split}{suffix}_metrics.json").write_text(json.dumps(summary, indent=1))

    short = {m: round(summary["per_image"][m]["mean"], 4) for m in ("dice", "cldice", "thin_recall", "precision", "recall")}
    print(f"RESULT {run_dir.name} {args.split}{suffix} threshold={threshold} {short} thick_recall={summary['thick_recall_mean']:.4f}", flush=True)


if __name__ == "__main__":
    main()
