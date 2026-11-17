"""External test on CHASE_DB1: images from another camera, country and age group.

    python -m medskel_ai.chase --run runs/unet_fives [--flips]

Nothing is trained or tuned here. The model and its threshold come from FIVES.

CHASE_DB1 (Kingston University): 28 photographs, both eyes of 14 children,
hand-held Nidek NM-200-D camera, 30 degree field of view. Each image has TWO
expert masks, drawn independently by two human observers:

    Image_01L.jpg   Image_01L_1stHO.png   Image_01L_2ndHO.png

Because there are two humans, we can ask a fair question: is the model as close
to observer 1 as observer 2 is? (human vs human = the realistic upper limit).

Three careful points, all decided BEFORE looking at any CHASE result:

1) Scale. The network learned how wide vessels are IN PIXELS on FIVES.
   FIVES: retina disc about FIVES_FOV_PIXELS wide for 50 degrees.
   CHASE: retina disc about 960-1000 px wide for 30 degrees.
   So one degree of retina covers a different number of pixels. We enlarge each
   CHASE image so that pixels-per-degree match FIVES, predict, and shrink the
   PROBABILITY map back to the original size (bilinear is fine for
   probabilities; masks are never interpolated). This "matched" setting is our
   main result. We also report the "native" setting (no resizing) for honesty.
   The field-of-view angles are the cameras' nominal values, so the match is approximate.

2) Field of view (FOV). CHASE images have a dark border around the retina and no
   official FOV mask. We build one by thresholding brightness, and count only
   pixels inside it. (On FIVES we counted all pixels. Sensitivity and Dice are
   barely affected; specificity is a little lower inside the FOV, which is the
   stricter choice.)

3) Independence. The two eyes of one child are similar. For confidence
   intervals we resample CHILDREN (14), not images (28).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from .data import load_image, load_mask
from .evaluate import bootstrap_ci, image_metrics, summarize, write_csv
from .predict import load_model, predict_probability

FIVES_FOV_PIXELS = 2009.0  # median width of the retina disc, measured on 24 FIVES training images (range 1554-2038)
FIVES_FOV_DEGREES = 50.0  # FIVES paper (Jin et al. 2022): TRC-NW8 camera, 50 degree field of view
CHASE_FOV_DEGREES = 30.0  # Kingston University dataset page: Nidek NM-200-D, 30 degree field of view
REPORT = ["dice", "cldice", "recall", "specificity", "precision", "thin_recall"]


def list_chase(root: str | Path) -> list[dict]:
    """Find every image with both observer masks. The child number is the two digits in the name."""
    items = []
    for image in sorted(Path(root).rglob("Image_*.jpg")):
        first = image.with_name(image.stem + "_1stHO.png")
        second = image.with_name(image.stem + "_2ndHO.png")
        if not (first.exists() and second.exists()):
            raise FileNotFoundError(f"Observer masks missing for {image.name}")
        code = image.stem.split("_")[1]  # '01L'
        items.append({"id": f"chase/{code}", "child": code[:2], "eye": code[2:], "image": str(image), "mask1": str(first), "mask2": str(second)})
    return items


def field_of_view(image: np.ndarray, threshold: int = 20) -> np.ndarray:
    """Boolean mask of the retina disc: brighter than the dark border, largest blob, holes filled."""
    bright = image.max(axis=2) > threshold
    labels, count = ndimage.label(bright)
    if count == 0:
        return np.ones(bright.shape, dtype=bool)
    sizes = ndimage.sum(bright, labels, index=range(1, count + 1))
    return ndimage.binary_fill_holes(labels == (int(np.argmax(sizes)) + 1))


def fov_width(fov: np.ndarray) -> int:
    """Widest horizontal extent of the field of view, in pixels."""
    columns = np.where(fov.any(axis=0))[0]
    return int(columns[-1] - columns[0] + 1)


@torch.no_grad()
def predict_at_scale(model, image: np.ndarray, device, scale: float, flips: bool) -> np.ndarray:
    """Enlarge the image by `scale`, predict, and bring the probability map back to the original size."""
    height, width = image.shape[:2]
    if abs(scale - 1.0) < 1e-6:
        return predict_probability(model, image, device, flips=flips)
    x = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().unsqueeze(0)
    big = torch.nn.functional.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
    big = big[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy()
    prob = predict_probability(model, big, device, flips=flips, as_tensor=True)[None, None]
    prob = torch.nn.functional.interpolate(prob, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
    return prob[0, 0].cpu().numpy()


def _table(rows: list[dict], children: list[str]) -> dict:
    return summarize(rows, metrics=REPORT, clusters=children)


def main() -> None:
    parser = argparse.ArgumentParser(description="External evaluation on CHASE_DB1 (no training, no tuning).")
    parser.add_argument("--data", default=r"C:\data\CHASEDB1")
    parser.add_argument("--run", default="runs/unet_fives")
    parser.add_argument("--flips", action="store_true")
    parser.add_argument("--fives-fov-pixels", type=float, default=FIVES_FOV_PIXELS)
    args = parser.parse_args()

    run_dir = Path(args.run)
    suffix = "_flips" if args.flips else ""
    model, device, threshold = load_model(run_dir / "best.pt", flips=args.flips)
    items = list_chase(args.data)
    children = [item["child"] for item in items]

    per_image, human_rows, scales = [], [], []
    model_rows = {(setting, observer): [] for setting in ("matched", "native") for observer in ("observer1", "observer2")}
    for item in items:
        image = load_image(item["image"])
        truth = {"observer1": load_mask(item["mask1"]).astype(bool), "observer2": load_mask(item["mask2"]).astype(bool)}
        if truth["observer1"].shape != image.shape[:2]:
            raise ValueError(f"{item['id']}: image and mask sizes differ")
        fov = field_of_view(image)
        # pixels per degree in FIVES divided by pixels per degree in this image
        matched = (args.fives_fov_pixels / FIVES_FOV_DEGREES) / (fov_width(fov) / CHASE_FOV_DEGREES)
        scales.append(matched)
        row = {"id": item["id"], "child": item["child"], "scale_matched": round(matched, 4)}
        for setting, scale in (("matched", matched), ("native", 1.0)):
            pred = predict_at_scale(model, image, device, scale, args.flips) >= threshold
            for observer in ("observer1", "observer2"):
                m = image_metrics(pred, truth[observer], cache_key=f"{item['id']}_{observer}", region=fov)
                model_rows[(setting, observer)].append(m)
                row.update({f"{setting}_{observer}_{k}": m[k] for k in REPORT})
        # Human against human: observer 2 plays the role of the "prediction", observer 1 is the reference.
        h = image_metrics(truth["observer2"], truth["observer1"], cache_key=f"{item['id']}_observer1", region=fov)
        human_rows.append(h)
        row.update({f"human_{k}": h[k] for k in REPORT})
        per_image.append(row)
        print(f"chase{suffix}: {item['id']} done", flush=True)

    # Paired difference: (model vs observer 1) minus (observer 2 vs observer 1), image by image.
    difference = {}
    for k in REPORT:
        d = np.array([m[k] - h[k] for m, h in zip(model_rows[("matched", "observer1")], human_rows)], dtype=float)
        low, high = bootstrap_ci(d, children)
        difference[k] = {"mean": float(np.nanmean(d)), "ci95": [low, high]}

    summary = {
        "dataset": "CHASE_DB1",
        "images": len(items),
        "children": len(set(children)),
        "flips": args.flips,
        "threshold_from_fives_validation": threshold,
        "scale_matched_mean": float(np.mean(scales)),
        "fives_fov_pixels": args.fives_fov_pixels,
        "model": {f"{setting}_vs_{observer}": _table(rows, children) for (setting, observer), rows in model_rows.items()},
        "human_observer2_vs_observer1": _table(human_rows, children),
        "model_matched_vs_observer1_minus_human": difference,
        "note": "Metrics inside the field of view. 95% intervals resample children (cluster bootstrap), because both eyes of a child are not independent.",
    }
    (run_dir / f"chase{suffix}_metrics.json").write_text(json.dumps(summary, indent=1))
    write_csv(run_dir / f"chase{suffix}_per_image.csv", per_image)

    def short(table):
        return {k: round(table[k]["mean"], 4) for k in ("dice", "cldice", "recall", "specificity")}

    print(f"RESULT {run_dir.name} chase{suffix} threshold={threshold} scale={summary['scale_matched_mean']:.3f}", flush=True)
    for name, table in summary["model"].items():
        print(f"  model {name}: {short(table)}", flush=True)
    print(f"  human observer2 vs observer1: {short(summary['human_observer2_vs_observer1'])}", flush=True)


if __name__ == "__main__":
    main()
