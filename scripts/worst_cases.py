"""Save small side-by-side figures (photo | expert mask | prediction) for chosen FIVES images.

    python scripts/worst_cases.py --run runs/unet_fives train/174_D test/122_G test/123_G

These three were the worst images of the first report. The figures go to
<run>/worst_cases/ (not in Git). The script also prints the expert vessel fraction.
"""

import argparse
from pathlib import Path

from medskel_ai.data import list_pairs, load_image, load_mask
from medskel_ai.evaluate import confusion_counts, metrics_from_counts, save_figure
from medskel_ai.predict import load_model, predict_probability

import torch

parser = argparse.ArgumentParser()
parser.add_argument("ids", nargs="+")
parser.add_argument("--data", default=r"C:\data\FIVES")
parser.add_argument("--run", default="runs/unet_fives")
args = parser.parse_args()

model, device, threshold = load_model(Path(args.run) / "best.pt")
pairs = {p["id"]: p for split in ("train", "test") for p in list_pairs(args.data, split)}
out = Path(args.run) / "worst_cases"
out.mkdir(exist_ok=True)
for image_id in args.ids:
    pair = pairs[image_id]
    image, mask = load_image(pair["image"]), load_mask(pair["mask"])
    pred = predict_probability(model, image, device) >= threshold
    m = metrics_from_counts(*confusion_counts(torch.from_numpy(pred), torch.from_numpy(mask).bool()))
    title = f"{image_id}  Dice {m['dice']:.3f}  expert vessel pixels {mask.mean() * 100:.2f}%  predicted {pred.mean() * 100:.2f}%"
    save_figure(out / (image_id.replace("/", "_") + ".png"), image, mask, pred, title, size=420, overlay=False)
    print(title, f"brightness mean {image.mean():.1f}")
