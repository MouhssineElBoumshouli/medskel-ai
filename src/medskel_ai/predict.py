"""Predict the vessels of a whole photograph.

The network was trained on 512 x 512 crops, and a full 2048 x 2048 image does
not fit in 4 GB of GPU memory. So we:

    1. cut the image into 512 x 512 tiles that overlap by 64 pixels,
    2. predict each tile,
    3. put the tiles back in place and AVERAGE the probabilities where tiles overlap.

The overlap matters: near a tile border the network sees less context, so its
prediction there is weaker. Averaging hides the seams.

Optional "flips" (test-time augmentation, TTA): predict the image four times
(as it is, mirrored left-right, mirrored up-down, both), mirror each answer
back, and average the four probability maps. A vessel is still a vessel in a
mirror, so the four answers should agree; averaging removes some random error.
It costs 4x the time and needs no retraining.

Command line:
    python -m medskel_ai.predict --image some_retina.png --out outputs/prediction
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import load_image
from .model import build_model

DEFAULT_CHECKPOINT = "runs/unet_fives/best.pt"


def load_model(checkpoint: str | Path = DEFAULT_CHECKPOINT, device: str | torch.device | None = None, flips: bool = False):
    """Load a trained model. Returns (model, device, threshold).

    The threshold comes from threshold.json (or threshold_flips.json) next to
    the checkpoint. It was chosen on validation images. Fallback: 0.5.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model = build_model(saved.get("arch", "plain"), saved.get("base", 32), pretrained=False).to(device)
    model.load_state_dict(saved["model"])
    model.eval()  # batch norm now uses its stored averages instead of batch statistics
    threshold_file = Path(checkpoint).parent / ("threshold_flips.json" if flips else "threshold.json")
    threshold = json.loads(threshold_file.read_text())["threshold"] if threshold_file.exists() else 0.5
    return model, device, threshold


def _tile_starts(length: int, tile: int, stride: int) -> list[int]:
    """Start positions so that tiles cover 0..length; the last tile is pushed back to fit."""
    starts = list(range(0, max(length - tile, 0) + 1, stride))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def _predict_tiled(model, x: torch.Tensor, device, tile: int, overlap: int, batch_size: int) -> torch.Tensor:
    """x: [3, H, W] float tensor on the CPU, H and W >= tile. Returns probabilities [H, W] on the device."""
    height, width = x.shape[1:]
    stride = tile - overlap
    positions = [(y, x0) for y in _tile_starts(height, tile, stride) for x0 in _tile_starts(width, tile, stride)]
    prob_sum = torch.zeros(height, width, device=device)  # sum of probabilities
    hits = torch.zeros(height, width, device=device)  # how many tiles covered each pixel
    use_amp = device.type == "cuda"
    for start in range(0, len(positions), batch_size):
        chunk = positions[start : start + batch_size]
        tiles = torch.stack([x[:, y : y + tile, x0 : x0 + tile] for y, x0 in chunk]).to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(tiles)  # [n, 1, tile, tile]
        probs = torch.sigmoid(logits.float())[:, 0]  # logits -> probabilities
        for (y, x0), p in zip(chunk, probs):
            prob_sum[y : y + tile, x0 : x0 + tile] += p
            hits[y : y + tile, x0 : x0 + tile] += 1
    return prob_sum / hits  # average where tiles overlap


@torch.no_grad()
def predict_probability(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    tile: int = 512,
    overlap: int = 64,
    batch_size: int = 4,
    flips: bool = False,
    as_tensor: bool = False,
):
    """Vessel probability for every pixel of one photograph.

    image : uint8 [H, W, 3] (R, G, B)
    flips : average over the 4 mirror versions of the image (slower, see top of file)
    return: float32 [H, W], values 0..1 (numpy array, or a tensor on the device if as_tensor=True)
    """
    height, width = image.shape[:2]
    x = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0  # [3, H, W]
    # If the image is smaller than one tile, pad it with black on the right/bottom.
    pad_h, pad_w = max(tile - height, 0), max(tile - width, 0)
    if pad_h or pad_w:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))

    # Each entry lists the axes to mirror: none, left-right (axis 2), up-down (axis 1), both.
    versions = [(), (2,), (1,), (1, 2)] if flips else [()]
    total = None
    for axes in versions:
        mirrored = torch.flip(x, dims=axes) if axes else x
        prob = _predict_tiled(model, mirrored, device, tile, overlap, batch_size)
        prob = torch.flip(prob, dims=[a - 1 for a in axes]) if axes else prob  # mirror the answer back
        total = prob if total is None else total + prob
    prob = (total / len(versions))[:height, :width]
    return prob if as_tensor else prob.cpu().numpy()


def make_overlay(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), strength: float = 0.7) -> np.ndarray:
    """Paint the predicted vessel pixels in colour on top of the photograph."""
    out = image.astype(np.float32).copy()
    selected = mask.astype(bool)
    out[selected] = (1 - strength) * out[selected] + strength * np.array(color, dtype=np.float32)
    return out.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment the vessels of one retinal photograph.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", default="outputs/prediction")
    parser.add_argument("--flips", action="store_true", help="average over mirrored versions (4x slower)")
    parser.add_argument("--threshold", type=float, default=None, help="default: the validation-chosen threshold")
    args = parser.parse_args()

    model, device, threshold = load_model(args.checkpoint, flips=args.flips)
    if args.threshold is not None:
        threshold = args.threshold
    image = load_image(args.image)
    prob = predict_probability(model, image, device, flips=args.flips)
    mask = prob >= threshold

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    Image.fromarray((prob * 255).astype(np.uint8)).save(out / f"{stem}_probability.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(out / f"{stem}_mask.png")
    Image.fromarray(make_overlay(image, mask)).save(out / f"{stem}_overlay.png")
    print(f"threshold {threshold:.2f}, vessel pixels {mask.mean() * 100:.2f}% -> saved 3 files in {out}")


if __name__ == "__main__":
    main()
