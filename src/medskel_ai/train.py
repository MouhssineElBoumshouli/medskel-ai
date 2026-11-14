"""Train the U-Net on FIVES crops.

    python -m medskel_ai.train --minutes 50

What one training step does:
    1. forward : crops -> U-Net -> logits
    2. loss    : compare logits with the expert mask (BCE + soft Dice)
    3. backward: compute how each weight should change to lower the loss
    4. step    : Adam changes the weights a little

After each epoch (one pass over the training photographs) we measure the Dice
score on fixed validation crops and keep the weights with the best score.
The official test set is never opened by this file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import FivesCrops, list_pairs, make_split
from .model import build_model, count_parameters


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """1 - Dice, computed with probabilities instead of hard 0/1 predictions.

    Hard predictions (p >= 0.5) have no useful gradient. Probabilities do, so
    this version can be minimised by gradient descent. eps avoids 0/0.
    """
    p = torch.sigmoid(logits)
    intersection = (p * target).sum()
    return 1 - (2 * intersection + eps) / (p.sum() + target.sum() + eps)


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    """Shrink bright shapes by one pixel: each pixel becomes the MINIMUM of itself and its 4 neighbours."""
    vertical = -nn.functional.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0))
    horizontal = -nn.functional.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1))
    return torch.min(vertical, horizontal)


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    """Erode, then grow back (3x3 maximum). Thin lines do not survive this; thick shapes do."""
    return nn.functional.max_pool2d(_soft_erode(x), 3, stride=1, padding=1)


def soft_skeleton(x: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """A differentiable "centre line" of a probability map (Shit et al., clDice, CVPR 2021).

    A real skeleton algorithm makes hard yes/no decisions, so it has no gradient.
    This version uses only min and max over neighbours, which do have gradients.
    Idea: "x minus opened x" keeps what is too thin to survive an opening, i.e.
    the ridge. We shrink the shape step by step and collect the ridge each time.
    `iterations` must be at least the half-width of the widest vessel in pixels.
    """
    skeleton = torch.relu(x - _soft_open(x))
    for _ in range(iterations):
        x = _soft_erode(x)
        ridge = torch.relu(x - _soft_open(x))
        skeleton = skeleton + torch.relu(ridge - skeleton * ridge)
    return skeleton


def soft_cldice_loss(logits: torch.Tensor, target: torch.Tensor, iterations: int = 10, eps: float = 1.0) -> torch.Tensor:
    """1 - soft clDice. Punishes broken or missing centre lines, whatever the vessel width."""
    p = torch.sigmoid(logits)
    skeleton_pred, skeleton_true = soft_skeleton(p, iterations), soft_skeleton(target, iterations)
    t_prec = ((skeleton_pred * target).sum() + eps) / (skeleton_pred.sum() + eps)  # predicted centre line inside the true mask
    t_sens = ((skeleton_true * p).sum() + eps) / (skeleton_true.sum() + eps)  # true centre line inside the prediction
    return 1 - 2 * t_prec * t_sens / (t_prec + t_sens)


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, cldice_weight: float = 0.0) -> torch.Tensor:
    """BCE looks at every pixel on its own. Dice looks at the overlap of the whole mask.

    Vessels are only a small share of the pixels. BCE alone can get lazy and
    favour background; Dice ignores true negatives, so it pushes for vessels.
    Optional third term: soft clDice (connectivity), multiplied by cldice_weight.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss = bce + soft_dice_loss(logits, target)
    if cldice_weight > 0:
        loss = loss + cldice_weight * soft_cldice_loss(logits, target)
    return loss


def hard_dice(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Dice of the thresholded prediction (probability >= 0.5, i.e. logit >= 0)."""
    pred = logits >= 0
    truth = target >= 0.5
    tp = (pred & truth).sum().item()
    total = pred.sum().item() + truth.sum().item()
    return 2 * tp / total if total > 0 else 1.0


@torch.no_grad()
def validate(model, loader, device, use_amp) -> tuple[float, float]:
    """Mean loss and mean per-image Dice on the fixed validation crops."""
    model.eval()
    losses, dices = [], []
    for images, masks, _ in loader:
        images = images.flatten(0, 1).to(device)  # [1, K, 3, c, c] -> [K, 3, c, c]
        masks = masks.flatten(0, 1).to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        logits = logits.float()
        losses.append(bce_dice_loss(logits, masks).item())
        dices.append(hard_dice(logits, masks))
    return float(np.mean(losses)), float(np.mean(dices))


def save_curves(history: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.plot(epochs, [h["train_loss"] for h in history], label="train loss")
    left.plot(epochs, [h["val_loss"] for h in history], label="validation loss")
    left.set_xlabel("epoch"), left.set_ylabel("BCE + soft Dice loss"), left.legend(), left.grid(alpha=0.3)
    right.plot(epochs, [h["val_dice"] for h in history], color="green", label="validation Dice (fixed crops)")
    right.set_xlabel("epoch"), right.set_ylabel("Dice"), right.legend(), right.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the U-Net on FIVES.")
    parser.add_argument("--data", default=r"C:\data\FIVES")
    parser.add_argument("--out", default="runs/unet_fives")
    parser.add_argument("--minutes", type=float, default=50.0, help="training time budget")
    parser.add_argument("--max-epochs", type=int, default=0, help="optional hard limit (0 = only the time budget)")
    parser.add_argument("--limit", type=int, default=0, help="use only N train and N val images (smoke test)")
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--batch-images", type=int, default=2)
    parser.add_argument("--crops-per-image", type=int, default=2)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--arch", default="plain", choices=["plain", "resnet34"], help="plain = our U-Net; resnet34 = pretrained comparison model")
    parser.add_argument("--cldice-weight", type=float, default=0.0, help="weight of the soft clDice loss term (0 = off)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="continue from last.pt in --out")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"  # mixed precision: 16-bit floats where safe -> less memory, faster
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_pairs, val_pairs = make_split(list_pairs(args.data, "train"), seed=args.seed)
    assert not {p["id"] for p in train_pairs} & {p["id"] for p in val_pairs}, "train and validation overlap"
    if args.limit:
        train_pairs, val_pairs = train_pairs[: args.limit], val_pairs[: args.limit]

    train_set = FivesCrops(train_pairs, crop=args.crop, crops_per_image=args.crops_per_image, train=True)
    val_set = FivesCrops(val_pairs, crop=args.crop, train=False)
    workers = dict(num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=use_amp)
    train_loader = DataLoader(train_set, batch_size=args.batch_images, shuffle=True, drop_last=True, **workers)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **workers)

    model = build_model(args.arch, args.base).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    history: list[dict] = []
    best_dice, elapsed_before = -1.0, 0.0
    if args.resume and (out / "last.pt").exists():
        saved = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best_dice, elapsed_before = saved["history"], saved["best_dice"], saved["elapsed"]
        print(f"resumed after epoch {len(history)} ({elapsed_before / 60:.1f} min done)", flush=True)

    config = vars(args) | {
        "parameters": count_parameters(model),
        "train_images": len(train_pairs),
        "val_images": len(val_pairs),
        "loss": "BCEWithLogits + soft Dice" + (f" + {args.cldice_weight} * soft clDice" if args.cldice_weight > 0 else ""),
        "optimizer": "Adam",
        "lr_schedule": "cosine from lr to 0 over the time budget",
        "mixed_precision": use_amp,
        "device": torch.cuda.get_device_name(0) if use_amp else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    (out / "config.json").write_text(json.dumps(config, indent=1))
    print(json.dumps(config, indent=1), flush=True)

    budget = args.minutes * 60
    start = time.time()
    epoch = len(history)
    while True:
        elapsed = elapsed_before + time.time() - start
        if elapsed >= budget or (args.max_epochs and epoch >= args.max_epochs):
            break
        epoch += 1
        model.train()
        losses = []
        for images, masks, _ in train_loader:
            # Learning rate follows half a cosine wave: big steps first, tiny steps at the end.
            progress = min((elapsed_before + time.time() - start) / budget, 1.0)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = lr

            images = images.flatten(0, 1).to(device, non_blocking=True)  # [2, 2, 3, c, c] -> [4, 3, c, c]
            masks = masks.flatten(0, 1).to(device, non_blocking=True)  #   [2, 2, 1, c, c] -> [4, 1, c, c]

            optimizer.zero_grad(set_to_none=True)  # forget the gradients of the previous step
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)  # forward pass
            loss = bce_dice_loss(logits.float(), masks, args.cldice_weight)  # loss in full 32-bit precision
            scaler.scale(loss).backward()  # backpropagation: d loss / d every weight
            scaler.step(optimizer)  # Adam update
            scaler.update()
            losses.append(loss.item())

        val_loss, val_dice = validate(model, val_loader, device, use_amp)
        elapsed = elapsed_before + time.time() - start
        row = {
            "epoch": epoch,
            "minutes": round(elapsed / 60, 2),
            "lr": lr,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "val_dice": val_dice,
        }
        history.append(row)
        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice
            torch.save({"model": model.state_dict(), "base": args.base, "arch": args.arch, "epoch": epoch, "val_dice": val_dice}, out / "best.pt")
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history, "best_dice": best_dice, "elapsed": elapsed},
            out / "last.pt",
        )
        with open(out / "history.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(history)
        save_curves(history, out / "curves.png")
        memory = f" gpu_peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GB" if use_amp else ""
        print(
            f"epoch {epoch:3d} | {row['minutes']:6.1f} min | lr {lr:.2e} | train loss {row['train_loss']:.4f} | "
            f"val loss {val_loss:.4f} | val Dice {val_dice:.4f}{' *best*' if is_best else ''}{memory}",
            flush=True,
        )

    print(f"finished: {epoch} epochs, best validation Dice (fixed crops, threshold 0.5) = {best_dice:.4f}", flush=True)


if __name__ == "__main__":
    main()
