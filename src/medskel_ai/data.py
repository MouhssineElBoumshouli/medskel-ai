"""FIVES data: find image/mask pairs, make the train/validation split, cut crops.

Folder layout that this code expects (checked against the real download):

    FIVES/
      train/Original/1_A.png ...        600 colour photographs, 2048 x 2048
      train/Ground truth/1_A.png ...    600 expert vessel masks, same names
      test/Original/...                 200 photographs
      test/Ground truth/...             200 masks

The letter in the file name is the disease group:
A = AMD, D = diabetic retinopathy, G = glaucoma, N = normal.

Shapes used everywhere in this project:
    image tensor  [3, H, W]  float32, values 0..1, channel order R, G, B
    mask tensor   [1, H, W]  float32, values exactly 0 (background) or 1 (vessel)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

GROUPS = {"A": "AMD", "D": "diabetic retinopathy", "G": "glaucoma", "N": "normal"}


# --------------------------------------------------------------------------
# Finding files
# --------------------------------------------------------------------------
def _subdir(parent: Path, name: str) -> Path:
    """Find a sub-folder by name, ignoring upper/lower case ("Ground truth" vs "Ground Truth")."""
    if parent.is_dir():
        for child in parent.iterdir():
            if child.is_dir() and child.name.lower() == name.lower():
                return child
    raise FileNotFoundError(f"Folder '{name}' not found inside {parent}")


def _numeric_key(stem: str):
    """Sort '2_A' before '10_A' (plain text sorting would put '10_A' first)."""
    number = stem.split("_")[0]
    return (int(number) if number.isdigit() else 0, stem)


def list_pairs(root: str | Path, split: str) -> list[dict]:
    """Return every image/mask pair of one official split ('train' or 'test').

    A pair is only accepted when the image and the mask have exactly the same
    file name. If any file has no partner we stop with an error: training on a
    wrongly paired mask would be silent and very damaging.
    """
    split_dir = Path(root) / split
    image_dir = _subdir(split_dir, "Original")
    mask_dir = _subdir(split_dir, "Ground truth")
    images = {p.stem: p for p in image_dir.glob("*.png")}
    masks = {p.stem: p for p in mask_dir.glob("*.png")}
    if set(images) != set(masks):
        only_images = sorted(set(images) - set(masks))[:5]
        only_masks = sorted(set(masks) - set(images))[:5]
        raise ValueError(
            f"Image/mask names do not match in {split_dir}. "
            f"Images without mask: {only_images}. Masks without image: {only_masks}."
        )
    pairs = []
    for stem in sorted(images, key=_numeric_key):
        group = stem.split("_")[-1]
        pairs.append(
            {
                "id": f"{split}/{stem}",  # '1_A' exists in BOTH train and test, so keep the split in the id
                "split": split,
                "group": group if group in GROUPS else "?",
                "image": str(images[stem]),
                "mask": str(masks[stem]),
            }
        )
    return pairs


# Found by our audit of the real download (results/data_audit.json): these two
# training masks are completely black although the photographs show a retina.
# They are label errors, so we use them neither for training nor for validation.
EMPTY_TRAIN_MASKS = frozenset({"train/447_G", "train/448_G"})


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_split(pairs: list[dict], val_fraction: float = 0.15, seed: int = 42, exclude=EMPTY_TRAIN_MASKS):
    """Split the official TRAIN pairs into train and validation parts.

    - Only training images are used. The official test set is never touched here.
    - Stratified: we take the same share from each disease group (A, D, G, N).
    - Deterministic: the same seed always gives the same split.
    - Byte-identical photographs stay together. Our audit found the same photo
      stored under two names several times (for example 55_A and 134_A). If one
      copy were in train and the other in validation, the validation score
      would be too optimistic. So copies always go to the same part.

    Limit: FIVES has 800 images from 573 patients and publishes no patient ids.
    Two DIFFERENT photos of the same patient can still land in different parts.
    We cannot rule this out, and we say so in the report.
    """
    by_group: dict[str, dict[str, list[dict]]] = {}  # group -> file hash -> pairs with that hash
    for pair in pairs:
        if pair["id"] in exclude:
            continue
        by_group.setdefault(pair["group"], {}).setdefault(_sha256(pair["image"]), []).append(pair)
    rng = random.Random(seed)
    train, val = [], []
    for group in sorted(by_group):
        # A "unit" is one photo, or several identical copies of one photo.
        units = sorted(by_group[group].values(), key=lambda unit: _numeric_key(Path(unit[0]["image"]).stem))
        rng.shuffle(units)
        n_val = int(round(sum(len(unit) for unit in units) * val_fraction))
        taken = 0
        for unit in units:
            if taken < n_val:
                val.extend(unit)
                taken += len(unit)
            else:
                train.extend(unit)
    return train, val


# --------------------------------------------------------------------------
# Reading one file
# --------------------------------------------------------------------------
def load_image(path: str | Path) -> np.ndarray:
    """Read a photograph as uint8 array [H, W, 3] in R, G, B order."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def load_mask(path: str | Path, strict: bool = False) -> np.ndarray:
    """Read an expert mask as uint8 array [H, W] with values exactly 0 or 1.

    FIVES stores vessels as white (255) and background as black (0).
    With strict=True we refuse any other grey value instead of rounding it.
    """
    with Image.open(path) as im:
        grey = np.asarray(im.convert("L"))
    if strict:
        values = np.unique(grey)
        if not set(values.tolist()) <= {0, 255}:
            raise ValueError(f"Mask {path} is not binary, it contains values {values[:10]}")
    return (grey > 127).astype(np.uint8)


def resize_image_bilinear(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a photograph. Bilinear blending is fine for photographs."""
    return np.asarray(Image.fromarray(image).resize((width, height), Image.BILINEAR))


def resize_mask_nearest(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a 0/1 mask with NEAREST neighbour.

    Bilinear would blend 0 and 1 into values like 0.3, which are not labels.
    Nearest neighbour only copies existing pixels, so the result stays 0/1.
    """
    resized = Image.fromarray(mask.astype(np.uint8)).resize((width, height), Image.NEAREST)
    return np.asarray(resized)


# --------------------------------------------------------------------------
# Dataset of crops
# --------------------------------------------------------------------------
class FivesCrops(Dataset):
    """Cut square crops out of full-resolution FIVES images.

    We never shrink the photographs: thin vessels are 1-2 pixels wide and would
    disappear. Instead the network sees 512 x 512 windows at full resolution.

    train=True : random crop positions + random flips/rotations/brightness.
    train=False: four fixed crops around the image centre, no randomness, so
                 the validation score of two epochs can be compared fairly.

    One item = all crops of one photograph:
        images [K, 3, crop, crop], masks [K, 1, crop, crop], id (text)
    """

    def __init__(self, pairs: list[dict], crop: int = 512, crops_per_image: int = 2, train: bool = True):
        self.pairs = pairs
        self.crop = crop
        self.crops_per_image = crops_per_image
        self.train = train

    def __len__(self) -> int:
        return len(self.pairs)

    def _positions(self, height: int, width: int) -> list[tuple[int, int]]:
        c = self.crop
        if self.train:
            ys = torch.randint(0, height - c + 1, (self.crops_per_image,)).tolist()
            xs = torch.randint(0, width - c + 1, (self.crops_per_image,)).tolist()
            return list(zip(ys, xs))
        # Fixed 2 x 2 block of crops around the centre (clamped inside the image).
        ys = [min(max(height // 2 - c, 0), height - c), min(height // 2, height - c)]
        xs = [min(max(width // 2 - c, 0), width - c), min(width // 2, width - c)]
        return [(y, x) for y in ys for x in xs]

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        image = load_image(pair["image"])  # [H, W, 3] uint8
        mask = load_mask(pair["mask"])  # [H, W]    uint8 0/1
        if image.shape[:2] != mask.shape:
            raise ValueError(f"{pair['id']}: image {image.shape[:2]} and mask {mask.shape} differ in size")
        height, width = mask.shape
        if height < self.crop or width < self.crop:
            raise ValueError(f"{pair['id']}: image is smaller than the crop size {self.crop}")

        images, masks = [], []
        for y, x in self._positions(height, width):
            img = image[y : y + self.crop, x : x + self.crop]
            msk = mask[y : y + self.crop, x : x + self.crop]
            if self.train:
                img, msk = _augment_geometry(img, msk)
            img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
            msk_t = torch.from_numpy(np.ascontiguousarray(msk)).unsqueeze(0).float()
            if self.train:
                img_t = _augment_brightness(img_t)
            images.append(img_t)
            masks.append(msk_t)
        return torch.stack(images), torch.stack(masks), pair["id"]


def _augment_geometry(image: np.ndarray, mask: np.ndarray):
    """Flip / rotate the image AND the mask in exactly the same way, so they stay aligned."""
    if torch.rand(1).item() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]  # left-right flip
    if torch.rand(1).item() < 0.5:
        image, mask = image[::-1], mask[::-1]  # up-down flip
    k = int(torch.randint(0, 4, (1,)).item())  # 0, 90, 180 or 270 degrees
    return np.rot90(image, k), np.rot90(mask, k)


def _augment_brightness(image: torch.Tensor) -> torch.Tensor:
    """Small brightness/contrast change. Only for the image: the mask does not change with lighting."""
    contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.4  # 0.8 .. 1.2
    brightness = (torch.rand(1).item() - 0.5) * 0.2  # -0.1 .. 0.1
    return (image * contrast + brightness).clamp(0.0, 1.0)


# --------------------------------------------------------------------------
# Audit of the real files (run once, writes a small JSON report)
# --------------------------------------------------------------------------
def audit(root: str | Path) -> dict:
    """Check every file: counts, sizes, binary masks, and identical files across splits."""
    report: dict = {"root": str(root), "splits": {}}
    hashes: dict[str, list[str]] = {}
    for split in ("train", "test"):
        pairs = list_pairs(root, split)
        sizes, nonbinary, size_mismatch, groups, empty = {}, [], [], {}, []
        fractions: dict[str, list[float]] = {}
        for pair in pairs:
            groups[pair["group"]] = groups.get(pair["group"], 0) + 1
            hashes.setdefault(_sha256(pair["image"]), []).append(pair["id"])
            with Image.open(pair["image"]) as im:
                image_size = im.size
            with Image.open(pair["mask"]) as im:
                mask_size = im.size
                grey = np.asarray(im.convert("L"))
            sizes[str(image_size)] = sizes.get(str(image_size), 0) + 1
            if image_size != mask_size:
                size_mismatch.append(pair["id"])
            if not set(np.unique(grey).tolist()) <= {0, 255}:
                nonbinary.append(pair["id"])
            if grey.max() == 0:
                empty.append(pair["id"])
            if split == "train":  # label statistics only from training data
                fractions.setdefault(pair["group"], []).append(float((grey > 127).mean()))
        entry = {
            "pairs": len(pairs),
            "groups": groups,
            "image_sizes": sizes,
            "image_mask_size_mismatch": size_mismatch,
            "masks_with_values_other_than_0_255": nonbinary,
            "completely_black_masks": empty,
        }
        if split == "train":
            everything = [f for values in fractions.values() for f in values]
            entry["vessel_pixel_fraction_mean"] = float(np.mean(everything))
            entry["vessel_pixel_fraction_min"] = float(np.min(everything))
            entry["vessel_pixel_fraction_max"] = float(np.max(everything))
            entry["vessel_pixel_fraction_by_group"] = {g: float(np.mean(v)) for g, v in sorted(fractions.items())}
        report["splits"][split] = entry
    report["identical_image_files"] = [ids for ids in hashes.values() if len(ids) > 1]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the FIVES files and write the split manifest.")
    parser.add_argument("--data", default=r"C:\data\FIVES")
    parser.add_argument("--out", default="results/data_audit.json")
    parser.add_argument("--split-out", default="results/split_seed42.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train, val = make_split(list_pairs(args.data, "train"), seed=args.seed)
    Path(args.split_out).parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": args.seed,
        "note": "Validation comes only from the official training set. Identical image files are kept in the same part. No patient ids are published for FIVES.",
        "excluded_empty_masks": sorted(EMPTY_TRAIN_MASKS),
        "train": [p["id"] for p in train],
        "val": [p["id"] for p in val],
    }
    Path(args.split_out).write_text(json.dumps(manifest, indent=1))
    print(f"split: {len(train)} train / {len(val)} validation -> {args.split_out}")

    report = audit(args.data)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
