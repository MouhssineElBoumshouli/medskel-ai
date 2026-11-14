"""SYNTHETIC fixtures only. Nothing here is a real retina or a real FIVES file.

We build a tiny fake dataset with the same folder layout and file naming as
FIVES, so the loader code can be tested without the real data.
"""

import numpy as np
import pytest
from PIL import Image

SIZE = 96  # fake images are 96 x 96 (real FIVES images are 2048 x 2048)


def synthetic_pair(seed: int, size: int = SIZE):
    """A dark image with a few bright straight 'vessels' and the matching 0/1 mask."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((size, size), dtype=np.uint8)
    for _ in range(3):
        row, col = rng.integers(8, size - 8, size=2)
        mask[row : row + 2, :] = 1  # horizontal line, 2 pixels thick
        mask[:, col : col + 2] = 1  # vertical line
    image = rng.integers(20, 60, size=(size, size, 3), dtype=np.uint8)
    image[mask == 1] += 120  # vessels are brighter than the background
    return image, mask


@pytest.fixture
def fake_fives(tmp_path):
    """tmp/FIVES with 40 train pairs (10 per group) and 8 test pairs (2 per group)."""
    root = tmp_path / "FIVES"
    seed = 0
    for split, per_group in (("train", 10), ("test", 2)):
        (root / split / "Original").mkdir(parents=True)
        (root / split / "Ground truth").mkdir(parents=True)
        number = 0
        for group in "ADGN":
            for _ in range(per_group):
                number += 1
                seed += 1
                image, mask = synthetic_pair(seed)
                Image.fromarray(image).save(root / split / "Original" / f"{number}_{group}.png")
                Image.fromarray(mask * 255).save(root / split / "Ground truth" / f"{number}_{group}.png")
    return root
