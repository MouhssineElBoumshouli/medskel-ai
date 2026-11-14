"""Data tests on SYNTHETIC fixtures (see conftest.py)."""

import numpy as np
import pytest
import torch
from PIL import Image

from medskel_ai.data import FivesCrops, list_pairs, load_mask, make_split, resize_mask_nearest


def test_pairs_have_matching_names(fake_fives):
    pairs = list_pairs(fake_fives, "train")
    assert len(pairs) == 40
    for pair in pairs:
        image_name = pair["image"].replace("\\", "/").split("/")[-1]
        mask_name = pair["mask"].replace("\\", "/").split("/")[-1]
        assert image_name == mask_name
        assert "Original" in pair["image"] and "Ground truth" in pair["mask"]
        assert pair["group"] in "ADGN"


def test_missing_mask_is_an_error(fake_fives):
    (fake_fives / "train" / "Ground truth" / "1_A.png").unlink()
    with pytest.raises(ValueError, match="do not match"):
        list_pairs(fake_fives, "train")


def test_split_has_no_overlap_and_is_stratified(fake_fives):
    pairs = list_pairs(fake_fives, "train")
    train, val = make_split(pairs, val_fraction=0.2, seed=42)
    train_ids, val_ids = {p["id"] for p in train}, {p["id"] for p in val}
    assert not train_ids & val_ids  # no image in both parts
    assert train_ids | val_ids == {p["id"] for p in pairs}  # nothing lost
    for group in "ADGN":  # 20% of 10 = 2 validation images per group
        assert sum(p["group"] == group for p in val) == 2


def test_split_is_deterministic_and_never_contains_test(fake_fives):
    pairs = list_pairs(fake_fives, "train")
    first = make_split(pairs, seed=42)
    second = make_split(pairs, seed=42)
    assert [p["id"] for p in first[1]] == [p["id"] for p in second[1]]
    test_files = {p["image"] for p in list_pairs(fake_fives, "test")}
    assert not test_files & {p["image"] for p in first[0] + first[1]}


def test_identical_photos_stay_in_the_same_part(fake_fives):
    """The same photo saved under two names must not be split between train and validation."""
    original = fake_fives / "train" / "Original"
    for seed in range(20):
        (original / "2_A.png").write_bytes((original / "1_A.png").read_bytes())  # 2_A is now a copy of 1_A
        train, val = make_split(list_pairs(fake_fives, "train"), val_fraction=0.3, seed=seed)
        val_ids = {p["id"] for p in val}
        assert ("train/1_A" in val_ids) == ("train/2_A" in val_ids)
        assert len(train) + len(val) == 40


def test_known_empty_masks_are_excluded(fake_fives):
    pairs = list_pairs(fake_fives, "train")
    train, val = make_split(pairs, exclude={"train/3_A"})
    assert "train/3_A" not in {p["id"] for p in train + val}
    assert len(train) + len(val) == 39


def test_mask_is_exactly_zero_or_one(fake_fives):
    pair = list_pairs(fake_fives, "train")[0]
    mask = load_mask(pair["mask"], strict=True)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 1}
    assert 0 < mask.mean() < 0.5  # vessels are the minority; an inverted mask would fail here


def test_strict_mask_rejects_grey_values(tmp_path):
    grey = np.full((8, 8), 100, dtype=np.uint8)
    Image.fromarray(grey).save(tmp_path / "bad.png")
    with pytest.raises(ValueError, match="not binary"):
        load_mask(tmp_path / "bad.png", strict=True)


def test_nearest_resize_keeps_labels_binary():
    mask = (np.random.default_rng(0).random((64, 64)) > 0.7).astype(np.uint8)
    resized = resize_mask_nearest(mask, 37, 51)
    assert resized.shape == (51, 37)
    assert set(np.unique(resized).tolist()) <= {0, 1}


@pytest.mark.parametrize("train", [True, False])
def test_crop_shapes_types_and_values(fake_fives, train):
    pairs = list_pairs(fake_fives, "train")
    dataset = FivesCrops(pairs, crop=32, crops_per_image=2, train=train)
    images, masks, image_id = dataset[0]
    k = 2 if train else 4  # validation always gives 4 fixed crops
    assert images.shape == (k, 3, 32, 32) and images.dtype == torch.float32
    assert masks.shape == (k, 1, 32, 32) and masks.dtype == torch.float32
    assert 0.0 <= images.min() and images.max() <= 1.0
    assert set(masks.unique().tolist()) <= {0.0, 1.0}
    assert image_id == "train/1_A"


def test_validation_crops_are_always_the_same(fake_fives):
    dataset = FivesCrops(list_pairs(fake_fives, "train"), crop=32, train=False)
    assert torch.equal(dataset[3][0], dataset[3][0])
    assert torch.equal(dataset[3][1], dataset[3][1])


def test_augmented_image_and_mask_stay_aligned(fake_fives):
    """In the fake data vessels are bright. After flips/rotations the bright pixels must still be the mask pixels."""
    torch.manual_seed(0)
    dataset = FivesCrops(list_pairs(fake_fives, "train"), crop=48, crops_per_image=4, train=True)
    for index in range(5):
        images, masks, _ = dataset[index]
        brightness = images.mean(dim=1, keepdim=True)  # [K, 1, 48, 48]
        for b, m in zip(brightness, masks):  # each crop has its own brightness change
            if m.sum() > 0:  # a crop can contain no vessel at all
                assert b[m == 1].min() > b[m == 0].max()
