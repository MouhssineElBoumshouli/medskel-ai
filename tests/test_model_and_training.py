"""Model, loss, metric and prediction tests. All inputs are SYNTHETIC."""

import numpy as np
import torch

from conftest import synthetic_pair
from medskel_ai.evaluate import confusion_counts, metrics_from_counts
from medskel_ai.model import UNet, count_parameters
from medskel_ai.predict import _tile_starts, make_overlay, predict_probability
from medskel_ai.train import bce_dice_loss, soft_dice_loss


def test_output_shape_matches_input():
    model = UNet(base=8)
    out = model(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 1, 64, 64)
    assert torch.isfinite(out).all()


def test_odd_input_size_still_works():
    # 70 is not divisible by 16; the decoder upsamples to the exact skip size.
    assert UNet(base=8)(torch.rand(1, 3, 70, 90)).shape == (1, 1, 70, 90)


def test_parameter_count_of_the_real_model():
    # First conv of the real model: 3 input channels * 32 filters * 3*3 weights = 864.
    model = UNet(base=32)
    assert model.enc1.block[0].weight.numel() == 3 * 32 * 3 * 3
    assert 7_000_000 < count_parameters(model) < 9_000_000


def test_soft_dice_loss_extremes():
    target = torch.zeros(1, 1, 8, 8)
    target[..., :4] = 1
    perfect = (target * 2 - 1) * 20  # logits +20 on vessels, -20 elsewhere
    assert soft_dice_loss(perfect, target) < 0.01
    assert soft_dice_loss(-perfect, target) > 0.9


def test_metrics_worked_example():
    # truth:      1 1 1 1 0 0 0 0 0 0
    # prediction: 1 1 1 0 1 0 0 0 0 0   -> TP=3, FP=1, FN=1, TN=5
    truth = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]).bool()
    pred = torch.tensor([1, 1, 1, 0, 1, 0, 0, 0, 0, 0]).bool()
    counts = confusion_counts(pred, truth)
    assert counts == (3, 1, 1, 5)
    m = metrics_from_counts(*counts)
    assert abs(m["dice"] - 0.75) < 1e-9
    assert abs(m["iou"] - 0.6) < 1e-9
    assert abs(m["precision"] - 0.75) < 1e-9
    assert abs(m["recall"] - 0.75) < 1e-9
    assert abs(m["specificity"] - 5 / 6) < 1e-9
    assert abs(m["accuracy"] - 0.8) < 1e-9


def test_all_background_prediction_has_high_accuracy_but_zero_dice():
    # 5% vessels: predicting "nothing" is 95% accurate and still useless.
    m = metrics_from_counts(tp=0, fp=0, fn=5, tn=95)
    assert m["accuracy"] == 0.95 and m["dice"] == 0.0 and m["recall"] == 0.0


def test_tiles_cover_the_whole_image():
    for length in (512, 600, 2048):
        starts = _tile_starts(length, 512, 448)
        covered = np.zeros(length, dtype=bool)
        for s in starts:
            assert 0 <= s and s + 512 <= length
            covered[s : s + 512] = True
        assert covered.all()


def test_tiled_prediction_shape_and_range():
    model = UNet(base=8).eval()
    image = np.random.default_rng(0).integers(0, 255, size=(100, 150, 3), dtype=np.uint8)
    prob = predict_probability(model, image, torch.device("cpu"), tile=64, overlap=16)
    assert prob.shape == (100, 150)
    assert prob.min() >= 0.0 and prob.max() <= 1.0
    overlay = make_overlay(image, prob >= 0.5)
    assert overlay.shape == image.shape and overlay.dtype == np.uint8


def test_overfit_one_batch():
    """The basic 'can it learn at all' check: memorise 2 synthetic images.

    This proves the plumbing (forward, loss, backward, update) works.
    It says NOTHING about how well the model generalises.
    """
    torch.manual_seed(0)
    pairs = [synthetic_pair(seed, size=64) for seed in (1, 2)]
    images = torch.stack([torch.from_numpy(i).permute(2, 0, 1).float() / 255 for i, _ in pairs])
    masks = torch.stack([torch.from_numpy(m).unsqueeze(0).float() for _, m in pairs])

    model = UNet(base=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    first_loss = None
    for _ in range(200):
        optimizer.zero_grad()
        loss = bce_dice_loss(model(images), masks)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        first_loss = first_loss or loss.item()
    assert loss.item() < 0.3 * first_loss  # the loss fell by more than 70%

    model.eval()
    with torch.no_grad():
        pred = model(images) >= 0
    tp = (pred & (masks == 1)).sum().item()
    dice = 2 * tp / (pred.sum().item() + masks.sum().item())
    assert dice > 0.9
