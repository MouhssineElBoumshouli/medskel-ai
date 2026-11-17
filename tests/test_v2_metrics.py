"""Tests for the v2 additions: clDice, thin vessels, flips, bootstrap, soft clDice loss, CHASE helpers. SYNTHETIC inputs only."""

import numpy as np
import torch
from PIL import Image

from medskel_ai.chase import field_of_view, fov_width, list_chase
from medskel_ai.evaluate import bootstrap_ci, image_metrics
from medskel_ai.model import UNet
from medskel_ai.predict import predict_probability
from medskel_ai.topology import skeleton, thin_vessel_mask, topology_metrics
from medskel_ai.train import soft_cldice_loss, soft_skeleton


def two_vessels():
    """One thick vessel (9 px wide) and one thin vessel (1 px wide), both horizontal, 80 px long."""
    truth = np.zeros((60, 100), dtype=bool)
    truth[10:19, 10:90] = True  # thick: 720 pixels
    truth[40, 10:90] = True  # thin: 80 pixels
    return truth


def test_thin_mask_separates_thin_from_thick():
    truth = two_vessels()
    thin = thin_vessel_mask(truth, skeleton(truth))
    assert thin[40, 10:90].all()  # the 1-px vessel is thin
    assert not thin[10:19, 30:70].any()  # the middle of the 9-px vessel is not


def test_width_limit_is_three_pixels():
    for width, expected_thin in ((1, True), (3, True), (5, False), (9, False)):
        truth = np.zeros((40, 100), dtype=bool)
        truth[10 : 10 + width, 10:90] = True
        thin = thin_vessel_mask(truth, skeleton(truth))
        assert bool(thin[10, 50]) == expected_thin, width


def test_dice_hides_a_lost_thin_vessel_but_cldice_and_thin_recall_show_it():
    truth = two_vessels()
    pred = truth.copy()
    pred[40, :] = False  # the model misses the whole thin vessel
    m = image_metrics(pred, truth)
    assert m["dice"] > 0.94  # Dice still looks great: only 80 of 800 pixels are gone
    assert m["thin_recall"] == 0.0  # but every thin pixel was missed
    assert m["thick_recall"] == 1.0
    assert m["cl_sensitivity"] < 0.6  # and about half of the centre-line length is lost
    assert m["cldice"] < 0.75


def test_perfect_prediction_scores_one():
    truth = two_vessels()
    m = topology_metrics(truth, truth)
    assert m["cldice"] == 1.0 and m["thin_recall"] == 1.0


def test_a_broken_vessel_lowers_cl_sensitivity():
    truth = two_vessels()
    pred = truth.copy()
    pred[:, 45:55] = False  # cut both vessels: a 10 px gap
    m = topology_metrics(pred, truth)
    assert 0.8 < m["cl_sensitivity"] < 0.95


def test_flips_on_a_symmetric_input_give_a_symmetric_answer():
    torch.manual_seed(0)
    model = UNet(base=8).eval()
    image = np.random.default_rng(0).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    plain = predict_probability(model, image, torch.device("cpu"), tile=64, overlap=16)
    averaged = predict_probability(model, image, torch.device("cpu"), tile=64, overlap=16, flips=True)
    mirrored = predict_probability(model, image[:, ::-1], torch.device("cpu"), tile=64, overlap=16, flips=True)
    assert averaged.shape == plain.shape == (64, 64)
    # Averaging over all mirrors makes the result independent of mirroring the input.
    assert np.allclose(averaged, mirrored[:, ::-1], atol=1e-5)


def test_bootstrap_interval_contains_the_mean_and_shrinks_with_more_data():
    rng = np.random.default_rng(0)
    small, large = rng.normal(0.8, 0.1, 20), rng.normal(0.8, 0.1, 500)
    low_s, high_s = bootstrap_ci(small, n_boot=2000)
    low_l, high_l = bootstrap_ci(large, n_boot=2000)
    assert low_s < small.mean() < high_s
    assert (high_l - low_l) < (high_s - low_s)


def test_cluster_bootstrap_is_wider_when_pairs_are_identical():
    values = np.repeat(np.random.default_rng(0).normal(0.8, 0.1, 14), 2)  # 14 "children", two identical "eyes"
    children = np.repeat(np.arange(14), 2)
    low_i, high_i = bootstrap_ci(values, n_boot=4000)
    low_c, high_c = bootstrap_ci(values, clusters=children, n_boot=4000)
    assert (high_c - low_c) > (high_i - low_i)


def test_bootstrap_ignores_nan():
    low, high = bootstrap_ci([0.5, 0.7, float("nan"), 0.6], n_boot=500)
    assert 0.5 <= low <= high <= 0.7


def test_soft_skeleton_keeps_thin_lines_and_thins_thick_ones():
    truth = torch.from_numpy(two_vessels()).float()[None, None]
    skel = soft_skeleton(truth, iterations=10)[0, 0]
    assert skel[40, 20:80].min() > 0.9  # the thin line is its own skeleton
    assert skel[10:19, 20:80].sum() < 0.3 * truth[0, 0, 10:19, 20:80].sum()  # the thick one became much thinner


def test_soft_cldice_loss_extremes_and_gradient():
    truth = torch.from_numpy(two_vessels()).float()[None, None]
    perfect = (truth * 2 - 1) * 20
    assert soft_cldice_loss(perfect, truth) < 0.05
    missing_thin = perfect.clone()
    missing_thin[..., 40, :] = -20
    assert soft_cldice_loss(missing_thin, truth) > 0.2
    logits = torch.zeros_like(truth, requires_grad=True)
    soft_cldice_loss(logits, truth).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_chase_listing_and_field_of_view(tmp_path):
    yy, xx = np.mgrid[:80, :100]
    disc = (yy - 40) ** 2 + (xx - 50) ** 2 < 38**2
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[disc] = (150, 80, 40)
    image[40, 45:55] = 0  # a dark spot inside the retina must stay inside the field of view
    for code in ("01L", "01R", "02L"):
        Image.fromarray(image).save(tmp_path / f"Image_{code}.jpg", quality=95)
        for observer in ("1stHO", "2ndHO"):
            Image.fromarray(np.zeros((80, 100), dtype=np.uint8)).save(tmp_path / f"Image_{code}_{observer}.png")
    items = list_chase(tmp_path)
    assert [i["id"] for i in items] == ["chase/01L", "chase/01R", "chase/02L"]
    assert [i["child"] for i in items] == ["01", "01", "02"]
    fov = field_of_view(np.asarray(Image.open(items[0]["image"]).convert("RGB")))
    assert fov[40, 50] and not fov[2, 2]
    assert 70 <= fov_width(fov) <= 80
