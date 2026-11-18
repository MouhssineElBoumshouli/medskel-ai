"""Tests for the measurement step. SYNTHETIC shapes only, with answers known in advance.

The MedSkel tests need a clone of the original medskel repository next to this
one (or MEDSKEL_REPO). Without it they are skipped, not failed.
"""

import numpy as np
import pytest

from medskel_ai import morphometry as mm
from medskel_ai import study


def tube(length=300, radius=6, size=(80, 400), angle=0.0):
    """A straight 'vessel': all pixels within `radius` of a horizontal segment."""
    yy, xx = np.mgrid[: size[0], : size[1]]
    cy, x0 = size[0] / 2, (size[1] - length) / 2
    x = np.clip(xx, x0, x0 + length)
    return np.hypot(xx - x, yy - cy) <= radius


def upstream_available():
    try:
        mm.import_upstream()
        return True
    except FileNotFoundError:
        return False


needs_upstream = pytest.mark.skipif(not upstream_available(), reason="original medskel repo not cloned next to this one")


# ---- measurements that need only the mask -----------------------------------

def test_vessel_density_counts_only_inside_the_field_of_view():
    mask = np.zeros((10, 10), bool)
    mask[:, :2] = True  # 20 vessel pixels
    fov = np.zeros((10, 10), bool)
    fov[:, :5] = True  # 50 FOV pixels, all 20 vessel pixels inside
    assert mm.vessel_density(mask, fov) == pytest.approx(0.4)
    assert mm.vessel_density(mask) == pytest.approx(0.2)


def test_fractal_dimension_of_a_line_is_1_and_of_a_square_is_2():
    line = np.zeros((512, 512), bool)
    line[256, :] = True
    square = np.ones((512, 512), bool)
    assert mm.fractal_dimension(line) == pytest.approx(1.0, abs=0.05)
    assert mm.fractal_dimension(square) == pytest.approx(2.0, abs=0.05)
    assert np.isnan(mm.fractal_dimension(np.zeros((64, 64), bool)))


def test_relative_difference():
    assert mm.relative_difference(90, 110) == pytest.approx(0.2)
    assert mm.relative_difference(5, 5) == 0
    assert np.isnan(mm.relative_difference(0, 0))
    assert np.isnan(mm.relative_difference(float("nan"), 1))


# ---- skeleton measurements (need the original MedSkel code) ------------------

@needs_upstream
@pytest.mark.parametrize("method", mm.METHODS)
def test_straight_tube_has_known_length_width_and_tortuosity(method):
    mask = tube(length=300, radius=6)
    r = mm.measure(mask, method)
    # A skeleton cannot reach the round end caps exactly, so allow about one radius at each end.
    assert 300 - 2 * 6 - 4 <= r["total_length"] <= 300 + 2 * 6 + 4
    assert r["median_calibre"] == pytest.approx(2 * 6, abs=3)
    assert r["median_tortuosity"] == pytest.approx(1.0, abs=0.05)
    assert r["n_bifurcations"] == 0


@needs_upstream
@pytest.mark.parametrize("method", mm.METHODS)
def test_vertical_tube_width_is_right_too(method):
    """Guards against mixing up x (column) and y (row) when sampling the width."""
    mask = tube(length=300, radius=8).T
    assert mm.measure(mask, method)["median_calibre"] == pytest.approx(16, abs=3)


@needs_upstream
def test_original_medskel_width_is_biased_so_we_recompute_it():
    """Documents the bug we work around: MedSkel's own branch width uses the tips."""
    _, voronoi, metrics = mm.import_upstream()
    mask = tube(length=300, radius=6)
    sk = voronoi.skeletonize(mask, epsilon=mm.MEDSKEL_EPSILON, prune=mm.MEDSKEL_PRUNE)
    assert metrics.vessel_summary(sk, mask)["median_calibre"] < 6  # far below the true 12
    assert mm.medskel_median_calibre(sk, mask) == pytest.approx(12, abs=3)


@needs_upstream
@pytest.mark.parametrize("method", mm.METHODS)
def test_empty_mask_gives_zero_length_not_a_crash(method):
    r = mm.measure(np.zeros((50, 50), bool), method)
    assert r["total_length"] == 0 and np.isnan(r["median_calibre"])


def test_missing_upstream_gives_a_clear_message(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "_upstream", None)
    with pytest.raises(FileNotFoundError, match="git clone"):
        mm.import_upstream(tmp_path / "nowhere")


# ---- statistics ---------------------------------------------------------------

def test_holm_correction():
    adjusted = study.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    # sorted: a 0.01*3=0.03, c 0.03*2=0.06, b max(0.06, 0.04*1)=0.06
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_cluster_bootstrap_interval_contains_the_median():
    rng = np.random.default_rng(1)
    values = rng.normal(5, 1, size=60)
    low, high = study.cluster_bootstrap(values, clusters=np.repeat(np.arange(30), 2))
    assert low < np.median(values) < high


def fake_rows(seed=0):
    """Synthetic measurement table: 4 groups, the 'D' group has a lower density on purpose."""
    rng = np.random.default_rng(seed)
    rows = []
    for dataset, n in (("fives_train", 30), ("fives_test", 10)):
        for g in study.GROUPS:
            for i in range(n):
                image_id = f"{dataset}/{i}_{g}"
                base = {m: rng.normal(1.0, 0.05) for m in mm.ALL_MEASURES}
                base["vessel_density"] = rng.normal(0.05 if g == "D" else 0.08, 0.005)
                for source in (("expert", "cnn") if dataset == "fives_test" else ("expert",)):
                    for method in mm.METHODS:
                        rows.append({"key": f"{image_id}|{source}", "dataset": dataset, "image_id": image_id, "source": source,
                                     "group": g, "child": "", "method": method, "seconds": 1.0,
                                     **{m: v * rng.normal(1.0, 0.01) for m, v in base.items()}})
    return rows


def test_group_table_finds_the_planted_difference():
    table = study.group_table(fake_rows(), "thinning")
    assert table["vessel_density"]["kruskal_p"] < 0.001
    assert table["vessel_density"]["vs_normal_p_holm"]["D"] < 0.001
    assert table["vessel_density"]["vs_normal_rank_biserial"]["D"] < -0.9  # D is lower than N


def test_classifier_beats_chance_only_when_there_is_signal():
    result = study.classifier_study(fake_rows(), "thinning", n_permutations=50)
    assert result["test_expert_masks"]["auc_D_vs_N"] > 0.95  # density separates D from N
    assert result["shuffled_label_control"]["mean_balanced_accuracy"] < 0.45


def test_agreement_study_on_identical_masks_is_zero():
    rows = [r for r in fake_rows() if r["dataset"] == "fives_test"]
    for r in rows:  # make the CNN measurements identical to the expert ones
        if r["source"] == "cnn":
            twin = next(x for x in rows if x["image_id"] == r["image_id"] and x["source"] == "expert" and x["method"] == r["method"])
            r.update({m: twin[m] for m in mm.ALL_MEASURES})
    out = study.agreement_study(rows, "fives_test", ["expert"], clusters_key=None)
    assert out["pairs"]["cnn_vs_expert"]["thinning"]["total_length"]["median"] == 0


# ---- app helpers --------------------------------------------------------------

def test_percentile_of():
    assert mm.percentile_of(5, [1, 2, 3, 4]) == 100
    assert mm.percentile_of(0, [1, 2, 3, 4]) == 0
    assert mm.percentile_of(2, [1, 2, 3, 4]) == pytest.approx(37.5)  # one below, one tie counted half
    assert np.isnan(mm.percentile_of(float("nan"), [1, 2]))


@needs_upstream
def test_measure_medskel_matches_measure_and_draws_on_the_vessel():
    mask = tube(length=300, radius=6)
    row, skeleton = mm.measure_medskel(mask)
    ref = mm.measure(mask, "medskel")
    for key in mm.ALL_MEASURES:
        assert row[key] == pytest.approx(ref[key], nan_ok=True)
    photo = np.full(mask.shape + (3,), 200, np.uint8)
    drawn = mm.draw_centre_lines(photo, skeleton, color=(255, 0, 0))
    red = (drawn[..., 0] == 255) & (drawn[..., 1] == 0)
    rows, cols = np.where(red)
    assert red.any() and abs(rows.mean() - mask.shape[0] / 2) < 2  # the line runs along the middle of the tube
    assert cols.max() - cols.min() > 250


def test_report_card_table_and_csv():
    from medskel_ai import report_card as rc

    values = {m: [float(i) for i in range(1, 101)] for m in mm.ALL_MEASURES}
    reference = {"groups": {g: {"name": g, "n": 100, "values": values} for g in "NADG"}}
    measures = {m: 3.0 for m in mm.ALL_MEASURES}
    rows = rc.table_rows(measures, reference)
    assert rows[0]["compare"].startswith("lower than 98% of healthy eyes")  # 2.5th percentile
    assert "outside the usual healthy range" in rows[0]["compare"]
    text = rc.markdown_table(measures, reference, comparable=True)
    assert "not a diagnosis" in text and "Tortuosity" not in text
    assert "No comparison" in rc.markdown_table(measures, reference, comparable=False)
    assert rc.csv_text(measures, reference).splitlines()[0].startswith("measure,value")


def test_reference_file_matches_the_study_results():
    """The shipped reference must be the CNN + MedSkel test-set numbers (n = 50 normal eyes)."""
    from medskel_ai import report_card as rc

    ref = rc.load_reference(study.REFERENCE)
    if ref is None:
        pytest.skip("results/morphometry/reference_cnn_medskel.json not present")
    assert ref["groups"]["N"]["n"] == 50
    assert np.median(ref["groups"]["N"]["values"]["vessel_density"]) == pytest.approx(0.1226, abs=5e-4)
