"""Vessel measurements ("morphometry") from a 0/1 vessel mask.

This is the second half of MedSkel AI. The U-Net draws the vessels. This file
turns a drawing into numbers.

Two ways to find the centre line ("skeleton") of the vessels
------------------------------------------------------------
1. thinning  : the standard method. Peel pixels off the border of the mask,
               layer by layer, until a line 1 pixel wide is left
               (scikit-image `skeletonize`).
2. medskel   : the geometric method from the original MedSkel project
               (Saidou et al. 2024, bisectors). The mask outline becomes a
               polygon, and the centre line is the set of points that are at
               the same distance from two sides of the outline (Voronoi).
               Setting: epsilon = 4 px, prune = 1.0, the same setting that gave
               the 3.2% vs 9.3% inter-observer result in the original repo.

We do NOT copy the MedSkel code here. We import it read-only from a local clone
of github.com/MouhssineElBoumshouli/medskel (MIT licence), so both projects stay
separate. The measurement formulas come from the same file
(`medskel.metrics.vessel_summary`), so the two methods are measured by exactly
the same code. A difference between them is then a difference of method, not
of measuring code.

The measurements (all in PIXELS; there is no validated pixel-to-millimetre scale)
-------------------------------------------------------------------------------
From the skeleton (one value per method):
    total_length      sum of the lengths of all vessel branches
    n_bifurcations    number of points where a vessel splits (degree >= 3)
    median_calibre    median vessel width (2 x distance from centre line to border)
    median_tortuosity median of (branch length / straight distance between its ends);
                      1.0 = perfectly straight
From the mask itself (the same for both methods):
    vessel_density    share of the retina (field of view) covered by vessels
    fractal_dimension box-counting dimension: how much the tree "fills" the
                      plane. A line has 1, a filled area has 2; retinal trees
                      are usually reported somewhere around 1.4 to 1.7.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

UPSTREAM_URL = "https://github.com/MouhssineElBoumshouli/medskel"
UPSTREAM_COMMIT = "1a838a8"  # the commit whose results CLAUDE.md quotes (3.2% vs 9.3%)
METHODS = ("thinning", "medskel")
MEDSKEL_EPSILON = 4.0
MEDSKEL_PRUNE = 1.0
SKELETON_MEASURES = ("total_length", "n_bifurcations", "median_calibre", "median_tortuosity")
MASK_MEASURES = ("vessel_density", "fractal_dimension")
ALL_MEASURES = MASK_MEASURES + SKELETON_MEASURES

_upstream = None


def default_upstream_path() -> Path:
    """MEDSKEL_REPO if set, otherwise a folder 'medskel' next to this repository."""
    env = os.environ.get("MEDSKEL_REPO")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "medskel"


def import_upstream(repo_path: str | Path | None = None):
    """Import (baseline, voronoi, metrics) from the original MedSkel repository, read-only."""
    global _upstream
    if _upstream is not None:
        return _upstream
    path = Path(repo_path) if repo_path else default_upstream_path()
    if not (path / "medskel" / "voronoi.py").exists():
        raise FileNotFoundError(
            f"The original MedSkel code was not found at {path}.\n"
            f"Clone it next to this repository:\n"
            f"    git clone {UPSTREAM_URL} {path}\n"
            f"    git -C {path} checkout {UPSTREAM_COMMIT}\n"
            f"or set MEDSKEL_REPO to its folder."
        )
    sys.path.insert(0, str(path))
    from medskel import baseline, metrics, voronoi  # noqa: E402  (imported from the clone)

    _upstream = (baseline, voronoi, metrics)
    return _upstream


def upstream_commit(repo_path: str | Path | None = None) -> str | None:
    """The git commit of the MedSkel clone, so results say exactly which code made them."""
    path = Path(repo_path) if repo_path else default_upstream_path()
    head = path / ".git" / "HEAD"
    try:
        ref = head.read_text().strip()
        if ref.startswith("ref:"):
            return (path / ".git" / ref.split()[1]).read_text().strip()[:7]
        return ref[:7]
    except OSError:
        return None


# ----------------------------------------------------------------------------
# Measurements that only need the mask
# ----------------------------------------------------------------------------

def vessel_density(mask: np.ndarray, fov: np.ndarray | None = None) -> float:
    """Share of the field of view that is vessel. Without a FOV: share of the whole image."""
    mask = np.asarray(mask, dtype=bool)
    if fov is None:
        return float(mask.mean())
    fov = np.asarray(fov, dtype=bool)
    area = fov.sum()
    return float((mask & fov).sum() / area) if area else float("nan")


def fractal_dimension(mask: np.ndarray, min_box: int = 2) -> float:
    """Box-counting dimension of a 0/1 mask.

    Cover the image with a grid of boxes of side s, and count N(s) = how many
    boxes contain at least one vessel pixel. Repeat for s = 2, 4, 8, ...
    For a shape of dimension D, N(s) grows like (1/s)^D. So D is the slope of
    log N(s) against log(1/s), found with a straight-line fit.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    # Crop to the vessels and pad to a power of two, so every box size divides the image.
    rows, cols = np.where(mask)
    mask = mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    side = 1 << int(np.ceil(np.log2(max(mask.shape))))
    padded = np.zeros((side, side), dtype=bool)
    padded[: mask.shape[0], : mask.shape[1]] = mask
    sizes, counts = [], []
    s = min_box
    while s <= side // 4:  # stop before boxes get so big that almost all are full
        boxes = padded.reshape(side // s, s, side // s, s).any(axis=(1, 3))
        sizes.append(s)
        counts.append(boxes.sum())
        s *= 2
    if len(sizes) < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(1.0 / np.array(sizes)), np.log(np.array(counts)), 1)
    return float(slope)


# ----------------------------------------------------------------------------
# Measurements that need a skeleton
# ----------------------------------------------------------------------------

def skeleton_of(mask: np.ndarray, method: str, repo_path=None):
    """Centre line of the mask with the chosen method (pixel array or MedSkel graph)."""
    baseline, voronoi, _ = import_upstream(repo_path)
    mask = np.asarray(mask, dtype=bool)
    if method == "thinning":
        return baseline.thinning(mask)
    if method == "medskel":
        return voronoi.skeletonize(mask, epsilon=MEDSKEL_EPSILON, prune=MEDSKEL_PRUNE)
    raise ValueError(f"unknown method {method!r}, use one of {METHODS}")


def _densify(poly: np.ndarray, step: float = 1.0) -> np.ndarray:
    """Points every `step` pixels along a polyline (x, y), including both ends."""
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 2:
        return poly
    out = [poly[0]]
    for a, b in zip(poly[:-1], poly[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        out.extend(a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:])
    return np.array(out)


def medskel_median_calibre(skeleton, mask: np.ndarray) -> float:
    """Median vessel width for a MedSkel skeleton, measured like the thinning baseline.

    Why not use MedSkel's own number: in the original code (voronoi.py,
    `_reduce_to_branches`) each branch keeps its radii only at the vertices that
    survive polyline simplification. On a straight vessel those are mostly the
    two TIPS, where the radius shrinks to almost zero, so the width comes out far
    too small (a 12-pixel-wide straight test vessel gives 1.3 px). We do not edit
    the original repository. Instead, for both methods, width = 2 x the distance
    from the centre line to the nearest background pixel, averaged along each
    branch (sampled every pixel), then the median over branches.
    """
    from scipy.ndimage import distance_transform_edt

    dt = distance_transform_edt(np.asarray(mask, dtype=bool))
    h, w = dt.shape
    widths = []
    for poly in skeleton.polylines():
        pts = _densify(poly)
        if len(pts) == 0:
            continue
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        widths.append(2 * float(np.mean(dt[yi, xi])))
    return float(np.median(widths)) if widths else float("nan")


def measure(mask: np.ndarray, method: str, fov: np.ndarray | None = None, repo_path=None) -> dict:
    """All measurements of one mask with one skeleton method. Also records the time it took."""
    _, _, metrics = import_upstream(repo_path)
    mask = np.asarray(mask, dtype=bool)
    row = {"method": method, "vessel_density": vessel_density(mask, fov), "fractal_dimension": fractal_dimension(mask)}
    start = time.perf_counter()
    if mask.any():
        skeleton = skeleton_of(mask, method, repo_path)
        summary = metrics.vessel_summary(skeleton, mask)
        if method == "medskel":  # replace the biased width (see medskel_median_calibre)
            summary["median_calibre"] = medskel_median_calibre(skeleton, mask)
    else:  # an empty mask has no vessels to measure
        summary = {"total_length": 0.0, "n_bifurcations": 0, "median_calibre": float("nan"), "median_tortuosity": float("nan")}
    row.update({k: float(summary[k]) for k in SKELETON_MEASURES})
    row["seconds"] = time.perf_counter() - start
    return row


def measure_medskel(mask: np.ndarray, fov: np.ndarray | None = None, repo_path=None):
    """For the app: MedSkel measurements AND the skeleton itself (to draw the centre lines).

    Same numbers as `measure(mask, "medskel", fov)`, computed once.
    """
    _, _, metrics = import_upstream(repo_path)
    mask = np.asarray(mask, dtype=bool)
    row = {"vessel_density": vessel_density(mask, fov), "fractal_dimension": fractal_dimension(mask)}
    if not mask.any():
        row.update({"total_length": 0.0, "n_bifurcations": 0.0, "median_calibre": float("nan"), "median_tortuosity": float("nan")})
        return row, None
    skeleton = skeleton_of(mask, "medskel", repo_path)
    summary = metrics.vessel_summary(skeleton, mask)
    summary["median_calibre"] = medskel_median_calibre(skeleton, mask)
    row.update({k: float(summary[k]) for k in SKELETON_MEASURES})
    return row, skeleton


def draw_centre_lines(image: np.ndarray, skeleton, color=(255, 60, 60), thickness: int = 2, dim: float = 0.55) -> np.ndarray:
    """Darken the photo a little and draw the MedSkel centre lines on it."""
    import cv2

    out = (np.asarray(image, dtype=np.float32) * dim).astype(np.uint8)
    out = np.ascontiguousarray(out)
    if skeleton is None:
        return out
    for poly in skeleton.polylines():
        pts = np.round(np.asarray(poly)).astype(np.int32).reshape(-1, 1, 2)  # (x, y) pixel coordinates
        if len(pts) >= 2:
            cv2.polylines(out, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return out


def percentile_of(value: float, reference: list[float]) -> float:
    """Share (0-100) of reference values that are below `value` (ties count half)."""
    ref = np.asarray([r for r in reference if np.isfinite(r)], dtype=float)
    if not np.isfinite(value) or len(ref) == 0:
        return float("nan")
    return float(100.0 * (np.sum(ref < value) + 0.5 * np.sum(ref == value)) / len(ref))


def relative_difference(a: float, b: float) -> float:
    """|a - b| / mean(a, b). Same definition as the original MedSkel inter-observer experiment.

    Relative, not absolute: a method that simply finds fewer branches would
    otherwise look "more stable" just because its numbers are smaller.
    """
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    mean = 0.5 * (abs(a) + abs(b))
    return abs(a - b) / mean if mean > 1e-12 else float("nan")
