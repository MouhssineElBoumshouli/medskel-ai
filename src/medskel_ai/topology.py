"""Two measurements that Dice does not give us: connectivity and thin vessels.

Dice counts pixels. A thick vessel has many pixels, a thin one has few. So a
model can lose most thin vessels, or cut a vessel in two, and Dice hardly moves.

1) clDice ("centerline Dice", Shit et al., CVPR 2021)
   A SKELETON is the 1-pixel-wide centre line of a shape. Every vessel, thick
   or thin, gives the same kind of line. So on skeletons a thin vessel counts
   as much per unit of length as a thick one.

       Tprec = share of the PREDICTED skeleton that lies inside the expert mask
               (low = we drew vessels that do not exist)
       Tsens = share of the EXPERT skeleton that lies inside the predicted mask
               (low = we missed vessels or broke them)
       clDice = 2 * Tprec * Tsens / (Tprec + Tsens)

   Link to the medskel project: medskel also turns a vessel mask into centre
   lines, with a geometric (polygon / Voronoi) method. Here we use the standard
   pixel thinning of scikit-image only as a measuring tool.

2) Thin-vessel recall
   "Thin" needs a definition. Ours, fixed before looking at results:
     a) take the skeleton of the EXPERT mask;
     b) at each skeleton pixel, the distance transform gives the distance d to
        the nearest background pixel. The local vessel width is about 2*d - 1
        (a 1-px line has d = 1, a 3-px line has d = 2, a 5-px line has d = 3);
     c) every vessel pixel gets the width of its nearest skeleton pixel;
     d) a vessel pixel is THIN if that width is <= 3 pixels (d <= 2).
        Honest detail: for even widths the formula is one pixel too small
        (a 4-px vessel also has d = 2). So "thin" really means 1 to 4 pixels wide.
   Thin-vessel recall = share of the thin expert pixels that the model found.
   The definition uses only the expert mask, never the prediction.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

THIN_MAX_WIDTH = 3  # pixels, at the resolution of the mask that is given


def skeleton(mask: np.ndarray) -> np.ndarray:
    """1-pixel-wide centre lines of a 0/1 mask (boolean array)."""
    return skeletonize(mask.astype(bool))


def thin_vessel_mask(truth: np.ndarray, truth_skeleton: np.ndarray, max_width: int = THIN_MAX_WIDTH) -> np.ndarray:
    """Boolean mask of the expert vessel pixels that belong to thin vessels (width <= max_width)."""
    truth = truth.astype(bool)
    if not truth_skeleton.any():
        return np.zeros_like(truth)
    distance = distance_transform_edt(truth)  # distance of each vessel pixel to the background
    # For every pixel: the position of the nearest skeleton pixel.
    _, (rows, cols) = distance_transform_edt(~truth_skeleton, return_indices=True)
    width = 2 * distance[rows, cols] - 1  # width of the vessel at that skeleton pixel
    return truth & (width <= max_width)


def _ratio(top: float, bottom: float, if_empty: float) -> float:
    return float(top) / float(bottom) if bottom > 0 else if_empty


def topology_metrics(pred: np.ndarray, truth: np.ndarray, truth_skeleton=None, thin=None) -> dict[str, float]:
    """clDice parts and thin / thick recall for one image.

    pred, truth: boolean arrays of the same shape.
    truth_skeleton, thin: optional precomputed results for the expert mask (they never change).
    """
    pred, truth = pred.astype(bool), truth.astype(bool)
    if truth_skeleton is None:
        truth_skeleton = skeleton(truth)
    if thin is None:
        thin = thin_vessel_mask(truth, truth_skeleton)
    pred_skeleton = skeleton(pred)

    both_empty = not truth.any() and not pred.any()
    tprec = _ratio((pred_skeleton & truth).sum(), pred_skeleton.sum(), 1.0 if both_empty else 0.0)
    tsens = _ratio((truth_skeleton & pred).sum(), truth_skeleton.sum(), 1.0 if both_empty else 0.0)
    thick = truth & ~thin
    return {
        "cldice": _ratio(2 * tprec * tsens, tprec + tsens, 0.0),
        "cl_precision": tprec,
        "cl_sensitivity": tsens,
        # NaN = this image has no thin (or no thick) expert pixels; such images are skipped in the mean.
        "thin_recall": _ratio((pred & thin).sum(), thin.sum(), float("nan")),
        "thick_recall": _ratio((pred & thick).sum(), thick.sum(), float("nan")),
        "thin_share_of_vessel_pixels": _ratio(thin.sum(), truth.sum(), float("nan")),
    }
