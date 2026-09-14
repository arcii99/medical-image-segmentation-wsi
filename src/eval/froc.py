"""FROC analysis, CAMELYON16 protocol (ADR-013, primary metric).

Pixel Dice is dominated by large lesions and says little about whether a
micrometastasis was found.  FROC asks the clinically relevant question: at an
acceptable false-positive rate per slide, what fraction of lesions is
detected?
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OPERATING_POINTS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

# CAMELYON16 excludes isolated tumor cells from FROC, by MAJOR AXIS not area.
#
# Clinically (AJCC), a deposit under 0.2 mm in largest dimension is "isolated
# tumor cells" -- pN0(i+) -- and does not stage the node as positive. The
# official CAMELYON16 evaluation therefore drops lesions whose major axis is
# below 275 um from the ground truth entirely:
#
#   * they are NOT in the denominator, so missing one is not a miss;
#   * a prediction landing on one is NOT a false positive either.
#
# The second half matters as much as the first. Treating ITCs as ordinary
# negatives would punish a model for flagging real (if unstageable) tumor
# cells, which is exactly backwards.
ITC_MAX_AXIS_UM = 275.0
MACRO_MIN_AXIS_UM = 2000.0
MICRO_MIN_AXIS_UM = 200.0


def major_axis_um(poly, mpp: float) -> float:
    """Largest dimension of a polygon, in microns.

    Uses the minimum rotated bounding rectangle, which is the standard
    approximation and is orientation-independent -- unlike the axis-aligned
    bounding box, which inflates the size of a diagonal lesion.
    """
    try:
        rect = poly.minimum_rotated_rectangle
        xs, ys = rect.exterior.coords.xy
        edges = [((xs[i + 1] - xs[i]) ** 2 + (ys[i + 1] - ys[i]) ** 2) ** 0.5
                 for i in range(len(xs) - 1)]
        return max(edges) * mpp
    except Exception:  # noqa: BLE001  degenerate geometry
        minx, miny, maxx, maxy = poly.bounds
        return max(maxx - minx, maxy - miny) * mpp


def classify(poly, mpp: float) -> str:
    """'macro' (>2 mm) | 'micro' (0.2-2 mm) | 'itc' (<0.2 mm)."""
    ax = major_axis_um(poly, mpp)
    if ax >= MACRO_MIN_AXIS_UM:
        return "macro"
    return "micro" if ax >= MICRO_MIN_AXIS_UM else "itc"


def split_evaluable(polys, mpp: float, itc_max_axis_um: float = ITC_MAX_AXIS_UM):
    """Partition ground-truth polygons into (evaluable, itc_excluded)."""
    evaluable, itc = [], []
    for g in polys:
        (itc if major_axis_um(g, mpp) < itc_max_axis_um else evaluable).append(g)
    return evaluable, itc


@dataclass
class Detection:
    slide_id: str
    score: float
    hit: bool
    lesion_key: str | None = None


def match(lesions, gt_polys, slide_id: str, mpp: float, offset_xy=(0.0, 0.0),
          itc_max_axis_um: float = ITC_MAX_AXIS_UM):
    """Match predicted lesions to ground truth, CAMELYON16 convention.

    Parameters
    ----------
    lesions : predicted connected components (from infer.postproc.apply)
    gt_polys : ground-truth polygons in the HEATMAP coordinate frame
    mpp : microns per pixel of that frame

    Returns
    -------
    (detections, n_evaluable_gt, n_itc)

    Scoring rules, all three of which matter:

    * A ground-truth lesion counts at most once. Extra predictions on an
      already-claimed lesion are dropped -- neither hit nor false positive.
    * Lesions with major axis below ``itc_max_axis_um`` are isolated tumor
      cells. They are excluded from the denominator, and a prediction landing
      on one is dropped rather than charged as a false positive.
    * Everything else unmatched is a false positive.
    """
    from shapely.geometry import Point

    evaluable, itc = split_evaluable(gt_polys, mpp, itc_max_axis_um)
    dets: list[Detection] = []
    claimed: set[int] = set()
    ox, oy = offset_xy

    for les in sorted(lesions, key=lambda l: -l.score_max):
        cy, cx = les.centroid_yx
        pt = Point(cx + ox, cy + oy)          # heatmap pixel frame
        hit = next((i for i, g in enumerate(evaluable) if g.contains(pt)), None)
        if hit is not None:
            if hit not in claimed:
                claimed.add(hit)
                dets.append(Detection(slide_id, les.score_max, True,
                                      f"{slide_id}:{hit}"))
            continue                           # duplicate hit: ignored
        if any(g.contains(pt) for g in itc):
            continue                           # landed on an ITC: not an FP
        dets.append(Detection(slide_id, les.score_max, False))

    return dets, len(evaluable), len(itc)


def curve(detections, n_gt: int, n_slides: int):
    """Sensitivity as a function of average false positives per slide."""
    if n_gt == 0 or n_slides == 0:
        return np.array([]), np.array([])
    d = sorted(detections, key=lambda x: -x.score)
    tp = fp = 0
    sens, fpps = [], []
    for det in d:
        tp += det.hit
        fp += not det.hit
        sens.append(tp / n_gt)
        fpps.append(fp / n_slides)
    return np.asarray(fpps), np.asarray(sens)


def sensitivity_at(fpps: np.ndarray, sens: np.ndarray,
                   points=OPERATING_POINTS) -> dict[str, float]:
    out = {}
    for p in points:
        ok = fpps <= p
        out[str(p)] = float(sens[ok].max()) if ok.any() else 0.0
    return out


def average_sensitivity(at: dict[str, float]) -> float:
    return float(np.mean(list(at.values()))) if at else 0.0


def bootstrap_ci(values, n: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Slide-level bootstrap. With ~50 test slides, 3-point gaps are noise."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [rng.choice(v, v.size, replace=True).mean() for _ in range(n)]
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))
