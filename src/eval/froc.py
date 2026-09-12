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


@dataclass
class Detection:
    slide_id: str
    score: float
    hit: bool
    lesion_key: str | None = None


def match(lesions, gt_polys, slide_id: str, mpp: float, offset_xy=(0.0, 0.0)):
    """Match predicted lesions to ground-truth polygons by centroid containment.

    Returns (detections, n_gt_lesions).  Each GT lesion can be hit at most
    once; extra predictions on the same lesion are neither hits nor false
    positives, matching the CAMELYON16 scoring convention.
    """
    from shapely.geometry import Point
    dets: list[Detection] = []
    claimed: set[int] = set()
    ox, oy = offset_xy
    for les in sorted(lesions, key=lambda l: -l.score_max):
        cy, cx = les.centroid_yx
        pt = Point(cx * mpp / 1.0 + ox, cy * mpp / 1.0 + oy)
        hit_idx = next((i for i, g in enumerate(gt_polys) if g.contains(pt)), None)
        if hit_idx is None:
            dets.append(Detection(slide_id, les.score_max, False))
        elif hit_idx not in claimed:
            claimed.add(hit_idx)
            dets.append(Detection(slide_id, les.score_max, True, f"{slide_id}:{hit_idx}"))
    return dets, len(gt_polys)


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
