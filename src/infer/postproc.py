"""Heatmap post-processing (ADR-012).

Every threshold here is expressed in **physical units** and converted at
runtime from the heatmap's own mpp.  Pixel constants silently change meaning
whenever the working resolution or heatmap downsample changes; gate V0.5 greps
for them.

0.02 mm^2 sits well below the isolated-tumor-cell threshold (0.2 mm largest
dimension), so genuinely reportable findings survive while sub-cellular-
cluster speckle -- stain artifacts, macrophages, folds -- is removed.  Each
speckle is a false positive under FROC, so this is worth ~0.13 average
sensitivity at no model cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from src.utils.geometry import mm2_to_px, um_to_px

__all__ = ["PostProcConfig", "Lesion", "apply", "threshold_from_ckpt"]


@dataclass(frozen=True)
class PostProcConfig:
    min_lesion_mm2: float = 0.02
    opening_um: float = 10.0
    fill_holes: bool = True


@dataclass(frozen=True)
class Lesion:
    label: int
    area_px: int
    area_mm2: float
    centroid_yx: tuple[float, float]
    score_mean: float
    score_max: float


def apply(heat: np.ndarray, threshold: float, mpp: float,
          cfg: PostProcConfig | None = None) -> tuple[np.ndarray, list[Lesion]]:
    """Threshold, clean, and label. Returns (labels, lesions)."""
    cfg = cfg or PostProcConfig()
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0,1), got {threshold}")

    binary = np.asarray(heat, dtype=np.float32) > float(threshold)
    if cfg.fill_holes:
        binary = ndi.binary_fill_holes(binary)

    r = um_to_px(cfg.opening_um, mpp)
    if r >= 1:
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        binary = ndi.binary_opening(binary, (xx * xx + yy * yy) <= r * r)

    labels, n = ndi.label(binary)
    if n == 0:
        return labels.astype(np.int32), []

    min_px = mm2_to_px(cfg.min_lesion_mm2, mpp)
    sizes = np.bincount(labels.reshape(-1)); sizes[0] = 0
    keep = np.flatnonzero(sizes >= min_px)
    remap = np.zeros(sizes.size, dtype=np.int32)
    remap[keep] = np.arange(1, keep.size + 1, dtype=np.int32)
    labels = remap[labels]

    lesions: list[Lesion] = []
    if keep.size:
        idx = np.arange(1, keep.size + 1)
        cents = ndi.center_of_mass(labels > 0, labels, idx)
        means = ndi.mean(heat, labels, idx)
        maxs = ndi.maximum(heat, labels, idx)
        areas = np.bincount(labels.reshape(-1), minlength=keep.size + 1)[1:]
        for k in range(keep.size):
            lesions.append(Lesion(
                label=k + 1, area_px=int(areas[k]),
                area_mm2=float(areas[k] * mpp * mpp / 1e6),
                centroid_yx=(float(cents[k][0]), float(cents[k][1])),
                score_mean=float(means[k]), score_max=float(maxs[k])))
    return labels.astype(np.int32), lesions


def threshold_from_ckpt(ckpt: dict) -> float:
    """Tau is checkpoint-coupled. Defaulting to 0.5 here is a silent error."""
    tau = ckpt.get("threshold")
    if tau is None:
        raise KeyError(
            "checkpoint has no fitted threshold; evaluating with a default 0.5 "
            "is a 5-10 point F1 error when the sampler distorts the prior (ADR-006)")
    return float(tau)


def _cli() -> None:
    import argparse
    p = argparse.ArgumentParser(prog="src.infer.postproc")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if not a.selftest:
        p.print_help(); return
    for mpp in (2.0, 0.5):
        print(f"mpp {mpp} um/px -> min_area 0.02 mm^2 = {mm2_to_px(0.02, mpp):,} px      OK")
    print(f"opening radius 10 um -> {um_to_px(10, 2.0)} px @2.0 / "
          f"{um_to_px(10, 0.5)} px @0.5     OK")


if __name__ == "__main__":
    _cli()
