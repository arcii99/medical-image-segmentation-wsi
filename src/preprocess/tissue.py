"""Tissue localisation (ADR-003).

Runs on a ~8 um/px thumbnail and returns a boolean tissue mask.  Roughly
60-85% of a WSI is blank glass; removing it cuts the patch count ~4x and stops
the loss being dominated by empty space.

Three guards make this robust rather than merely plausible:

* **Union of two Otsu branches** -- HSV saturation (glass is achromatic) and
  inverted grayscale (tissue is dark).  Each covers the other's failure mode.
* **Bimodality guard** -- Otsu on a unimodal histogram returns an arbitrary
  split, so a near-empty slide gets half its glass labelled tissue (BUG-004).
* **Pen / artifact rejection** -- marker ink is highly saturated and passes the
  saturation branch, then becomes confident false positives at inference
  (BUG-005).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi

from src.utils.geometry import mm2_to_px, um_to_px

log = logging.getLogger(__name__)

__all__ = ["TissueConfig", "TissueResult", "TissueDetectionError", "tissue_mask"]


class TissueDetectionError(RuntimeError):
    """Raised when the tissue mask violates its post-condition (contract A2)."""


@dataclass(frozen=True)
class TissueConfig:
    median_blur: int = 7               # px, on the thumbnail
    close_um: float = 40.0             # morphological closing radius
    open_um: float = 24.0              # morphological opening radius
    min_object_mm2: float = 0.02       # drop specks smaller than this
    bimodality_min: float = 0.08       # inter-class / total variance
    floor_sat: int = 20                # physical floor on the Otsu threshold
    floor_gray: int = 40
    pen_sat_min: float = 0.35          # normalised saturation
    pen_value_min: float = 0.20        # reject near-black (edges, shadow)
    he_hue_bands: tuple[tuple[float, float], ...] = ((0.70, 0.95), (0.00, 0.10))
    frac_min: float = 0.01             # contract A2 lower bound
    frac_max: float = 0.90             # contract A2 upper bound


@dataclass
class TissueResult:
    mask: np.ndarray                   # bool, thumbnail resolution
    tissue_frac: float
    pen_frac: float
    qc_flag: str                       # "ok" | "unimodal_fallback" | "high_pen"
    sat_threshold: int
    gray_threshold: int


# --------------------------------------------------------------------------
def tissue_mask(
    thumb_rgb: np.ndarray,
    mpp: float,
    cfg: TissueConfig | None = None,
    strict: bool = True,
) -> TissueResult:
    """Compute a tissue mask from a low-resolution RGB thumbnail.

    Parameters
    ----------
    thumb_rgb : (H, W, 3) uint8
        Thumbnail, alpha already composited onto white by ``SlideReader``.
    mpp : float
        Microns per pixel **of the thumbnail**.  All morphology radii are
        specified in microns and converted here, so the algorithm behaves
        identically at any thumbnail resolution (gate V0.5).
    strict : bool
        Enforce the contract-A2 bounds.  Disable only for QC tooling.
    """
    cfg = cfg or TissueConfig()
    if thumb_rgb.ndim != 3 or thumb_rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {thumb_rgb.shape}")

    k = cfg.median_blur | 1  # must be odd
    img = cv2.medianBlur(thumb_rgb, k)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]
    gray_inv = 255 - cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Artifacts are excluded BEFORE thresholding, not after.
    #
    # Order matters: a black border (unscanned area, coverslip edge) sits at
    # saturation 0, which manufactures a spurious bimodality in the saturation
    # histogram.  Otsu then splits black-vs-glass instead of glass-vs-tissue,
    # returns a threshold of ~0, and every glass pixel with a hint of colour
    # noise is promoted to tissue.  Subtracting the artifact afterwards
    # removes the black region but leaves the poisoned threshold in place, and
    # the slide indexes at ~0.67 tissue fraction instead of ~0.00.
    pen = _pen_and_artifact_mask(hsv, cfg)
    valid = ~pen
    if valid.sum() < 0.01 * valid.size:
        raise TissueDetectionError(
            f"only {valid.mean():.2%} of the thumbnail is non-artifact; "
            "slide is probably blank or corrupt")

    t_sat, ok_sat = _otsu_guarded(sat[valid], cfg.bimodality_min, cfg.floor_sat)
    t_gray, ok_gray = _otsu_guarded(gray_inv[valid], cfg.bimodality_min,
                                    cfg.floor_gray)
    qc = "ok" if (ok_sat or ok_gray) else "unimodal_fallback"

    mask = ((sat > t_sat) | (gray_inv > t_gray)) & valid
    pen_frac = float(pen.mean())

    r_close = um_to_px(cfg.close_um, mpp)
    r_open = um_to_px(cfg.open_um, mpp)
    mask = ndi.binary_closing(mask, _disk(r_close))
    mask = ndi.binary_opening(mask, _disk(r_open))
    mask = _remove_small(mask, mm2_to_px(cfg.min_object_mm2, mpp))
    mask = ndi.binary_fill_holes(mask)

    frac = float(mask.mean())
    if pen_frac > 0.05:
        qc = "high_pen"

    result = TissueResult(
        mask=mask.astype(bool), tissue_frac=frac, pen_frac=pen_frac,
        qc_flag=qc, sat_threshold=int(t_sat), gray_threshold=int(t_gray),
    )

    if strict:
        _assert_contract_a2(result, cfg)
    return result


# --------------------------------------------------------------------------
def _assert_contract_a2(r: TissueResult, cfg: TissueConfig) -> None:
    if r.tissue_frac > cfg.frac_max:
        raise TissueDetectionError(
            f"tissue fraction {r.tissue_frac:.4f} exceeds upper bound "
            f"{cfg.frac_max}; likely black-border (BUG-001) or ink (BUG-005)"
        )
    if r.tissue_frac < cfg.frac_min:
        raise TissueDetectionError(
            f"tissue fraction {r.tissue_frac:.4f} below lower bound "
            f"{cfg.frac_min}; slide is empty or thresholding is too aggressive"
        )


def _otsu_guarded(chan: np.ndarray, min_ratio: float, floor: int
                  ) -> tuple[int, bool]:
    """Otsu threshold, clamped from below by a physical floor.

    Two failure modes, one guard each:

    * **Unimodal histogram** (BUG-004).  On a near-empty slide Otsu still
      returns a threshold -- an arbitrary one splitting the glass
      distribution -- so half the glass is indexed as tissue and nothing
      raises.  The bimodality ratio detects this.

    * **Degenerate split on noise.**  The ratio alone is not enough: Otsu can
      split a pure-noise channel at 0, putting 5% of pixels below and 95%
      above, which scores as strongly "bimodal" while meaning nothing.  The
      floor handles this.  We know a priori that H&E tissue has saturation
      above ~20 and inverted-grey above ~40; a threshold below that is
      physically implausible no matter what the histogram says.
    """
    t, _ = cv2.threshold(chan.reshape(-1, 1), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = int(t)
    bimodal = _bimodality(chan, t) >= min_ratio
    if not bimodal:
        log.debug("unimodal histogram (ratio < %.3f); falling back to floor %d",
                  min_ratio, floor)
    return max(t, floor), bimodal


def _bimodality(chan: np.ndarray, t: int) -> float:
    """Inter-class variance divided by total variance at threshold ``t``."""
    x = chan.reshape(-1).astype(np.float64)
    total = x.var()
    if total <= 1e-9:
        return 0.0
    lo, hi = x[x <= t], x[x > t]
    if lo.size == 0 or hi.size == 0:
        return 0.0
    w0, w1 = lo.size / x.size, hi.size / x.size
    return float(w0 * w1 * (lo.mean() - hi.mean()) ** 2 / total)


def _pen_and_artifact_mask(hsv: np.ndarray, cfg: TissueConfig) -> np.ndarray:
    """Reject marker ink and near-black artifacts (BUG-005).

    Ink is highly saturated with a hue far from the H&E band; coverslip edges
    and shadows are near-black.
    """
    h = hsv[..., 0].astype(np.float32) / 180.0   # OpenCV hue is 0..179
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0

    in_he = np.zeros_like(h, dtype=bool)
    for lo, hi in cfg.he_hue_bands:
        in_he |= (h >= lo) & (h <= hi)

    ink = (s > cfg.pen_sat_min) & ~in_he
    dark = v < cfg.pen_value_min
    return ink | dark


def _disk(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def _remove_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.reshape(-1))
    sizes[0] = 0
    keep = sizes >= max(1, int(min_px))
    return keep[lab]


# --------------------------------------------------------------------------
# CLI (gate V2.3)
# --------------------------------------------------------------------------
def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(prog="src.preprocess.tissue")
    p.add_argument("--selftest-unimodal", action="store_true")
    args = p.parse_args()

    if not args.selftest_unimodal:
        p.print_help()
        return

    rng = np.random.default_rng(0)
    mpp = 8.0

    glass = np.clip(rng.normal(243, 3, (400, 400, 3)), 0, 255).astype(np.uint8)
    try:
        tissue_mask(glass, mpp)
        print("all-glass thumbnail   -> NO ERROR RAISED    FAIL")
        raise SystemExit(1)
    except TissueDetectionError as e:
        print(f"all-glass thumbnail   -> TissueDetectionError raised   OK\n"
              f"    ({e})")

    sparse = glass.copy()
    sparse[190:210, 190:210] = [140, 90, 160]          # ~0.25% tissue
    res = tissue_mask(sparse, mpp, strict=False)
    print(f"near-unimodal (0.25% tissue) -> qc_flag={res.qc_flag} "
          f"tissue_frac={res.tissue_frac:.4f}   OK")


if __name__ == "__main__":
    _cli()
