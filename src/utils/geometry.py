"""Coordinate and physical-unit conversions.

This module is the *only* place in the project allowed to convert between
pyramid levels or between physical units and pixels.

Two invariants are enforced here and relied upon everywhere else:

INV-1  Every persisted coordinate (patch index, annotations, heatmap attrs,
       logs) is in **level-0 pixels**.
INV-2  Downsample factors come from the slide file, never from ``2 ** level``.
       Scanners emit levels sized ``ceil(w / 2)``, so the effective factor is
       e.g. 1.9992175 and the error compounds to hundreds of pixels across a
       100k-pixel slide.  See docs/bug.md BUG-003.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "Window",
    "level_to_l0",
    "l0_to_level",
    "resolve_level_for_mpp",
    "um_to_px",
    "mm2_to_px",
    "px_to_um",
    "assert_level0_coords",
    "clamp_window",
    "block_downsample",
]


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Window:
    """A read window.

    ``x0``/``y0`` are **level-0** pixels (INV-1).
    ``w``/``h`` are pixels **at ``level``** -- this mirrors the
    ``read_region(location, level, size)`` contract of OpenSlide/tiffslide.
    """

    x0: int
    y0: int
    level: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError(f"non-positive window size: {self.w}x{self.h}")
        assert_level0_coords(self.x0, self.y0)

    @property
    def size(self) -> tuple[int, int]:
        return (self.w, self.h)

    def extent_l0(self, downsamples: Sequence[float]) -> tuple[float, float]:
        """Width/height of this window measured in level-0 pixels."""
        s = downsamples[self.level]
        return (self.w * s, self.h * s)

    def bbox_l0(self, downsamples: Sequence[float]) -> tuple[float, float, float, float]:
        dw, dh = self.extent_l0(downsamples)
        return (self.x0, self.y0, self.x0 + dw, self.y0 + dh)


# --------------------------------------------------------------------------
# Level conversions
# --------------------------------------------------------------------------
def level_to_l0(
    xy: tuple[float, float], downsamples: Sequence[float], level: int
) -> tuple[float, float]:
    """Convert a coordinate at ``level`` into level-0 space."""
    s = float(downsamples[level])
    return (xy[0] * s, xy[1] * s)


def l0_to_level(
    xy: tuple[float, float], downsamples: Sequence[float], level: int
) -> tuple[float, float]:
    """Convert a level-0 coordinate into ``level`` space.

    Uses the slide's reported factor, never ``2 ** level`` (INV-2).
    """
    s = float(downsamples[level])
    return (xy[0] / s, xy[1] / s)


def resolve_level_for_mpp(
    downsamples: Sequence[float],
    mpp_level0: float,
    target_mpp: float,
    tolerance: float = 0.05,
) -> tuple[int, float]:
    """Pick the pyramid level closest to ``target_mpp``.

    Implements ADR-002: magnification is selected by microns-per-pixel, not by
    level index, because "20x" means different level indices on different
    scanners.

    Returns
    -------
    (level, residual_scale)
        ``residual_scale = actual_mpp / target_mpp``.  A caller must resample
        the patch by this factor when ``abs(residual - 1) > tolerance``; below
        the tolerance the scale error is inside the range covered by
        augmentation and is accepted as-is.

    Raises
    ------
    ValueError
        If ``mpp_level0`` is missing or non-positive.  Guessing physical scale
        is worse than dropping the slide (ADR-002 Consequences).
    """
    if mpp_level0 is None or not math.isfinite(mpp_level0) or mpp_level0 <= 0:
        raise ValueError(
            f"slide has no usable level-0 MPP (got {mpp_level0!r}); "
            "refusing to guess physical scale"
        )
    if target_mpp <= 0:
        raise ValueError(f"target_mpp must be positive, got {target_mpp}")

    mpps = [mpp_level0 * float(d) for d in downsamples]
    # Compare in log space so 2x too coarse and 2x too fine are penalised equally.
    level = int(np.argmin([abs(math.log(m / target_mpp)) for m in mpps]))
    residual = mpps[level] / target_mpp
    return level, residual


def needs_resample(residual_scale: float, tolerance: float = 0.05) -> bool:
    return abs(residual_scale - 1.0) > tolerance


# --------------------------------------------------------------------------
# Physical units
# --------------------------------------------------------------------------
def um_to_px(microns: float, mpp: float) -> int:
    """Microns -> pixels at a given microns-per-pixel, rounded, minimum 1.

    Physical thresholds (morphology radii, minimum lesion size) must be written
    in physical units and converted here.  Hard-coded pixel constants silently
    change meaning whenever the working resolution changes; gate V0.5 greps
    for them.
    """
    if mpp <= 0:
        raise ValueError(f"mpp must be positive, got {mpp}")
    return max(1, int(round(microns / mpp)))


def px_to_um(pixels: float, mpp: float) -> float:
    return pixels * mpp


def mm2_to_px(area_mm2: float, mpp: float) -> int:
    """Square millimetres -> pixel count at a given microns-per-pixel."""
    if mpp <= 0:
        raise ValueError(f"mpp must be positive, got {mpp}")
    area_um2 = area_mm2 * 1_000_000.0
    return max(1, int(round(area_um2 / (mpp * mpp))))


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------
def assert_level0_coords(x0: float, y0: float) -> None:
    """Cheap guard on INV-1.

    Cannot prove a coordinate is level-0, but catches the common failure of
    passing a negative or non-finite origin.
    """
    if not (math.isfinite(x0) and math.isfinite(y0)):
        raise ValueError(f"non-finite origin ({x0}, {y0})")
    if x0 < 0 or y0 < 0:
        raise ValueError(
            f"negative origin ({x0}, {y0}); read_region takes level-0 "
            "coordinates and they are never negative"
        )


def clamp_window(
    x0: int,
    y0: int,
    level: int,
    size: int,
    level_dims: Sequence[tuple[int, int]],
    downsamples: Sequence[float],
) -> tuple[int, int]:
    """Shift a window left/up so it stays inside the slide.

    Clamping rather than skipping is deliberate: skipping margin positions
    leaves uncovered strips in the reconstructed heatmap, which read as
    confident negatives (docs/bug.md BUG-012).  The extra overlap that
    clamping introduces is absorbed by the stitcher's ``num/den``
    normalisation for free.
    """
    lw, lh = level_dims[level]
    s = float(downsamples[level])
    max_x0 = int(math.floor((lw - size) * s))
    max_y0 = int(math.floor((lh - size) * s))
    if max_x0 < 0 or max_y0 < 0:
        raise ValueError(
            f"window {size}px at level {level} exceeds level dims {lw}x{lh}"
        )
    return (min(max(int(x0), 0), max_x0), min(max(int(y0), 0), max_y0))


# --------------------------------------------------------------------------
# Array helpers
# --------------------------------------------------------------------------
def block_downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """Exact block-mean downsample of a 2-D float array by an integer factor.

    Used by the stitcher to bring patch-resolution probabilities down to
    heatmap resolution.  Block mean (not subsampling) so no probability mass
    is thrown away.
    """
    if factor == 1:
        return arr
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    h, w = arr.shape[:2]
    if h % factor or w % factor:
        raise ValueError(
            f"shape {arr.shape} is not divisible by factor {factor}; "
            "patch size and heatmap downsample must be commensurate"
        )
    return arr.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
