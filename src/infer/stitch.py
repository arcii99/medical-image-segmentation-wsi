"""Gigapixel heatmap reconstruction (ADR-009, Architecture section 6).

Overlapping patch predictions are accumulated into two on-disk arrays and
normalised at the end::

    w(u, v) = exp(-((u - c)^2 + (v - c)^2) / (2 sigma^2))
    num += p * w
    den += w
    heat = num / (den + eps)

Design points that are load-bearing:

* **Accumulators are float32.** In float16, ``4.0 + 0.0003 == 4.0``, so border
  contributions vanish, ``den`` under-counts and the quotient inflates toward
  1.0 (BUG-009).  Only the final quotient is cast to float16.
* **Partition of unity is not required.**  Dividing by ``den`` normalises any
  window, so stride and sigma are independent knobs.
* **Coverage is asserted before writing.**  A dropped batch or a skipped
  margin position leaves ``den == 0``, the quotient reads 0.0, and a hole in a
  heatmap is indistinguishable from a confident negative under FROC
  (BUG-012).

The store is duck-typed: anything supporting ``arr[slice, slice]`` get/set
works, so this runs against zarr on disk and plain numpy in tests.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from src.utils.geometry import block_downsample

log = logging.getLogger(__name__)

__all__ = ["GaussianStitcher", "CoverageError", "StitchConfig", "gaussian_window"]


class CoverageError(RuntimeError):
    """Raised when tissue was left uncovered by the sliding window."""


class ArrayStore(Protocol):
    shape: tuple[int, ...]
    dtype: Any

    def __getitem__(self, key: Any) -> np.ndarray: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...


@dataclass(frozen=True)
class StitchConfig:
    patch_size: int = 512              # px at the working level
    heat_downsample: int = 8           # heatmap resolution, relative to level 0
    sigma_divisor: float = 8.0         # sigma = patch_size / divisor
    chunk: int = 1024
    eps: float = 1e-8
    window_floor: float = 1e-3         # see gaussian_window()
    hole_den: float = 1e-6             # below this, the pixel was never written
    low_weight_den: float = 0.05       # reported, not fatal
    coverage_max_uncovered: float = 1e-4


# --------------------------------------------------------------------------
def gaussian_window(size: int, sigma: float, floor: float = 1e-3) -> np.ndarray:
    """Separable isotropic Gaussian centred on the patch, peak ~1.0, plus a floor.

    The floor exists for numerical conditioning at the slide border.  With
    ``sigma = size / 8`` the raw Gaussian is ~1.4e-7 at the patch corner, and
    the outermost pixels of a slide are covered by exactly one patch (nothing
    can extend past the edge).  ``num/den`` is still algebraically correct
    there, but a 7e6:1 dynamic range in a float32 accumulator is a poor thing
    to rely on.

    A floor of 1e-3 bounds the range at 1000:1, leaves the interior minimum
    essentially unchanged (measured 0.0735 -> 0.0775 at stride = size/2), and
    still vetoes border predictions 1000-fold wherever a neighbouring patch
    centre covers the same pixel -- which is the property the window exists
    for (BUG-010).
    """
    c = (size - 1) / 2.0
    ax = np.arange(size, dtype=np.float32) - c
    g = np.exp(-(ax ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    return (np.outer(g, g) + np.float32(floor)).astype(np.float32)


# --------------------------------------------------------------------------
class GaussianStitcher:
    """Accumulate patch probabilities into a slide-level heatmap."""

    def __init__(
        self,
        level0_shape: tuple[int, int],          # (H0, W0) in level-0 px
        level_downsample: float,                # downsample of the working level
        cfg: StitchConfig | None = None,
        store_factory: Any | None = None,
    ):
        self.cfg = cfg or StitchConfig()
        self.level0_shape = level0_shape
        self.level_downsample = float(level_downsample)

        # Patch pixels -> heatmap pixels.
        #
        # The block-downsample factor must be an integer, but the true ratio
        # never is: level downsamples are ~1.9996, not 2.0 (INV-2), so a
        # nominal 4x is really 4.0008x.  Rounding is safe because each patch's
        # destination is computed independently from its own level-0 origin
        # (``hx = x0 / heat_downsample``) rather than by accumulating tile
        # offsets.  The residual is therefore bounded by the within-patch
        # error -- here 512 * 0.0008 / 8 = 0.05 heatmap px -- and does not
        # compound across the slide the way BUG-003 did.
        ratio = self.cfg.heat_downsample / self.level_downsample
        self.patch_to_heat = int(round(ratio))
        self.ratio_residual = abs(ratio - self.patch_to_heat) / self.patch_to_heat
        if self.patch_to_heat < 1 or self.ratio_residual > 0.01:
            raise ValueError(
                f"heat_downsample {self.cfg.heat_downsample} is not within 1% "
                f"of an integer multiple of the working-level downsample "
                f"{self.level_downsample} (ratio {ratio:.4f}, "
                f"residual {self.ratio_residual:.2%})"
            )
        if self.ratio_residual > 1e-6:
            log.debug("patch->heat ratio %.5f rounded to %d (residual %.3f%%, "
                      "max within-patch drift %.3f heat px)", ratio,
                      self.patch_to_heat, self.ratio_residual * 100,
                      self.cfg.patch_size * self.ratio_residual / self.patch_to_heat)
        if self.cfg.patch_size % self.patch_to_heat:
            raise ValueError(
                f"patch_size {self.cfg.patch_size} not divisible by "
                f"{self.patch_to_heat}"
            )

        self.tile = self.cfg.patch_size // self.patch_to_heat   # heat px per patch
        self.extent_l0 = self.cfg.patch_size * self.level_downsample
        h0, w0 = level0_shape
        d = self.cfg.heat_downsample
        self.shape = (int(math.ceil(h0 / d)), int(math.ceil(w0 / d)))

        factory = store_factory or _numpy_store
        self.num: ArrayStore = factory(self.shape, np.float32, self.cfg.chunk, "num")
        self.den: ArrayStore = factory(self.shape, np.float32, self.cfg.chunk, "den")

        # BUG-009: this is not a stylistic assertion.
        if self.num.dtype != np.float32 or self.den.dtype != np.float32:
            raise TypeError(
                f"accumulators must be float32, got num={self.num.dtype} "
                f"den={self.den.dtype}; float16 silently drops border weights"
            )

        sigma_patch = self.cfg.patch_size / self.cfg.sigma_divisor
        self.window = gaussian_window(
            self.tile, sigma_patch / self.patch_to_heat, self.cfg.window_floor
        )
        self.n_added = 0

    # -- accumulation ---------------------------------------------------
    def add(self, probs: np.ndarray, coords: Sequence[tuple[int, int]]) -> None:
        """Add a batch of probability maps.

        Parameters
        ----------
        probs : (B, S, S) or (B, 1, S, S) float array in [0, 1]
            Post-sigmoid predictions at the **working level**.
        coords : sequence of (x0, y0)
            Level-0 top-left corner of each patch.
        """
        arr = np.asarray(probs, dtype=np.float32)
        if arr.ndim == 4:
            if arr.shape[1] != 1:
                raise ValueError(f"expected 1 channel, got {arr.shape[1]}")
            arr = arr[:, 0]
        if arr.ndim != 3:
            raise ValueError(f"expected (B,S,S) or (B,1,S,S), got {arr.shape}")
        if len(coords) != arr.shape[0]:
            raise ValueError(f"{arr.shape[0]} maps but {len(coords)} coords")
        if arr.shape[1] != self.cfg.patch_size or arr.shape[2] != self.cfg.patch_size:
            raise ValueError(
                f"patch size mismatch: got {arr.shape[1:]}, "
                f"configured {self.cfg.patch_size}"
            )
        lo, hi = float(arr.min(initial=0.0)), float(arr.max(initial=0.0))
        if lo < -1e-3 or hi > 1.0 + 1e-3:
            raise ValueError(f"probabilities out of range [{lo:.4f}, {hi:.4f}]")

        d = self.cfg.heat_downsample
        for p, (x0, y0) in zip(arr, coords):
            small = block_downsample(p, self.patch_to_heat)
            self._accumulate(small, int(x0), int(y0))
            self.n_added += 1

    def _accumulate(self, small: np.ndarray, x0_l0: int, y0_l0: int) -> None:
        """Place one downsampled patch, given its **level-0** origin.

        Destination placement has to account for two separate roundings:

        * The heatmap is sized ``ceil(H0 / d)``, so its last row represents a
          partial level-0 row.
        * A patch's level-0 extent is ``patch_size * level_downsample``, which
          is ``127.97`` heatmap pixels, not ``128``.

        Rounding the origin down therefore leaves the final row and column
        unwritten -- ``den == 0``, which reads as a confident negative rather
        than as missing data (BUG-012).  A patch flush against the slide's far
        edge is anchored to the heatmap's far edge instead, shifting it by at
        most one heatmap pixel (``heat_downsample`` level-0 px, 2 um at the
        defaults) and only for patches already at the border.
        """
        H, W = self.shape
        H0, W0 = self.level0_shape
        t, d = self.tile, self.cfg.heat_downsample

        hx = int(round(x0_l0 / d))
        hy = int(round(y0_l0 / d))
        if x0_l0 + self.extent_l0 >= W0 - self.level_downsample:
            hx = W - t
        if y0_l0 + self.extent_l0 >= H0 - self.level_downsample:
            hy = H - t
        if H >= t:
            hy = min(max(hy, 0), H - t)
        if W >= t:
            hx = min(max(hx, 0), W - t)

        y1, x1 = min(hy + t, H), min(hx + t, W)
        y0, x0 = max(hy, 0), max(hx, 0)
        if y1 <= y0 or x1 <= x0:
            return
        sy0, sx0 = y0 - hy, x0 - hx
        sub = small[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
        win = self.window[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]

        self.num[y0:y1, x0:x1] = self.num[y0:y1, x0:x1] + sub * win
        self.den[y0:y1, x0:x1] = self.den[y0:y1, x0:x1] + win

    # -- finalisation ---------------------------------------------------
    def coverage_report(self, tissue: np.ndarray | None = None
                        ) -> tuple[float, np.ndarray]:
        """Fraction of tissue never written by any patch, and the mask of it.

        This detects **holes** -- dropped batches, skipped margin positions --
        not merely low-weight borders.  A pixel written once at weight 1e-3
        still yields the correct ``num/den`` quotient; a pixel written zero
        times yields 0.0 and looks like a confident negative (BUG-012).
        """
        den = np.asarray(self.den[:, :])
        weak = den < self.cfg.hole_den
        if tissue is None:
            return float(weak.mean()), weak
        t = _resize_bool(tissue, self.shape)
        uncovered = weak & t
        denom = max(int(t.sum()), 1)
        return float(uncovered.sum() / denom), uncovered

    def finalize(self, tissue: np.ndarray | None = None,
                 dtype: Any = np.float16) -> np.ndarray:
        """Normalise, assert coverage, and return the heatmap.

        Raises
        ------
        CoverageError
            If more than ``coverage_max_uncovered`` of tissue has ``den``
            below threshold.  Failing loudly here is the whole point: a hole
            in the heatmap reads as a confident negative downstream and no
            metric will reveal it (BUG-012).
        """
        if self.n_added == 0:
            raise CoverageError("finalize() called before any patches were added")

        frac, _ = self.coverage_report(tissue)
        if frac > self.cfg.coverage_max_uncovered:
            raise CoverageError(
                f"{frac:.2%} of tissue uncovered by the sliding window "
                f"(threshold {self.cfg.coverage_max_uncovered:.2%}); "
                "a heatmap hole is indistinguishable from a confident negative"
            )

        num = np.asarray(self.num[:, :], dtype=np.float32)
        den = np.asarray(self.den[:, :], dtype=np.float32)

        self.low_weight_frac = float((den < self.cfg.low_weight_den).mean())
        if self.low_weight_frac > 0.30:
            log.warning("%.1f%% of the heatmap has den < %.3f; check stride "
                        "vs sigma", self.low_weight_frac * 100,
                        self.cfg.low_weight_den)

        heat = num / (den + self.cfg.eps)
        heat[den < self.cfg.hole_den] = 0.0

        hi = float(heat.max(initial=0.0))
        if hi > 1.0 + 1e-3:
            raise ValueError(f"heatmap exceeds 1.0 (max {hi:.5f}); check BUG-009")
        return np.clip(heat, 0.0, 1.0).astype(dtype)

    def attrs(self, mpp_level0: float, **extra: Any) -> dict[str, Any]:
        """Self-describing metadata for the heatmap store (gate V8.2)."""
        return {
            "mpp": mpp_level0 * self.cfg.heat_downsample,
            "downsample": self.cfg.heat_downsample,
            "level0_shape": list(self.level0_shape),
            "patch_size": self.cfg.patch_size,
            "sigma": self.cfg.patch_size / self.cfg.sigma_divisor,
            "n_patches": self.n_added,
            **extra,
        }


# --------------------------------------------------------------------------
def _numpy_store(shape: tuple[int, int], dtype: Any, chunk: int, name: str
                 ) -> np.ndarray:
    return np.zeros(shape, dtype=dtype)


def zarr_store_factory(root: Any):
    """Store factory backed by a zarr group (production path)."""
    def _make(shape: tuple[int, int], dtype: Any, chunk: int, name: str):
        import numcodecs  # noqa: PLC0415

        return root.zeros(
            name=name, shape=shape, chunks=(chunk, chunk), dtype=dtype,
            compressor=numcodecs.Blosc(cname="zstd", clevel=5), overwrite=True,
        )
    return _make


def _resize_bool(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    import cv2  # noqa: PLC0415

    out = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]),
                     interpolation=cv2.INTER_NEAREST)
    return out.astype(bool)
