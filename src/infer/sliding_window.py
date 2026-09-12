"""Sliding-window position planning for inference.

Responsibilities:

* Generate grid positions at the working level, in **level-0 coordinates**.
* Gate them by the tissue mask -- typically 20k of 72k positions survive on a
  45k x 105k slide, which is the difference between a 10-minute and a
  40-minute slide.
* **Clamp** margin positions instead of dropping them.  Dropping leaves
  uncovered strips at the right and bottom edges, which read as confident
  negatives (BUG-012).  Clamping produces slightly more overlap there, which
  the stitcher's ``num/den`` normalisation absorbs for free.
* Emit positions in **chunk-aligned order** so the zarr accumulator writes
  each chunk a handful of times rather than revisiting it across the whole
  traversal (ADR-009).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from src.utils.geometry import clamp_window

log = logging.getLogger(__name__)

__all__ = ["PlannerConfig", "Planner", "plan_positions"]


@dataclass(frozen=True)
class PlannerConfig:
    patch_size: int = 512        # px at the working level
    stride: int = 256            # px at the working level; 50% overlap (ADR-004)
    min_tissue: float = 0.02     # fraction of the window that must be tissue
    heat_chunk: int = 1024       # heatmap chunk size, for traversal ordering
    heat_downsample: int = 8


def plan_positions(
    level_dims: Sequence[tuple[int, int]],
    level_downsamples: Sequence[float],
    level: int,
    tissue: np.ndarray | None,
    cfg: PlannerConfig,
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """Return level-0 top-left coordinates, plus a planning summary.

    ``tissue`` is a boolean mask at *any* resolution; it is mapped to level-0
    by its own shape, so the caller does not have to pre-align it.
    """
    lw, lh = level_dims[level]
    s = float(level_downsamples[level])
    size, stride = cfg.patch_size, cfg.stride

    xs = list(range(0, max(lw - size, 0) + 1, stride))
    ys = list(range(0, max(lh - size, 0) + 1, stride))
    # Always include the far edge so the margin is covered.
    if not xs or xs[-1] != lw - size:
        xs.append(max(lw - size, 0))
    if not ys or ys[-1] != lh - size:
        ys.append(max(lh - size, 0))

    n_grid = len(xs) * len(ys)
    w0, h0 = level_dims[0]
    tissue_at_l0 = _TissueLookup(tissue, (h0, w0)) if tissue is not None else None

    kept: list[tuple[int, int]] = []
    for gy in ys:
        for gx in xs:
            x0 = int(round(gx * s))
            y0 = int(round(gy * s))
            x0, y0 = clamp_window(x0, y0, level, size, level_dims, level_downsamples)
            if tissue_at_l0 is not None:
                frac = tissue_at_l0.fraction(x0, y0, size * s, size * s)
                if frac < cfg.min_tissue:
                    continue
            kept.append((x0, y0))

    kept = _chunk_aligned_order(kept, cfg)
    return kept, {"grid": n_grid, "kept": len(kept)}


class _TissueLookup:
    """Fractional tissue coverage of a level-0 window, read off a mask."""

    def __init__(self, mask: np.ndarray, level0_hw: tuple[int, int]):
        self.mask = mask.astype(bool)
        h0, w0 = level0_hw
        mh, mw = self.mask.shape[:2]
        self.sy = h0 / mh
        self.sx = w0 / mw
        # Summed-area table: O(1) per window regardless of window size.
        self.sat = np.pad(
            self.mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0))
        )

    def fraction(self, x0: float, y0: float, w_l0: float, h_l0: float) -> float:
        mh, mw = self.mask.shape[:2]
        c0 = int(np.clip(round(x0 / self.sx), 0, mw))
        r0 = int(np.clip(round(y0 / self.sy), 0, mh))
        c1 = int(np.clip(round((x0 + w_l0) / self.sx), 0, mw))
        r1 = int(np.clip(round((y0 + h_l0) / self.sy), 0, mh))
        if c1 <= c0 or r1 <= r0:
            return 0.0
        total = (self.sat[r1, c1] - self.sat[r0, c1]
                 - self.sat[r1, c0] + self.sat[r0, c0])
        return float(total) / float((r1 - r0) * (c1 - c0))


def _chunk_aligned_order(
    coords: list[tuple[int, int]], cfg: PlannerConfig
) -> list[tuple[int, int]]:
    """Group positions by destination heatmap chunk (ADR-009 write locality)."""
    span = cfg.heat_chunk * cfg.heat_downsample   # level-0 px per chunk
    return sorted(coords, key=lambda xy: (xy[1] // span, xy[0] // span, xy[1], xy[0]))


# --------------------------------------------------------------------------
class Planner:
    """Position planner bound to one slide."""

    def __init__(self, reader, tissue: np.ndarray | None, level: int,
                 cfg: PlannerConfig | None = None):
        self.reader = reader
        self.cfg = cfg or PlannerConfig()
        self.level = level
        self.coords, self.summary = plan_positions(
            reader.level_dims, reader.level_downsamples, level, tissue, self.cfg
        )

    def __len__(self) -> int:
        return len(self.coords)

    def batches(self, batch_size: int) -> Iterator[list[tuple[int, int]]]:
        for i in range(0, len(self.coords), batch_size):
            yield self.coords[i:i + batch_size]

    def read_batch(self, coords: Sequence[tuple[int, int]]) -> np.ndarray:
        """(B, S, S, 3) uint8 for a batch of level-0 coordinates."""
        s = self.cfg.patch_size
        out = np.empty((len(coords), s, s, 3), dtype=np.uint8)
        for i, (x0, y0) in enumerate(coords):
            out[i] = self.reader.read_region(x0, y0, self.level, s, s)
        return out

    def describe(self) -> str:
        return (f"positions planned {self.summary['kept']:,} "
                f"(tissue-gated from {self.summary['grid']:,})")
