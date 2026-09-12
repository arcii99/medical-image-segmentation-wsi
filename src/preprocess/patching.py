"""Patch grid generation and labelling (ADR-004, ADR-005).

Emits coordinates only -- never pixels.  Patches are scored by exact polygon
area rather than rasterisation: ~1.2 ms/window against ~45 min/slide for
raster fills, and exact to 0.5% (gate V3.4).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from src.infer.sliding_window import _TissueLookup

__all__ = ["PatchConfig", "PatchRecord", "iter_patches"]


@dataclass(frozen=True)
class PatchConfig:
    size: int = 512            # px at the working level
    stride: int = 512          # non-overlapping at train time; jitter adds diversity
    min_tissue: float = 0.10
    tumor_threshold: float = 0.05   # above this => "tumor" bucket (ADR-006)


@dataclass(frozen=True)
class PatchRecord:
    slide_id: str
    x0: int                    # level-0
    y0: int                    # level-0
    level: int
    size: int
    tissue_frac: float
    tumor_frac: float


def iter_patches(
    slide_id: str,
    level_dims: Sequence[tuple[int, int]],
    level_downsamples: Sequence[float],
    level: int,
    tissue: np.ndarray,
    geometry=None,
    cfg: PatchConfig | None = None,
) -> Iterator[PatchRecord]:
    """Yield tissue-gated patch anchors with their tumor fractions."""
    cfg = cfg or PatchConfig()
    lw, lh = level_dims[level]
    s = float(level_downsamples[level])
    w0, h0 = level_dims[0]
    lookup = _TissueLookup(tissue, (h0, w0))
    extent = cfg.size * s

    for gy in range(0, max(lh - cfg.size, 0) + 1, cfg.stride):
        for gx in range(0, max(lw - cfg.size, 0) + 1, cfg.stride):
            x0, y0 = int(round(gx * s)), int(round(gy * s))
            tfrac = lookup.fraction(x0, y0, extent, extent)
            if tfrac < cfg.min_tissue:
                continue
            tumor = geometry.area_fraction(x0, y0, extent, extent) if geometry else 0.0
            yield PatchRecord(slide_id, x0, y0, level, cfg.size, tfrac, tumor)


def bucket(tumor_frac: float, threshold: float = 0.05) -> str:
    """Three-way bucket used by the balanced sampler (ADR-006)."""
    if tumor_frac > threshold:
        return "tumor"
    return "boundary" if tumor_frac > 0.0 else "normal"
