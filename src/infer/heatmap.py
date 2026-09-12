"""Heatmap persistence and overlay rendering."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def write_zarr(path: str | Path, heat: np.ndarray, attrs: dict[str, Any],
               chunk: int = 1024) -> Path:
    import numcodecs
    import zarr
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(path), mode="w")
    root.create_dataset("heat", data=heat, chunks=(chunk, chunk),
                        compressor=numcodecs.Blosc("zstd", clevel=5), overwrite=True)
    root.attrs.update(attrs)
    return path


def render_overlay(thumb_rgb: np.ndarray, heat: np.ndarray,
                   labels: np.ndarray | None = None, alpha: float = 0.45) -> np.ndarray:
    """Colourised heatmap composited over the slide thumbnail."""
    h, w = thumb_rgb.shape[:2]
    hm = cv2.resize(np.asarray(heat, np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    colour = cv2.applyColorMap((np.clip(hm, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
    a = (alpha * np.clip(hm, 0, 1))[..., None]
    out = (thumb_rgb.astype(np.float32) * (1 - a) + colour.astype(np.float32) * a)
    out = out.astype(np.uint8)
    if labels is not None:
        lab = cv2.resize(labels.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
        edges = cv2.morphologyEx((lab > 0).astype(np.uint8), cv2.MORPH_GRADIENT,
                                 np.ones((3, 3), np.uint8)).astype(bool)
        out[edges] = (255, 255, 0)
    return out
