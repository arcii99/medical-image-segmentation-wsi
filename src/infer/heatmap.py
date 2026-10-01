"""Heatmap persistence and overlay rendering."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def write_zarr(path: str | Path, heat: np.ndarray, attrs: dict[str, Any],
               chunk: int = 1024) -> Path:
    """Write the heatmap as a chunked, compressed zarr array.

    Supports both zarr 2.x and 3.x. Version 3 renamed ``create_dataset`` to
    ``create_array`` and replaced the ``compressor=`` argument with
    ``compressors=``. The old call raises
    ``AttributeError: 'Group' object has no attribute 'create_dataset'`` on
    zarr 3 -- after the GPU work is already done, which is the worst possible
    place to discover a dependency break.
    """
    import zarr
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(path), mode="w")

    major = int(zarr.__version__.split(".")[0])
    if major >= 3:
        from zarr.codecs import BloscCodec
        arr = root.create_array(
            "heat", shape=heat.shape, chunks=(chunk, chunk), dtype=heat.dtype,
            compressors=[BloscCodec(cname="zstd", clevel=5)], overwrite=True)
        arr[:] = heat
    else:
        import numcodecs
        root.create_dataset(
            "heat", data=heat, chunks=(chunk, chunk),
            compressor=numcodecs.Blosc("zstd", clevel=5), overwrite=True)
    root.attrs.update(attrs)
    return path


def read_zarr(path: str | Path):
    """Read back a heatmap written by :func:`write_zarr` (either zarr major)."""
    import zarr
    root = zarr.open(str(path), mode="r")
    return np.asarray(root["heat"]), dict(root.attrs)


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
