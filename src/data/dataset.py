"""Lazy patch dataset (ADR-005).

Pixels are materialised inside DataLoader workers from level-0 coordinates.
Slide handles are opened **lazily, keyed by PID**: a handle inherited across
fork() is shared by every worker, which interleaves seeks and returns torn or
duplicated patches (BUG-002).
"""
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from torch.utils.data import Dataset
except ImportError:                                   # keep module importable
    class Dataset:                                    # type: ignore[no-redef]
        pass

from src.io import annotations as ann
from src.io.slide import open_slide
from src.data.transforms import EvalTransform, TrainTransform
from src.utils.geometry import needs_resample

log = logging.getLogger(__name__)
MAX_OPEN_HANDLES = 64          # guards against fd exhaustion (BUG-014)


class PatchDataset(Dataset):
    def __init__(self, index, slide_paths: dict[str, Path],
                 annotation_paths: dict[str, Path] | None = None,
                 train: bool = True, jitter: int = 128, seed: int = 1337,
                 residual_scale: dict[str, float] | None = None):
        self.index = index.reset_index(drop=True)
        self.slide_paths = slide_paths
        self.annotation_paths = annotation_paths or {}
        self.train = train
        self.jitter = jitter if train else 0
        self.residual_scale = residual_scale or {}
        self.tf = TrainTransform(seed) if train else EvalTransform()
        self._readers: "OrderedDict[tuple[int,str], Any]" = OrderedDict()
        self._geoms: dict[str, Any] = {}
        self._pid = os.getpid()
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.index)

    # -- per-process resources -----------------------------------------
    def _reader(self, slide_id: str):
        if os.getpid() != self._pid:                  # inherited across fork
            self._readers.clear(); self._pid = os.getpid()
        key = (self._pid, slide_id)
        if key in self._readers:
            self._readers.move_to_end(key)
            return self._readers[key]
        r = open_slide(self.slide_paths[slide_id], slide_id=slide_id)
        self._readers[key] = r
        while len(self._readers) > MAX_OPEN_HANDLES:
            _, old = self._readers.popitem(last=False)
            old.close()
        return r

    def _geom(self, slide_id: str, reader):
        if slide_id not in self._geoms:
            self._geoms[slide_id] = ann.load(
                self.annotation_paths.get(slide_id), slide_id, reader.level_dims[0])
        return self._geoms[slide_id]

    def __getstate__(self) -> dict[str, Any]:
        s = self.__dict__.copy()
        s["_readers"] = OrderedDict()                 # BUG-002
        s["_geoms"] = {}
        return s

    # -- item ----------------------------------------------------------
    def __getitem__(self, i: int):
        row = self.index.iloc[i]
        sid = str(row.slide_id)
        reader = self._reader(sid)
        size, level = int(row["size"]), int(row.level)
        s = reader.level_downsamples[level]

        x0, y0 = int(row.x0), int(row.y0)
        if self.jitter:
            jx, jy = self.rng.integers(-self.jitter, self.jitter + 1, 2)
            x0 = max(0, x0 + int(jx * s)); y0 = max(0, y0 + int(jy * s))

        img = reader.read_region(x0, y0, level, size, size)
        mask = self._geom(sid, reader).rasterize(x0, y0, size, size, s)

        scale = self.residual_scale.get(sid, 1.0)
        if needs_resample(scale):
            d = (int(round(size / scale)), int(round(size / scale)))
            img = cv2.resize(img, d, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, d, interpolation=cv2.INTER_NEAREST)  # BUG-013
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

        x, m = self.tf(img, mask)
        assert set(np.unique(m)) <= {0.0, 1.0}, "mask is no longer binary"
        return {"image": x, "mask": m,
                "meta": {"slide_id": sid, "x0": x0, "y0": y0}}


def worker_init(worker_id: int) -> None:
    """Reset per-process state and decorrelate RNGs (BUG-002)."""
    import torch
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    ds._readers.clear(); ds._geoms.clear(); ds._pid = os.getpid()
    ds.rng = np.random.default_rng(1337 + worker_id)
    if hasattr(ds.tf, "rng"):
        ds.tf.rng = np.random.default_rng(9973 + worker_id)
