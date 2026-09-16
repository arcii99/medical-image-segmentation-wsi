"""Dataset over an exported patch set (see scripts/07_export_patchset.py).

Same output contract as :class:`src.data.dataset.PatchDataset` -- a dict with
``image``, ``mask`` and ``meta`` -- so the training loop does not know or care
which one it is reading from.

Differences that matter, and are stated here because they change what the
numbers mean:

* **No coordinate jitter.** The crops are fixed at export time. Geometric and
  photometric augmentation still run, but the model sees the same 512x512
  windows every epoch rather than a fresh nudge each time.
* **No slide access.** Hard-negative mining can only draw from the exported
  normals, not from the full pool.

Both are consequences of making the data portable, not oversights.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    from torch.utils.data import Dataset
except ImportError:                                  # keep importable
    class Dataset:                                   # type: ignore[no-redef]
        pass

from src.data.transforms import EvalTransform, TrainTransform

log = logging.getLogger(__name__)


class PatchSetDataset(Dataset):
    def __init__(self, root: str | Path, split: str, train: bool = True,
                 seed: int = 1337):
        self.root = Path(root)
        self.files = self.root / "files"
        if not self.files.exists():
            raise FileNotFoundError(
                f"{self.files} not found. Extract the shards first:\n"
                f"  mkdir -p {self.files} && "
                f"for t in {self.root}/shards/*.tar; do tar xf $t -C {self.files}; done")
        man = pd.read_parquet(self.root / "manifest.parquet")
        self.index = man[man.split == split].reset_index(drop=True)
        if self.index.empty:
            raise ValueError(f"no patches for split {split!r} in {self.root}")
        self.tf = TrainTransform(seed) if train else EvalTransform()
        log.info("patchset %s: %d patches from %d slides", split,
                 len(self.index), self.index.slide_id.nunique())

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        row = self.index.iloc[i]
        img = cv2.imread(str(self.files / f"{row.key}.jpg"), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(self.files / f"{row.key}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            raise FileNotFoundError(f"missing patch {row.key} under {self.files}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, m = self.tf(img, (msk > 127).astype(np.uint8))
        assert set(np.unique(m)) <= {0.0, 1.0}, "mask is no longer binary"
        return {"image": x, "mask": m,
                "meta": {"slide_id": str(row.slide_id), "key": str(row.key)}}


def build(root: str | Path, split: str, train: bool, seed: int = 1337):
    return PatchSetDataset(root, split, train, seed)
