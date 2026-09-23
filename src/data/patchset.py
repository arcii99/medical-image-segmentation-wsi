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
                 seed: int = 1337, aug: dict | None = None):
        self.root = Path(root)
        self.files = self.root / "files"
        # Locate images. The fast path is a flat files/ directory. But an
        # exported patch set can also arrive already unpacked into a nested
        # tree -- e.g. Kaggle expands shards into shards/shard_XXXX/*.jpg --
        # in which case a flat lookup finds nothing. When files/ is absent or
        # empty, build a filename -> path map by walking the whole root once.
        self._map: dict[str, Path] | None = None
        flat_ok = self.files.exists() and any(self.files.glob("*.jpg"))
        if not flat_ok:
            root_p = self.root
            imgs = list(root_p.rglob("*.jpg")) + list(root_p.rglob("*.png"))
            if not imgs:
                raise FileNotFoundError(
                    f"no .jpg/.png found under {self.root}. Extract the "
                    f"shards first, or point patchset_root at the data.")
            self._map = {f.name: f for f in imgs}
            log.info("patchset: nested layout, indexed %d files under %s",
                     len(self._map), self.root)
        man_hits = list(self.root.rglob("manifest.parquet"))
        if not man_hits:
            raise FileNotFoundError(f"manifest.parquet not found under {self.root}")
        man = pd.read_parquet(man_hits[0])
        self.index = man[man.split == split].reset_index(drop=True)
        if self.index.empty:
            raise ValueError(f"no patches for split {split!r} in {self.root}")
        self.tf = (TrainTransform(seed, **(aug or {})) if train
                   else EvalTransform())
        # worker_init reseeds this per worker; without it every worker draws
        # the identical augmentation sequence.
        self.rng = np.random.default_rng(seed)
        log.info("patchset %s: %d patches from %d slides", split,
                 len(self.index), self.index.slide_id.nunique())

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        row = self.index.iloc[i]
        if self._map is not None:
            jp = self._map.get(f"{row.key}.jpg")
            mp = self._map.get(f"{row.key}.png")
        else:
            jp = self.files / f"{row.key}.jpg"
            mp = self.files / f"{row.key}.png"
        img = cv2.imread(str(jp), cv2.IMREAD_COLOR) if jp else None
        msk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp else None
        if img is None or msk is None:
            raise FileNotFoundError(f"missing patch {row.key} under {self.root}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, m = self.tf(img, (msk > 127).astype(np.uint8))
        assert set(np.unique(m)) <= {0.0, 1.0}, "mask is no longer binary"
        return {"image": x, "mask": m,
                "meta": {"slide_id": str(row.slide_id), "key": str(row.key)}}


def build(root: str | Path, split: str, train: bool, seed: int = 1337):
    return PatchSetDataset(root, split, train, seed)
