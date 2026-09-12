#!/usr/bin/env python
"""Stage 02 -- build the patch coordinate index (the pipeline's narrow waist).

Emits coordinates only; ~40 bytes/row, so a 400-slide cohort is ~120 MB
instead of the ~2 TB a pre-extracted patch dump would cost (ADR-005).
"""
from __future__ import annotations

import logging
from pathlib import Path

import imageio.v3 as iio
import pandas as pd

from _common import base_parser  # noqa: E402
from src.data import index as idx  # noqa: E402
from src.io import annotations as ann  # noqa: E402
from src.io.slide import open_slide  # noqa: E402
from src.preprocess.patching import PatchConfig, iter_patches  # noqa: E402
from src.utils.config import load  # noqa: E402
from src.utils.logging import setup  # noqa: E402

log = logging.getLogger("stage02")


def main() -> int:
    p = base_parser("02_build_patch_index")
    p.add_argument("--out", default=None)
    p.add_argument("--allow-failed", action="store_true")
    p.add_argument("--allow-single-split", action="store_true",
                   help="skip the split-usability gate (synthetic fixtures only)")
    a = p.parse_args()
    cfg = load(a.config, a.set)
    setup()

    in_dir = Path(cfg.paths.index)
    out_dir = Path(a.out or cfg.paths.index); out_dir.mkdir(parents=True, exist_ok=True)
    slides = pd.read_parquet(in_dir / "slides.parquet")

    failed = in_dir / "failed_slides.csv"
    if failed.exists() and not a.allow_failed:
        log.error("%s is present: %d slides failed stage 01. Acknowledge with "
                  "--allow-failed, or fix them. Refusing to index a silent subset.",
                  failed, len(pd.read_csv(failed)))
        return 1

    pcfg = PatchConfig(size=cfg.patching.size, stride=cfg.patching.stride,
                       min_tissue=cfg.patching.min_tissue,
                       tumor_threshold=cfg.patching.tumor_threshold)
    rows = []
    for _, s in slides.iterrows():
        tissue = iio.imread(Path(cfg.paths.tissue) / f"{s.slide_id}.png") > 0
        with open_slide(s.path, slide_id=s.slide_id) as r:
            xml = Path(s.path).with_suffix(".xml")
            geom = ann.load(xml if xml.exists() else None, s.slide_id,
                            r.level_dims[0])
            n0 = len(rows)
            for rec in iter_patches(s.slide_id, r.level_dims, r.level_downsamples,
                                    int(s.level_w), tissue, geom, pcfg):
                rows.append({**rec.__dict__, "split": s.split})
            log.info("%-18s patches=%-7d tumor_area=%.4f mm^2",
                     s.slide_id, len(rows) - n0, geom.area_mm2(s.mpp0))

    if not rows:
        log.error("index is empty"); return 1
    df = pd.DataFrame(rows)
    idx.assert_no_split_leakage(df)                       # gate V4.2, blocking
    if not a.allow_single_split:
        idx.assert_splits_usable(df, pcfg.tumor_threshold)  # gate V4.6
    path = idx.save(df, out_dir / "patches.parquet")

    tumor = (df.tumor_frac > pcfg.tumor_threshold).sum()
    log.info("total patches %d | tumor %d (%.2f%%) | slides with tumor %d",
             len(df), tumor, 100 * tumor / len(df),
             df.loc[df.tumor_frac > 0, "slide_id"].nunique())
    log.info("splits: %s", df.groupby("split").size().to_dict())
    log.info("wrote %s (index_hash %s)", path, idx.index_hash(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
