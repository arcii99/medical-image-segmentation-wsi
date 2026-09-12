#!/usr/bin/env python
"""Stage 01 -- tissue localisation, slide metadata, split assignment.

Writes artifacts/tissue/{slide_id}.png and artifacts/index/slides.parquet.
A slide that fails is quarantined to failed_slides.csv rather than killing a
multi-hour job; stage 02 then refuses to run until that list is acknowledged,
so nobody silently trains on a subset they did not intend.
"""
from __future__ import annotations

import logging
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio.v3 as iio
import pandas as pd

from _common import base_parser, discover, git_state  # noqa: E402
from src.io.slide import open_slide  # noqa: E402
from src.preprocess.tissue import TissueConfig, tissue_mask  # noqa: E402
from src.utils.config import load  # noqa: E402
from src.utils.logging import setup  # noqa: E402
from src.utils.splits import assign, patient_id  # noqa: E402

log = logging.getLogger("stage01")


def process_one(args):
    path, cfg_d = args
    from src.utils.config import ResolvedConfig
    cfg = ResolvedConfig(cfg_d)
    slide_id = Path(path).stem
    try:
        with open_slide(path, slide_id=slide_id) as r:
            lt = r.thumbnail_level(cfg.data.thumbnail_mpp)
            thumb = r.read_level(lt)
            res = tissue_mask(thumb, r.meta.mpp_at(lt),
                              TissueConfig(**cfg.tissue.to_dict()))
            lw, residual = r.level_for_mpp(cfg.data.target_mpp,
                                           cfg.data.mpp_tolerance)
            out = Path(cfg.paths.tissue) / f"{slide_id}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(out, (res.mask.astype("uint8") * 255))
            patient = patient_id(slide_id, cfg.splits.get("patient_from", "slide_id"))
            return dict(slide_id=slide_id, path=str(path), backend=r.meta.backend,
                        mpp0=r.meta.mpp, level_w=lw, residual_scale=residual,
                        level_t=lt, thumb_h=thumb.shape[0], thumb_w=thumb.shape[1],
                        tissue_frac=res.tissue_frac, pen_frac=res.pen_frac,
                        qc_flag=res.qc_flag, patient_id=patient,
                        split=assign(patient, bounds=tuple(cfg.splits.bounds))), None
    except Exception as e:
        return None, dict(slide_id=slide_id, path=str(path),
                          error=f"{type(e).__name__}: {e}",
                          traceback=traceback.format_exc())


def main() -> int:
    p = base_parser("01_build_tissue_masks")
    p.add_argument("--slides-dir", default=None)
    p.add_argument("--slides", nargs="*", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    cfg = load(a.config, a.set)
    setup()
    idx_dir = Path(a.out or cfg.paths.index)
    if a.out:
        cfg = load(a.config, a.set + [f"paths.tissue={a.out}/tissue",
                                      f"paths.index={a.out}"])
    slides = ([Path(s) for s in a.slides] if a.slides
              else discover(Path(a.slides_dir or cfg.paths.raw_dir)))
    if not slides:
        log.error("no slides found"); return 1
    log.info("processing %d slides", len(slides))

    payload = [(s, cfg.to_dict()) for s in slides]
    rows, fails = [], []
    n_workers = int(cfg.hw.cpu_workers)
    if n_workers > 1 and len(slides) > 1:
        with ProcessPoolExecutor(n_workers) as ex:
            for ok, bad in ex.map(process_one, payload):
                (rows if ok else fails).append(ok or bad)
    else:
        for item in payload:
            ok, bad = process_one(item)
            (rows if ok else fails).append(ok or bad)

    idx_dir = Path(cfg.paths.index); idx_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows)
        sha, dirty = git_state()
        df["git_sha"] = sha; df["dirty"] = dirty
        df.to_parquet(idx_dir / "slides.parquet", index=False)
        for _, r in df.iterrows():
            log.info("%-18s tissue_frac=%.4f qc=%s L%d residual=%.4f",
                     r.slide_id, r.tissue_frac, r.qc_flag, r.level_w, r.residual_scale)
    if fails:
        pd.DataFrame(fails).to_csv(idx_dir / "failed_slides.csv", index=False)
        log.warning("%d slides failed -> %s", len(fails), idx_dir / "failed_slides.csv")
    log.info("ok=%d failed=%d", len(rows), len(fails))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
