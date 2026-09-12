#!/usr/bin/env python
"""Stage 04 -- sliding-window inference and heatmap reconstruction.

The model architecture is built from the **checkpoint's** config, never from
the current config file, and weights load with strict=True.  Building from the
live config while loading old weights either raises loudly (fine) or, if
anyone ever sets strict=False, loads partially and silently (catastrophic).
"""
from __future__ import annotations

import logging
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd

from _common import base_parser, git_state  # noqa: E402
from src.infer.heatmap import render_overlay, write_zarr  # noqa: E402
from src.infer.postproc import PostProcConfig, apply, threshold_from_ckpt  # noqa: E402
from src.infer.sliding_window import Planner, PlannerConfig  # noqa: E402
from src.infer.stitch import GaussianStitcher, StitchConfig  # noqa: E402
from src.io.slide import open_slide  # noqa: E402
from src.preprocess.tissue import TissueConfig, tissue_mask  # noqa: E402
from src.utils.config import ResolvedConfig, load  # noqa: E402
from src.utils.logging import setup  # noqa: E402

log = logging.getLogger("stage04")


def main() -> int:
    p = base_parser("04_infer_slide")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--slide", required=True)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    cfg = load(a.config + ["configs/infer_default.yaml"], a.set)
    setup()

    import torch
    from src.models.registry import build

    ck = torch.load(a.ckpt, map_location="cpu")
    mcfg = ResolvedConfig(ck["resolved_config"]).model
    model = build(mcfg)
    model.load_state_dict(ck.get("ema_state_dict") or ck["state_dict"], strict=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(dev)
    tau = threshold_from_ckpt(ck)

    out_dir = Path(a.out or cfg.paths.artifacts)
    slide_id = Path(a.slide).stem
    with open_slide(a.slide, slide_id=slide_id) as r:
        tpath = Path(cfg.paths.tissue) / f"{slide_id}.png"
        if tpath.exists():
            tissue = iio.imread(tpath) > 0
        else:
            lt = r.thumbnail_level(cfg.data.thumbnail_mpp)
            tissue = tissue_mask(r.read_level(lt), r.meta.mpp_at(lt),
                                 TissueConfig(**cfg.tissue.to_dict()),
                                 strict=False).mask

        lw, _ = r.level_for_mpp(cfg.data.target_mpp)
        pcfg = PlannerConfig(patch_size=cfg.patching.size, stride=cfg.infer.stride,
                             min_tissue=cfg.infer.min_tissue,
                             heat_chunk=cfg.infer.chunk,
                             heat_downsample=cfg.infer.heat_downsample)
        planner = Planner(r, tissue, lw, pcfg)
        log.info(planner.describe())

        h0, w0 = r.level_dims[0][1], r.level_dims[0][0]
        st = GaussianStitcher((h0, w0), r.level_downsamples[lw],
                              StitchConfig(patch_size=cfg.patching.size,
                                           heat_downsample=cfg.infer.heat_downsample,
                                           sigma_divisor=cfg.infer.sigma_divisor,
                                           chunk=cfg.infer.chunk))
        grid = []
        from src.data.transforms import normalize
        for coords in planner.batches(cfg.infer.batch_size):
            imgs = planner.read_batch(coords)
            x = torch.from_numpy(np.stack([normalize(i) for i in imgs])).to(dev)
            with torch.no_grad(), torch.autocast(device_type=dev.type,
                                                 dtype=torch.bfloat16,
                                                 enabled=dev.type == "cuda"):
                logits = model(x)
                if cfg.infer.tta:
                    for dims in ((-1,), (-2,), (-1, -2)):
                        logits = logits + torch.flip(model(torch.flip(x, dims)), dims)
                    logits = logits / 4.0
                probs = torch.sigmoid(logits.float()).cpu().numpy()
            st.add(probs, coords)
            for (cx, cy), pr in zip(coords, probs[:, 0]):
                grid.append(dict(x0=cx, y0=cy, prob_mean=float(pr.mean()),
                                 prob_max=float(pr.max())))

        frac, _ = st.coverage_report(tissue)
        log.info("coverage: uncovered tissue %.2f%%", frac * 100)
        heat = st.finalize(tissue=tissue)

        sha, dirty = git_state()
        attrs = st.attrs(r.meta.mpp, model_run_id=ck.get("run_id"), git_sha=sha,
                         threshold=tau, tta=bool(cfg.infer.tta),
                         stride=cfg.infer.stride)
        zpath = write_zarr(out_dir / "heatmaps" / f"{slide_id}.zarr", heat, attrs,
                           cfg.infer.chunk)
        pd.DataFrame(grid).to_parquet(
            out_dir / "heatmaps" / f"{slide_id}_grid.parquet", index=False)

        labels, lesions = apply(heat.astype(np.float32), tau, attrs["mpp"],
                                PostProcConfig(**cfg.postproc.to_dict()))
        log.info("lesions after post-processing: %d (total %.4f mm^2)",
                 len(lesions), sum(l.area_mm2 for l in lesions))

        lt = r.thumbnail_level(cfg.data.thumbnail_mpp)
        ov = render_overlay(r.read_level(lt), heat.astype(np.float32), labels)
        opath = out_dir / "overlays" / f"{slide_id}.png"
        opath.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(opath, ov)
        log.info("heatmap %s (%s)  overlay %s", zpath, heat.shape, opath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
