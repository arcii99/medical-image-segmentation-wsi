#!/usr/bin/env python
"""Stage 05 -- evaluation. FROC is the primary metric (ADR-013)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from _common import base_parser  # noqa: E402
from src.eval import froc as F  # noqa: E402
from src.eval.report import assert_attrs_match, assert_reportable, write  # noqa: E402
from src.eval.slide_metrics import dice_iou, roc_auc, slide_score  # noqa: E402
from src.infer.postproc import PostProcConfig, apply  # noqa: E402
from src.io import annotations as ann  # noqa: E402
from src.utils.config import load  # noqa: E402
from src.utils.logging import setup  # noqa: E402

log = logging.getLogger("stage05")


def main() -> int:
    p = base_parser("05_evaluate")
    p.add_argument("--run-id", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out", default=None)
    p.add_argument("--allow-dirty", action="store_true")
    a = p.parse_args()
    cfg = load(a.config + ["configs/infer_default.yaml"], a.set)
    setup()

    import torch
    import zarr

    ck = torch.load(Path(cfg.paths.ckpt) / a.run_id / "best.pt", map_location="cpu")
    assert_reportable(ck, a.allow_dirty)
    tau = float(ck["threshold"])

    slides = pd.read_parquet(Path(cfg.paths.index) / "slides.parquet")
    slides = slides[slides.split == a.split]
    dets, n_gt, n_itc, per_slide, labels_y, scores = [], 0, 0, [], [], []

    for _, s in slides.iterrows():
        zpath = Path(cfg.paths.heatmaps) / f"{s.slide_id}.zarr"
        if not zpath.exists():
            log.warning("missing heatmap for %s; run stage 04", s.slide_id); continue
        z = zarr.open(str(zpath))
        assert_attrs_match(dict(z.attrs), ck)
        heat = np.asarray(z["heat"], dtype=np.float32)
        mpp = float(z.attrs["mpp"])

        lab, lesions = apply(heat, tau, mpp, PostProcConfig(**cfg.postproc.to_dict()))
        xml = Path(s.path).with_suffix(".xml")
        geom = ann.load(xml if xml.exists() else None, s.slide_id)
        gt_polys = geom.lesions()

        d, n, n_i = F.match(lesions, gt_polys, s.slide_id, mpp)
        dets += d; n_gt += n; n_itc += n_i
        sc = slide_score(heat)
        labels_y.append(int(bool(gt_polys))); scores.append(sc)
        per_slide.append(dict(slide_id=s.slide_id, n_pred=len(lesions), n_gt=n,
                              score=sc, fp=sum(1 for x in d if not x.hit)))
        log.info("%-18s pred=%d gt=%d (+%d ITC excluded) score=%.3f",
                 s.slide_id, len(lesions), n, n_i, sc)

    fpps, sens = F.curve(dets, n_gt, max(len(per_slide), 1))
    at = F.sensitivity_at(fpps, sens, tuple(cfg.eval.froc_points))
    fps = [r["fp"] for r in per_slide] or [0]
    metrics = {
        "n_slides": len(per_slide), "n_gt_lesions": n_gt,
        "n_itc_excluded": n_itc, "threshold": tau,
        "froc_sensitivity": at, "froc_avg": F.average_sensitivity(at),
        "slide_auc": roc_auc(labels_y, scores),
        "fp_per_normal_slide": float(np.mean(
            [r["fp"] for r, y in zip(per_slide, labels_y) if not y] or [0])),
        "ci95": {"fp_per_slide": F.bootstrap_ci(fps, cfg.eval.bootstrap_n)},
    }
    out = Path(a.out or cfg.paths.reports) / a.run_id
    pd.DataFrame(per_slide).to_csv(out / "per_slide.csv", index=False) if out.exists() \
        else (out.mkdir(parents=True, exist_ok=True),
              pd.DataFrame(per_slide).to_csv(out / "per_slide.csv", index=False))
    write(out, metrics, ck.get("resolved_config"))
    log.info("FROC avg %.4f | AUC %.4f | %s", metrics["froc_avg"],
             metrics["slide_auc"], out / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
