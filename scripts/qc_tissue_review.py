#!/usr/bin/env python
"""Review stage-01 tissue masks against the challenge's own reference masks.

Eyeballing a handful of overlays tells you whether a mask looks plausible. It
does not tell you whether one scanner is systematically under-segmented
relative to another, which is exactly the question that matters when the two
vendors in a cohort differ by 2x in mean tissue fraction.

CAMELYON16 ships its own tissue masks in `CAMELYON16/background_tissue/`,
about 1 MB per slide. Fetch them with

    python scripts/00_fetch_data.py --fetch-reference-tissue

and this script turns the question into three measured numbers per slide:

    IoU        overlap between our mask and the reference
    recall     fraction of reference tissue we found   (low => under-detecting)
    precision  fraction of our mask that is real tissue (low => over-detecting)

The recall/precision split is what separates "my threshold is too strict" from
"this slide genuinely has little tissue". A slide with 1% tissue and recall
0.98 is fine; a slide with 1% tissue and recall 0.40 is losing more than half
of what is there.

    python scripts/qc_tissue_review.py --montage 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_reference(path: Path, shape: tuple[int, int]) -> np.ndarray | None:
    """Reference tissue raster, resized to our mask's resolution."""
    import cv2
    import tifffile
    try:
        with tifffile.TiffFile(str(path)) as tf:
            lvl = tf.series[0].levels
            arr = np.asarray(lvl[-1].asarray() if len(lvl) > 1
                             else tf.series[0].asarray())
    except Exception:  # noqa: BLE001
        return None
    if arr.ndim == 3:
        arr = arr[..., 0]
    ref = (arr > 0).astype(np.uint8)
    return cv2.resize(ref, (shape[1], shape[0]),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def _scores(ours: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    inter = int((ours & ref).sum())
    return {
        "iou": inter / max(int((ours | ref).sum()), 1),
        "recall": inter / max(int(ref.sum()), 1),
        "precision": inter / max(int(ours.sum()), 1),
        "ours_frac": float(ours.mean()),
        "ref_frac": float(ref.mean()),
    }


def montage(reader, ours: np.ndarray, ref: np.ndarray | None, out: Path) -> None:
    import cv2
    lt = reader.thumbnail_level(8.0)
    thumb = reader.read_level(lt)
    h, w = ours.shape
    thumb = cv2.resize(thumb, (w, h), interpolation=cv2.INTER_AREA)

    panels = [thumb]
    ov = thumb.copy()
    ov[ours] = (0.55 * ov[ours] + 0.45 * np.array([0, 200, 80])).astype(np.uint8)
    panels.append(ov)
    if ref is not None:
        rv = thumb.copy()
        rv[ref] = (0.55 * rv[ref] + 0.45 * np.array([80, 120, 255])).astype(np.uint8)
        panels.append(rv)
        diff = thumb.copy()
        diff[ref & ~ours] = (255, 40, 40)      # missed  -> red
        diff[ours & ~ref] = (255, 200, 0)      # extra   -> yellow
        panels.append(diff)

    gap = np.full((h, 6, 3), 255, np.uint8)
    strip = panels[0]
    for pnl in panels[1:]:
        strip = np.hstack([strip, gap, pnl])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))


def main() -> int:
    import imageio.v3 as iio
    import pandas as pd

    from src.io.slide import open_slide

    ap = argparse.ArgumentParser(prog="qc_tissue_review")
    ap.add_argument("--index", default="artifacts/index/slides.parquet")
    ap.add_argument("--tissue-dir", default="artifacts/tissue")
    ap.add_argument("--reference-dir", default="data/reference_tissue")
    ap.add_argument("--montage", type=int, default=0,
                    help="render N montages (worst IoU first)")
    ap.add_argument("--montage-dir", default="artifacts/qc/tissue")
    ap.add_argument("--csv", default="artifacts/qc/tissue_review.csv")
    a = ap.parse_args()

    df = pd.read_parquet(a.index)
    ref_dir = Path(a.reference_dir)
    have_ref = ref_dir.exists() and any(ref_dir.glob("*.tif"))
    if not have_ref:
        print(f"no reference masks in {ref_dir}")
        print("fetch them first:")
        print("  python scripts/00_fetch_data.py --fetch-reference-tissue")
        print("\nfalling back to montages without a reference.\n")

    rows = []
    for _, s in df.iterrows():
        mpath = Path(a.tissue_dir) / f"{s.slide_id}.png"
        if not mpath.exists():
            continue
        ours = iio.imread(mpath) > 0
        row = {"slide_id": s.slide_id, "vendor": s.get("vendor", "?"),
               "slide_class": s.get("slide_class", "?"),
               "tissue_frac": s.tissue_frac, "qc_flag": s.qc_flag,
               "sat_threshold": s.get("sat_threshold"),
               "gray_threshold": s.get("gray_threshold")}
        if have_ref:
            cand = list(ref_dir.glob(f"{s.slide_id}_tissue.tif")) or \
                   list(ref_dir.glob(f"{s.slide_id}.tif"))
            ref = _load_reference(cand[0], ours.shape) if cand else None
            if ref is not None:
                row.update(_scores(ours, ref))
        rows.append(row)

    if not rows:
        sys.exit("no stage-01 masks found")
    out = pd.DataFrame(rows)
    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.csv, index=False)

    print(f"reviewed {len(out)} slides\n")
    if "iou" in out:
        scored = out.dropna(subset=["iou"])
        print("AGAINST THE CHALLENGE REFERENCE MASKS")
        print(f"  overall   IoU {scored.iou.mean():.3f}   "
              f"recall {scored.recall.mean():.3f}   "
              f"precision {scored.precision.mean():.3f}   "
              f"(n={len(scored)})\n")
        g = scored.groupby("vendor")[["iou", "recall", "precision",
                                      "ours_frac", "ref_frac"]].mean()
        print(g.round(3).to_string())
        print("\n  READ IT LIKE THIS")
        print("    low recall on one vendor only  -> threshold too strict there")
        print("    low ref_frac AND high recall   -> that vendor genuinely has")
        print("                                      less tissue; nothing wrong")
        print("    low precision                  -> we are calling glass tissue")
        worst = scored.nsmallest(10, "recall")
        print("\n  WORST RECALL (most under-detected)")
        print(worst[["slide_id", "vendor", "ours_frac", "ref_frac",
                     "recall", "precision", "sat_threshold",
                     "gray_threshold"]].round(3).to_string(index=False))
    else:
        print(out.groupby("vendor").tissue_frac.describe()
              [["count", "mean", "50%", "min", "max"]].round(4).to_string())

    if {"sat_threshold", "gray_threshold"} <= set(out.columns) and \
            out.sat_threshold.notna().any():
        print("\nTHRESHOLDS ACTUALLY USED (ADR-016 floors are sat>=20, grey>=40)")
        print(out.groupby("vendor")[["sat_threshold", "gray_threshold"]]
              .describe()[[("sat_threshold", "mean"), ("sat_threshold", "min"),
                           ("gray_threshold", "mean"), ("gray_threshold", "min")]]
              .round(1).to_string())
        at_floor = ((out.sat_threshold <= 20) | (out.gray_threshold <= 40)).sum()
        print(f"\n  slides where a floor is binding: {at_floor}/{len(out)}")
        if at_floor:
            print("  A binding floor means Otsu wanted a LOWER threshold and was")
            print("  overruled. That is the floor doing its job on a blank slide,")
            print("  or cutting off faint tissue on a real one. Check the recall")
            print("  column for those slides specifically.")

    if a.montage:
        order = (out.dropna(subset=["iou"]).nsmallest(a.montage, "iou")
                 if "iou" in out and out.iou.notna().any()
                 else out.nsmallest(a.montage, "tissue_frac"))
        print(f"\nrendering {len(order)} montages -> {a.montage_dir}")
        paths = dict(zip(df.slide_id, df.path))
        for _, r in order.iterrows():
            ours = iio.imread(Path(a.tissue_dir) / f"{r.slide_id}.png") > 0
            ref = None
            if have_ref:
                cand = list(ref_dir.glob(f"{r.slide_id}_tissue.tif"))
                if cand:
                    ref = _load_reference(cand[0], ours.shape)
            with open_slide(paths[r.slide_id]) as rd:
                montage(rd, ours, ref,
                        Path(a.montage_dir) / f"{r.slide_id}.png")
            print(f"  {r.slide_id}")
        print("\npanels: original | ours (green) | reference (blue) | "
              "diff (red=missed, yellow=extra)")

    print(f"\nwrote {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
