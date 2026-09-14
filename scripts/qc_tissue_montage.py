#!/usr/bin/env python
"""Render a contact sheet of tissue masks for visual inspection.

Stage 01 produces a number per slide. A number cannot tell you whether the
mask traced the lymph node or traced a coverslip edge, and no downstream
metric will either -- a bad mask just quietly removes tissue from the index
or adds glass to it.

This builds one PNG showing each slide's thumbnail with the mask boundary
drawn over it, so a whole cohort can be checked in one look.

    # the extremes and the flagged ones -- check these first
    python scripts/qc_tissue_montage.py --mode suspect --out artifacts/qc_suspect.png

    # everything, for a full sweep
    python scripts/qc_tissue_montage.py --mode all --out artifacts/qc_all.png

    # one slide, large
    python scripts/qc_tissue_montage.py --slides tumor_016 --cell 900 \\
        --out artifacts/qc_tumor_016.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.slide import open_slide  # noqa: E402
from src.utils.config import load  # noqa: E402

OUTLINE = (255, 40, 40)      # RGB
LABEL_H = 34


def render_cell(row, tissue_dir: Path, cell: int, fill_alpha: float) -> np.ndarray:
    """Thumbnail with the mask boundary drawn on it, letterboxed to `cell`."""
    with open_slide(row.path, slide_id=row.slide_id) as r:
        thumb = r.read_level(int(row.level_t))

    mask_path = tissue_dir / f"{row.slide_id}.png"
    mask = (iio.imread(mask_path) > 0) if mask_path.exists() else None

    h, w = thumb.shape[:2]
    scale = min(cell / w, (cell - LABEL_H) / h)
    tw, th = max(int(w * scale), 1), max(int(h * scale), 1)
    img = cv2.resize(thumb, (tw, th), interpolation=cv2.INTER_AREA)

    if mask is not None:
        m = cv2.resize(mask.astype(np.uint8), (tw, th),
                       interpolation=cv2.INTER_NEAREST)
        if fill_alpha > 0:
            tint = img.astype(np.float32).copy()
            tint[m > 0] = tint[m > 0] * (1 - fill_alpha) + \
                np.array(OUTLINE, np.float32) * fill_alpha
            img = tint.astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, OUTLINE, max(1, cell // 300))

    canvas = np.full((cell, cell, 3), 248, np.uint8)
    y0, x0 = (cell - LABEL_H - th) // 2, (cell - tw) // 2
    canvas[y0:y0 + th, x0:x0 + tw] = img

    flag = str(getattr(row, "qc_flag", "ok"))
    line1 = f"{row.slide_id}"
    line2 = (f"frac {row.tissue_frac:.4f}"
             + (f"  {row.tissue_mm2:.0f}mm2" if hasattr(row, "tissue_mm2") else "")
             + (f"  {flag}" if flag != "ok" else ""))
    colour = (0, 0, 0) if flag == "ok" else (200, 0, 0)
    fs = cell / 900
    cv2.putText(canvas, line1, (6, cell - LABEL_H + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * fs, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, line2, (6, cell - LABEL_H + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42 * fs, colour, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (cell - 1, cell - 1), (210, 210, 210), 1)
    return canvas


def select(df: pd.DataFrame, mode: str, n: int) -> pd.DataFrame:
    if mode == "all":
        return df
    if mode == "sample":
        return df.sample(min(n, len(df)), random_state=0).sort_values("tissue_frac")
    # "suspect": the extremes plus anything flagged -- where errors live
    parts = [df.nsmallest(n // 3, "tissue_frac"),
             df.nlargest(n // 3, "tissue_frac")]
    flagged = df[df.get("qc_flag", pd.Series("ok", index=df.index)) != "ok"]
    if len(flagged):
        parts.append(flagged.head(n // 3))
    out = pd.concat(parts).drop_duplicates("slide_id")
    return out.sort_values("tissue_frac")


def main() -> int:
    ap = argparse.ArgumentParser(prog="qc_tissue_montage")
    ap.add_argument("--config", nargs="+",
                    default=["configs/base.yaml", "configs/data_camelyon16.yaml"])
    ap.add_argument("--mode", choices=["suspect", "sample", "all"],
                    default="suspect")
    ap.add_argument("--slides", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--cell", type=int, default=320)
    ap.add_argument("--cols", type=int, default=0, help="0 = square-ish")
    ap.add_argument("--fill-alpha", type=float, default=0.18)
    ap.add_argument("--out", default="artifacts/qc_tissue.png")
    ap.add_argument("--set", action="append", default=[], metavar="key=value")
    a = ap.parse_args()

    cfg = load(a.config, a.set)
    df = pd.read_parquet(Path(cfg.paths.index) / "slides.parquet")
    tissue_dir = Path(cfg.paths.tissue)

    sel = (df[df.slide_id.isin(a.slides)] if a.slides
           else select(df, a.mode, a.n))
    if sel.empty:
        sys.exit("no slides selected")
    print(f"rendering {len(sel)} slides at {a.cell}px ...", flush=True)

    cells = []
    for i, row in enumerate(sel.itertuples(), 1):
        try:
            cells.append(render_cell(row, tissue_dir, a.cell, a.fill_alpha))
        except Exception as e:  # noqa: BLE001
            print(f"  {row.slide_id}: {type(e).__name__}: {e}")
        if i % 10 == 0:
            print(f"  {i}/{len(sel)}", flush=True)
    if not cells:
        sys.exit("nothing rendered")

    cols = a.cols or int(np.ceil(np.sqrt(len(cells))))
    rows = int(np.ceil(len(cells) / cols))
    sheet = np.full((rows * a.cell, cols * a.cell, 3), 255, np.uint8)
    for i, c in enumerate(cells):
        r, k = divmod(i, cols)
        sheet[r * a.cell:(r + 1) * a.cell, k * a.cell:(k + 1) * a.cell] = c

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, sheet)
    print(f"\nwrote {out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    print("\nWhat to look for:")
    print("  - the outline should hug the tissue, not the coverslip or the")
    print("    slide edge")
    print("  - pale or faded sections should still be inside the outline")
    print("  - pen marks, air bubbles and dust should be OUTSIDE it")
    print("  - the two scanner vendors should not look systematically")
    print("    different; if one is consistently tighter, that is a threshold")
    print("    problem rather than biology")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
