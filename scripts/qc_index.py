#!/usr/bin/env python
"""Audit a built patch index before committing GPU hours to it.

Three questions, none of which the stage 02 summary answers:

1. **Did the tissue mask swallow any tumor?** The index only contains patches
   that passed the tissue gate. If a mask clipped a metastasis, those patches
   were never indexed and the label is simply gone -- silently, with no error
   anywhere. Comparing annotated tumor area against tumor area captured in
   the index measures this directly.

2. **How many lesions are smaller than the detection floor?** ADR-012 discards
   connected components below 0.02 mm^2 during post-processing. Any annotated
   lesion below that is unfindable by construction, so it caps FROC before the
   model has done anything. Worth knowing the size of that cap in advance.

3. **Is each split usable?** Not just populated -- does it carry enough tumor
   patches and enough lesion-bearing slides to fit a threshold and compute a
   meaningful FROC?

    python scripts/qc_index.py
    python scripts/qc_index.py --min-lesion-mm2 0.005    # try a lower floor
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import index as idx  # noqa: E402
from src.eval import froc as F  # noqa: E402
from src.io import annotations as ann  # noqa: E402
from src.utils.config import load  # noqa: E402

RULE = "-" * 74


def head(t: str) -> None:
    print(f"\n{RULE}\n{t}\n{RULE}")


def main() -> int:
    ap = argparse.ArgumentParser(prog="qc_index")
    ap.add_argument("--config", nargs="+",
                    default=["configs/base.yaml", "configs/data_camelyon16.yaml"])
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--itc-max-axis-um", type=float, default=None,
                    help="override the ITC exclusion threshold for this report")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    cfg = load(a.config, a.set)
    in_dir = Path(cfg.paths.index)
    patches = idx.load(in_dir / "patches.parquet")
    slides = pd.read_parquet(in_dir / "slides.parquet")
    if a.itc_max_axis_um is not None:
        F.ITC_MAX_AXIS_UM = a.itc_max_axis_um

    # ---------- 1. size and balance ----------
    head("1. INDEX SHAPE")
    print(f"  patches {len(patches):,}   slides {patches.slide_id.nunique()}")
    tf = patches.tumor_frac
    thr = float(cfg.patching.tumor_threshold)
    print(f"  tumor patches  (frac > {thr})  {(tf > thr).sum():,} "
          f"({(tf > thr).mean():.2%})")
    print(f"  boundary       (0 < frac <= {thr}) {((tf > 0) & (tf <= thr)).sum():,}")
    print(f"  pure negative                   {(tf == 0).sum():,}")

    m = patches.merge(slides[["slide_id", "split", "slide_class"]],
                      on="slide_id", how="left", suffixes=("", "_s"))
    m["split"] = m["split_s"].fillna(m["split"]) if "split_s" in m else m["split"]

    head("2. PATCHES PER SPLIT AND CLASS")
    t = pd.crosstab(m.slide_class, m.split)
    print(t.to_string())
    print()
    tum = m[m.tumor_frac > thr]
    print("  tumor patches per split:")
    for s in ("train", "val", "test"):
        sub = tum[tum.split == s]
        n_sl = sub.slide_id.nunique()
        print(f"    {s:6s} {len(sub):8,} patches across {n_sl:3d} slides")
        if n_sl < 5:
            print(f"           WARNING: {n_sl} tumor slides is thin for this split")

    # ---------- 3. did the tissue mask lose any tumor? ----------
    head("3. TUMOR AREA: ANNOTATED vs CAPTURED IN THE INDEX")
    print("  A shortfall means the tissue mask excluded annotated tumor, so")
    print("  those labels never entered the index.\n")
    rows = []
    for s in slides[slides.tumor_mm2 > 0].itertuples():
        p = patches[patches.slide_id == s.slide_id]
        if p.empty:
            rows.append((s.slide_id, s.tumor_mm2, 0.0, 0.0)); continue
        # stride == size in stage 02, so patches do not overlap
        px = float(p["size"].iloc[0])
        ds = float(slides.loc[slides.slide_id == s.slide_id, "residual_scale"].iloc[0])
        mpp_w = s.mpp0 * (2.0 ** int(p["level"].iloc[0]))
        area_per_patch = (px * mpp_w / 1000.0) ** 2        # mm^2
        captured = float((p.tumor_frac * area_per_patch).sum())
        rows.append((s.slide_id, s.tumor_mm2, captured,
                     captured / s.tumor_mm2 if s.tumor_mm2 else 0.0))
    cov = pd.DataFrame(rows, columns=["slide_id", "annotated_mm2",
                                      "indexed_mm2", "coverage"])
    print(f"  slides with annotation: {len(cov)}")
    print(f"  coverage  median {cov.coverage.median():.3f}   "
          f"p10 {cov.coverage.quantile(0.10):.3f}   min {cov.coverage.min():.3f}")
    bad = cov[cov.coverage < 0.80].sort_values("coverage")
    if len(bad):
        print(f"\n  {len(bad)} slides below 80% tumor coverage:")
        print(bad.head(12).to_string(index=False,
                                     float_format=lambda v: f"{v:.4f}"))
        print("\n  Inspect these with:")
        print("    python scripts/qc_tissue_montage.py --slides "
              f"{' '.join(bad.slide_id.head(4))} --cell 900 --out artifacts/qc_lost.png")
    else:
        print("\n  all annotated slides at >=80% coverage -- the tissue mask is "
              "not clipping tumor")

    # ---------- 4. lesion size, by the CAMELYON16 convention ----------
    head("4. LESION SIZE DISTRIBUTION (major axis, CAMELYON16 convention)")
    print("  Lesions are classified by LARGEST DIMENSION, not area, because")
    print("  that is how they are staged clinically and how the CAMELYON16")
    print("  benchmark scores them:")
    print(f"    macro   >= {F.MACRO_MIN_AXIS_UM/1000:.1f} mm     stages the node positive")
    print(f"    micro   >= {F.MICRO_MIN_AXIS_UM/1000:.1f} mm     stages the node positive")
    print(f"    ITC      < {F.MICRO_MIN_AXIS_UM/1000:.1f} mm     does NOT stage the node positive")
    print(f"\n  FROC excludes lesions with major axis < {F.ITC_MAX_AXIS_UM:.0f} um:")
    print("    not in the denominator, and a prediction on one is not a false")
    print("    positive either.\n")

    axes, cls, per_slide = [], [], []
    for s_ in slides[slides.tumor_mm2 > 0].itertuples():
        xml = Path(s_.path).with_suffix(".xml")
        if not xml.exists():
            continue
        g = ann.load(xml, s_.slide_id)
        a_um = [F.major_axis_um(poly, s_.mpp0) for poly in g.lesions()]
        c = [F.classify(poly, s_.mpp0) for poly in g.lesions()]
        axes.extend(a_um); cls.extend(c)
        ev = sum(1 for x in a_um if x >= F.ITC_MAX_AXIS_UM)
        per_slide.append((s_.slide_id, s_.split, len(a_um), ev,
                          max(a_um) if a_um else 0.0))
    if not axes:
        print("  no lesions found"); return 0

    arr = np.array(axes)
    print(f"  annotated polygons total {len(arr):,}")
    for q in (0.05, 0.25, 0.50, 0.75, 0.95, 1.00):
        print(f"    p{int(q*100):<3d} {np.quantile(arr, q):10.1f} um")

    counts = pd.Series(cls).value_counts()
    print("\n  clinical categories:")
    for k in ("macro", "micro", "itc"):
        n = int(counts.get(k, 0))
        print(f"    {k:6s} {n:6,}  ({n/len(arr):6.1%})")

    ev_mask = arr >= F.ITC_MAX_AXIS_UM
    print(f"\n  EVALUABLE under CAMELYON16 : {ev_mask.sum():,} / {len(arr):,} "
          f"({ev_mask.mean():.1%})")
    print(f"  excluded as ITC             : {(~ev_mask).sum():,} "
          f"({(~ev_mask).mean():.1%})")
    print("\n  The ITC exclusion is NOT a cap on achievable FROC -- those")
    print("  lesions are removed from the denominator, so sensitivity of 1.0")
    print("  remains reachable.")

    ps = pd.DataFrame(per_slide, columns=["slide_id", "split", "n_polygons",
                                          "n_evaluable", "largest_axis_um"])
    print("\n  evaluable lesions per split:")
    for sp in ("train", "val", "test"):
        sub = ps[ps.split == sp]
        n_les = int(sub.n_evaluable.sum())
        n_sl = int((sub.n_evaluable > 0).sum())
        print(f"    {sp:6s} {n_les:5,} lesions across {n_sl:3d} slides")
        if sp == "test" and n_les < 30:
            print("           WARNING: fewer than ~30 lesions makes each FROC")
            print("           operating point coarse; report wide CIs")

    itc_only = ps[(ps.n_evaluable == 0) & (ps.n_polygons > 0)]
    if len(itc_only):
        print(f"\n  {len(itc_only)} slides annotated but ITC-only (no evaluable "
              "lesion). They still count as tumor slides for slide-level AUC:")
        print(itc_only.to_string(index=False,
                                 float_format=lambda v: f"{v:.1f}"))

    if a.csv:
        cov.merge(ps, on="slide_id", how="outer").to_csv(a.csv, index=False)
        print(f"\nwrote {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
