#!/usr/bin/env python
"""Fast pre-flight audit of a downloaded cohort.

Reads only metadata and annotation XML -- no pixel decoding -- so a few
hundred slides take seconds. Run this BEFORE stage 01, because the questions
it answers are the ones that make a whole run worthless if answered wrong:

* Does every slide open with an available backend?
* Does every slide report microns-per-pixel? A slide without it is rejected
  by ADR-002 rather than guessed at, so a cohort where many lack it needs a
  decision before, not after, an 18-minute preprocessing run.
* Is the physical scale consistent? CAMELYON16 spans two scanners; the
  pipeline resolves the working level per slide, but a wildly bimodal MPP
  distribution is worth seeing.
* Is the pyramid deep enough to reach an ~8 um/px thumbnail? A shallow
  pyramid silently changes the effective scale of the tissue-detection
  morphology (BUG-019).
* Do the annotations parse, use the expected groups, and sit inside the
  slide's coordinate space?

    python scripts/qc_probe.py --slides-dir data/raw/camelyon16
    python scripts/qc_probe.py --slides-dir data/raw/camelyon16 --csv qc.csv
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io import annotations as ann  # noqa: E402
from src.io.slide import open_slide  # noqa: E402
from src.utils.splits import assign, patient_id  # noqa: E402

EXTS = (".tif", ".tiff", ".svs", ".ndpi")
# masks/ and background_tissue/ rasters share the slide naming scheme but are
# single-channel label images, not WSIs.
NOT_WSI = ("_mask", "_tissue")


def is_wsi(p) -> bool:
    return (p.suffix.lower() in EXTS
            and not p.stem.lower().endswith(NOT_WSI))


def probe(path: Path, target_mpp: float, thumb_mpp: float, rule: str) -> dict:
    row: dict = {"slide": path.stem, "ok": False, "error": ""}
    try:
        with open_slide(path) as r:
            m = r.meta
            row.update(backend=m.backend, vendor=m.vendor,
                       levels=m.level_count, mpp=round(m.mpp, 4),
                       w0=m.level_dims[0][0], h0=m.level_dims[0][1],
                       gpx=round(m.level_dims[0][0] * m.level_dims[0][1] / 1e9, 2))
            lw, resid = r.level_for_mpp(target_mpp)
            row.update(level_w=lw, residual=round(resid, 4),
                       resample=abs(resid - 1) > 0.05)
            lt = r.thumbnail_level(thumb_mpp)
            row.update(level_t=lt, thumb_mpp=round(m.mpp_at(lt), 2),
                       thumb_px=f"{m.level_dims[lt][0]}x{m.level_dims[lt][1]}")
            # non-power-of-two check (INV-2)
            row["ds_exact_pow2"] = all(
                abs(m.level_downsamples[i] - 2.0 ** i) < 1e-9
                for i in range(m.level_count))

            xml = path.with_suffix(".xml")
            if xml.exists():
                g = ann.load(xml, path.stem, m.level_dims[0])
                row.update(has_xml=True, groups=",".join(
                    f"{k}:{v}" for k, v in sorted(g.n_groups.items())),
                    lesions=g.n_lesions,
                    tumor_mm2=round(g.area_mm2(m.mpp), 4),
                    invalid_polys=g.n_invalid_before_repair)
            else:
                row.update(has_xml=False, groups="", lesions=0,
                           tumor_mm2=0.0, invalid_polys=0)
            n = path.stem.lower()
            row["name_class"] = ("tumor" if n.startswith("tumor")
                                 else "normal" if n.startswith("normal")
                                 else "test" if n.startswith("test") else "other")
            row["patient"] = patient_id(path.stem, rule)
            row["split"] = assign(row["patient"])
            row["ok"] = True
    except Exception as e:  # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def main() -> int:
    ap = argparse.ArgumentParser(prog="qc_probe")
    ap.add_argument("--slides-dir", required=True)
    ap.add_argument("--target-mpp", type=float, default=0.50)
    ap.add_argument("--thumb-mpp", type=float, default=8.0)
    ap.add_argument("--patient-from", default="slide_id")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    allf = sorted(Path(a.slides_dir).rglob("*"))
    skipped = [p for p in allf if p.suffix.lower() in EXTS and not is_wsi(p)]
    files = [p for p in allf if is_wsi(p)]
    if skipped:
        print(f"ignoring {len(skipped)} non-WSI rasters "
              f"(*_mask.tif / *_tissue.tif)\n")
    if a.limit:
        files = files[:a.limit]
    if not files:
        sys.exit(f"no slides found under {a.slides_dir}")
    print(f"probing {len(files)} slides ...\n", flush=True)

    rows = [probe(p, a.target_mpp, a.thumb_mpp, a.patient_from) for p in files]
    ok = [r for r in rows if r["ok"]]
    bad = [r for r in rows if not r["ok"]]

    # ---- failures first ----
    if bad:
        print(f"UNREADABLE: {len(bad)} slides\n")
        kinds = collections.Counter(r["error"].split(":")[0] for r in bad)
        for k, n in kinds.most_common():
            print(f"  {n:4d}  {k}")
        print()
        for r in bad[:5]:
            print(f"  {r['slide']:24s} {r['error']}")
        if len(bad) > 10:
            print(f"  ... and {len(bad)-10} more")
        print()

    if not ok:
        print("NOTHING READABLE. Stop here.")
        return 1

    def col(key):
        return [r[key] for r in ok]

    print(f"READABLE: {len(ok)} slides\n")
    print("  backends :", dict(collections.Counter(col("backend"))))
    print("  vendors  :", dict(collections.Counter(col("vendor"))))
    print("  levels   :", dict(sorted(collections.Counter(col("levels")).items())))

    mpps = collections.Counter(col("mpp"))
    print("\n  MICRONS PER PIXEL at level 0")
    for v, n in sorted(mpps.items()):
        print(f"    {v:.4f}  x{n}")
    if len(mpps) > 1:
        print("    (more than one scanner -- this is expected for CAMELYON16,")
        print("     and is exactly why the working level is resolved per slide)")

    lw = collections.Counter(col("level_w"))
    print("\n  WORKING LEVEL chosen for "
          f"{a.target_mpp} um/px: {dict(sorted(lw.items()))}")
    n_res = sum(col("resample"))
    print(f"    resampling needed on {n_res}/{len(ok)} slides "
          f"({n_res/len(ok):.0%})")
    if len(lw) > 1:
        print("    NOTE: different level indices across the cohort. Hard-coding")
        print("    a level would have trained on two different physical scales.")

    tm = collections.Counter(col("thumb_mpp"))
    print(f"\n  THUMBNAIL level for ~{a.thumb_mpp} um/px: "
          f"{dict(sorted(tm.items()))}")
    far = [r for r in ok if r["thumb_mpp"] > 2 * a.thumb_mpp
           or r["thumb_mpp"] < a.thumb_mpp / 2]
    if far:
        print(f"    WARNING: {len(far)} slides resolve more than 2x from target.")
        print("    Tissue morphology is specified in microns, so the effective")
        print("    structuring element changes size on these (BUG-019).")

    pow2 = sum(col("ds_exact_pow2"))
    print(f"\n  pyramid downsamples exactly powers of two: {pow2}/{len(ok)}")
    if pow2 < len(ok):
        print("    (as expected -- this is what INV-2 exists for)")

    sizes = sorted(col("gpx"))
    print(f"\n  slide size (gigapixels at level 0): "
          f"min {sizes[0]}  median {sizes[len(sizes)//2]}  max {sizes[-1]}")

    # ---- annotations ----
    with_xml = [r for r in ok if r["has_xml"]]
    print(f"\n  ANNOTATIONS: {len(with_xml)}/{len(ok)} slides have XML")
    if with_xml:
        groups = collections.Counter()
        for r in with_xml:
            for g in r["groups"].split(","):
                if g:
                    groups[g.split(":")[0]] += 1
        print("    groups seen:", dict(groups.most_common()))
        # Ask the parser what it knows rather than hardcoding a list here.
        # A duplicated vocabulary drifts: CAMELYON16 uses "Tumor"/"Exclusion",
        # which the parser handles, but a stale copy in this script reported
        # them as unrecognised and warned about a bug that did not exist.
        known = {g.lower() for g in ann.TUMOR_GROUPS + ann.EXCLUSION_GROUPS}
        unexpected = {g for g in groups if g.lower() not in known}
        if unexpected:
            print(f"    WARNING: unexpected group names {sorted(unexpected)}.")
            print("    The parser treats unknown groups as TUMOR. Check that is")
            print("    right, or exclusions will be labelled as cancer (BUG-008).")
        excl = {g for g in groups if g.lower() in
                {e.lower() for e in ann.EXCLUSION_GROUPS}}
        tum = {g for g in groups if g.lower() in
               {t.lower() for t in ann.TUMOR_GROUPS}}
        print(f"    recognised as TUMOR    : {sorted(tum) or 'none'}")
        print(f"    recognised as EXCLUSION: {sorted(excl) or 'none'}")
        if not excl:
            print("    NOTE: no exclusion groups in this subset -- nothing to "
                  "subtract, so BUG-008 cannot bite here.")
        areas = sorted(r["tumor_mm2"] for r in with_xml)
        print(f"    tumor area mm^2: min {areas[0]:.4f}  "
              f"median {areas[len(areas)//2]:.4f}  max {areas[-1]:.4f}")
        zero = [r for r in with_xml if r["tumor_mm2"] <= 0]
        if zero:
            print(f"    WARNING: {len(zero)} slides have XML but zero tumor area: "
                  f"{[r['slide'] for r in zero][:5]}")
        inv = sum(r["invalid_polys"] for r in with_xml)
        print(f"    self-intersecting polygons repaired: {inv}")
        les = sorted(r["lesions"] for r in with_xml)
        print(f"    lesions per slide: min {les[0]}  "
              f"median {les[len(les)//2]}  max {les[-1]}")

    # ---- splits ----
    print(f"\n  SPLITS (patient_from={a.patient_from})")
    # Classify by FILENAME. Classifying by annotation area silently reports
    # every slide as "normal" when the XML files are missing, which hides the
    # actual problem behind a wrong-looking split table.
    grid = collections.Counter()
    for r in ok:
        grid[(r["name_class"], r["split"])] += 1
    kinds = [k for k in ("tumor", "normal", "test", "other")
             if any(grid[(k, s)] for s in ("train", "val", "test"))]
    print(f"    {'':9s} {'train':>7s} {'val':>7s} {'test':>7s}")
    for kind in kinds:
        print(f"    {kind:9s} " + "".join(
            f"{grid[(kind, s)]:7d}" for s in ("train", "val", "test")))

    n_tumor_named = sum(1 for r in ok if r["name_class"] == "tumor")
    n_tumor_xml = sum(1 for r in ok
                      if r["name_class"] == "tumor" and r["has_xml"])
    if n_tumor_named and n_tumor_xml < n_tumor_named:
        print(f"\n    CRITICAL: {n_tumor_named} slides are named tumor_* but "
              f"only {n_tumor_xml} have annotations.")
        print("    Without XML they carry NO positive labels and are "
              "indistinguishable from normals.")

    problems = [f"{k} slides missing from {s}"
                for k in ("tumor", "normal") for s in ("train", "val", "test")
                if grid[(k, s)] == 0]
    if problems:
        print("\n    WARNING: " + "; ".join(problems))
        print("    Gate V4.6 will block stage 02. Check splits.patient_from,")
        print("    or fetch more slides.")
    else:
        print("\n    all six cells populated -- gate V4.6 should pass")

    if a.csv:
        import pandas as pd
        pd.DataFrame(rows).to_csv(a.csv, index=False)
        print(f"\nwrote {a.csv}")

    print("\nVERDICT:", "PROCEED to stage 01" if not bad and not problems
          else "REVIEW the warnings above before stage 01")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
