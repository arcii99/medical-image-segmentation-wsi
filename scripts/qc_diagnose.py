#!/usr/bin/env python
"""Deep diagnostic for a cohort that qc_probe flagged.

qc_probe answers "is this cohort usable?". When the answer is no, this script
answers "why?" by dumping ground truth rather than interpretation:

  1. What the S3 bucket actually contains (from the cached listing), broken
     down by path prefix and file extension.
  2. What is actually on local disk, by filename prefix and extension.
  3. For slides that fail to open: the FULL backend error, plus the raw
     tifffile structure (series, pages, shapes, compression, key tags).
  4. For slides whose microns-per-pixel looks wrong: every resolution-related
     property and the full level geometry.

Nothing here is inferred. Paste the output rather than summarising it.

    python scripts/qc_diagnose.py --slides-dir data/raw/camelyon16
"""
from __future__ import annotations

import argparse
import collections
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXTS = (".tif", ".tiff", ".svs", ".ndpi")
# masks/ and background_tissue/ rasters share the slide naming scheme but are
# single-channel label images, not WSIs.
NOT_WSI = ("_mask", "_tissue")


def is_wsi(p) -> bool:
    return (p.suffix.lower() in EXTS
            and not p.stem.lower().endswith(NOT_WSI))
RULE = "=" * 78


def head(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


# --------------------------------------------------------------------------
def section_bucket(listing: Path) -> None:
    head("1. WHAT THE BUCKET ACTUALLY CONTAINS")
    if not listing.exists():
        print(f"  no cached listing at {listing}")
        print("  regenerate with:")
        print("    aws s3 ls s3://camelyon-dataset/CAMELYON16/ --recursive \\")
        print("        --no-sign-request --region us-west-2 > data/.s3_listing.txt")
        return

    rows = []
    for line in listing.read_text().splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) == 4 and parts[2].isdigit():
            rows.append((parts[3], int(parts[2])))
    print(f"  {len(rows)} objects in the cached listing\n")

    print("  BY DIRECTORY PREFIX")
    by_dir: dict[str, list[int]] = collections.defaultdict(list)
    for key, size in rows:
        by_dir["/".join(key.split("/")[:-1]) or "<root>"].append(size)
    for d in sorted(by_dir):
        sizes = by_dir[d]
        print(f"    {d:52s} {len(sizes):5d} files  "
              f"{sum(sizes)/1024**3:8.1f} GB")

    print("\n  BY EXTENSION")
    by_ext: dict[str, list[int]] = collections.defaultdict(list)
    for key, size in rows:
        ext = "." + key.rsplit(".", 1)[-1].lower() if "." in key else "<none>"
        by_ext[ext].append(size)
    for e in sorted(by_ext, key=lambda x: -len(by_ext[x])):
        sizes = by_ext[e]
        print(f"    {e:12s} {len(sizes):5d} files  {sum(sizes)/1024**3:8.2f} GB")

    print("\n  FIRST 25 KEYS")
    for key, size in rows[:25]:
        print(f"    {size/1024**2:10.2f} MB  {key}")

    print("\n  ANY KEY CONTAINING 'anno', 'xml', 'lesion' or 'zip'")
    hits = [(k, s) for k, s in rows
            if any(t in k.lower() for t in ("anno", "xml", "lesion", "zip"))]
    if not hits:
        print("    NONE FOUND -- this is why no annotations were downloaded")
    for key, size in hits[:30]:
        print(f"    {size/1024**2:10.2f} MB  {key}")
    if len(hits) > 30:
        print(f"    ... and {len(hits)-30} more")


# --------------------------------------------------------------------------
def section_local(slides_dir: Path) -> list[Path]:
    head("2. WHAT IS ACTUALLY ON LOCAL DISK")
    if not slides_dir.exists():
        print(f"  {slides_dir} does not exist"); return []

    files = sorted(p for p in slides_dir.rglob("*") if p.is_file())
    print(f"  {len(files)} files under {slides_dir}\n")

    print("  BY EXTENSION")
    by_ext = collections.Counter(p.suffix.lower() for p in files)
    for e, n in by_ext.most_common():
        tot = sum(p.stat().st_size for p in files if p.suffix.lower() == e)
        print(f"    {e or '<none>':12s} {n:5d} files  {tot/1024**3:8.2f} GB")

    print("\n  BY FILENAME PREFIX (text before the first digit)")
    def prefix(p: Path) -> str:
        s = p.stem
        for i, c in enumerate(s):
            if c.isdigit():
                return s[:i].rstrip("_-") or "<numeric>"
        return s
    by_pre = collections.Counter(prefix(p) for p in files if is_wsi(p))
    for pre, n in by_pre.most_common():
        print(f"    {pre:20s} {n:5d} slides")
    xmls = [p for p in files if p.suffix.lower() == ".xml"]
    print(f"\n  XML files present: {len(xmls)}")
    for p in xmls[:5]:
        print(f"    {p.name}")

    print("\n  SLIDE/ANNOTATION PAIRING (what stage 02 looks for)")
    slides = [p for p in files if is_wsi(p)]
    n_raster = sum(1 for p in files if p.suffix.lower() in EXTS and not is_wsi(p))
    print(f"    non-WSI rasters ignored      : {n_raster}")
    paired = [p for p in slides if p.with_suffix(".xml").exists()]
    tumor_named = [p for p in slides if p.stem.lower().startswith("tumor")]
    print(f"    slides                       : {len(slides)}")
    print(f"    slides named tumor_*         : {len(tumor_named)}")
    print(f"    slides with a matching .xml  : {len(paired)}")
    unpaired = [p for p in tumor_named if not p.with_suffix(".xml").exists()]
    if unpaired:
        print(f"    tumor_* slides MISSING xml   : {len(unpaired)}  "
              f"e.g. {[p.stem for p in unpaired[:4]]}")
    return slides


# --------------------------------------------------------------------------
def section_failures(slides: list[Path], n: int) -> None:
    head(f"3. FULL ERROR FOR UP TO {n} FAILING SLIDES")
    from src.io.slide import open_slide

    shown = 0
    for p in slides:
        if shown >= n:
            break
        try:
            with open_slide(p):
                continue
        except Exception:
            shown += 1
            print(f"\n  --- {p.name}  ({p.stat().st_size/1024**3:.2f} GB) ---")
            print("  " + traceback.format_exc().replace("\n", "\n  "))
            _raw_tiff(p)
    if shown == 0:
        print("  no failures -- every slide opened")


def _raw_tiff(p: Path) -> None:
    """What tifffile itself sees, bypassing tiffslide entirely."""
    try:
        import tifffile
    except ImportError:
        print("  (tifffile unavailable)"); return
    try:
        with tifffile.TiffFile(str(p)) as tf:
            print(f"  tifffile: {len(tf.series)} series, {len(tf.pages)} pages, "
                  f"bigtiff={tf.is_bigtiff}")
            for i, s in enumerate(tf.series[:4]):
                print(f"    series[{i}] shape={s.shape} dtype={s.dtype} "
                      f"axes={s.axes} levels={len(s.levels)}")
            for i, pg in enumerate(tf.pages[:6]):
                try:
                    print(f"    page[{i}] shape={pg.shape} "
                          f"compression={pg.compression} "
                          f"photometric={pg.photometric} "
                          f"tiled={pg.is_tiled} "
                          f"planar={getattr(pg, 'planarconfig', None)}")
                except Exception as e:  # noqa: BLE001
                    print(f"    page[{i}] unreadable: {e}")
            pg = tf.pages[0]
            for tag in ("ImageDescription", "Software", "Make", "Model",
                        "XResolution", "YResolution", "ResolutionUnit"):
                if tag in pg.tags:
                    v = str(pg.tags[tag].value)
                    print(f"    tag {tag:18s} = {v[:120]}")
    except Exception as e:  # noqa: BLE001
        print(f"  tifffile ALSO failed: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------
def section_mpp(slides: list[Path], n: int) -> None:
    head(f"4. MPP AND GEOMETRY FOR UP TO {n} SLIDES IN EACH MPP CLUSTER")
    from src.io.slide import open_slide

    ok = []
    for p in slides:
        try:
            with open_slide(p) as r:
                ok.append((p, round(r.meta.mpp, 4)))
        except Exception:
            continue
    if not ok:
        print("  nothing opened"); return

    clusters: dict[float, list[Path]] = collections.defaultdict(list)
    for p, mpp in ok:
        clusters[mpp].append(p)

    print(f"  MPP clusters: "
          f"{ {k: len(v) for k, v in sorted(clusters.items())} }\n")

    for mpp in sorted(clusters):
        for p in clusters[mpp][:n]:
            print(f"\n  --- {p.name}   reported mpp={mpp} ---")
            with open_slide(p) as r:
                m = r.meta
                print(f"    backend={m.backend} vendor={m.vendor} "
                      f"levels={m.level_count}")
                for i, (dims, ds) in enumerate(
                        zip(m.level_dims, m.level_downsamples)):
                    print(f"      L{i}: {dims[0]:7d} x {dims[1]:7d}   "
                          f"downsample {ds:9.5f}   "
                          f"mpp {m.mpp * ds:8.4f}")
                w, h = m.level_dims[0]
                print(f"    physical extent at reported mpp: "
                      f"{w*m.mpp/1000:.1f} x {h*m.mpp/1000:.1f} mm")
                print("      (a glass slide is about 25 x 75 mm; a tissue "
                      "section is rarely over 25 x 50 mm)")
                keys = [k for k in m.properties
                        if any(t in k.lower() for t in
                               ("mpp", "resolution", "micron", "magnif",
                                "vendor", "objective", "spacing"))]
                print(f"    resolution-related properties ({len(keys)}):")
                for k in sorted(keys)[:18]:
                    print(f"      {k:42s} = {str(m.properties[k])[:60]}")
            _raw_tiff(p)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(prog="qc_diagnose")
    ap.add_argument("--slides-dir", required=True)
    ap.add_argument("--listing", default="data/.s3_listing.txt")
    ap.add_argument("--n-failures", type=int, default=3)
    ap.add_argument("--n-per-cluster", type=int, default=1)
    a = ap.parse_args()

    section_bucket(Path(a.listing))
    slides = section_local(Path(a.slides_dir))
    if slides:
        section_failures(slides, a.n_failures)
        section_mpp(slides, a.n_per_cluster)
    head("END -- paste this output in full")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
