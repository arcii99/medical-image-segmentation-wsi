#!/usr/bin/env python
"""Stage 00 -- plan and fetch a CAMELYON16 subset that fits your disk.

The full CAMELYON16 release is roughly 700 GB. Most machines cannot hold it,
and most of it is not needed to train a working model: normal slides are
plentiful and interchangeable, whereas tumor slides are the scarce resource.

This script lists the public S3 bucket, works out what will fit in a stated
budget, and downloads a subset chosen so that the resulting cohort is still
usable:

* Tumor and normal slides are taken TOGETHER at a target ratio, not tumor
  first. Filling the budget with tumor slides and no normals produces a
  cohort with no clean negatives: slide-level AUC becomes undefined and the
  false-positives-per-normal-slide metric cannot be computed at all. Both
  classes degrade together as the budget shrinks.
* Within each class the order is a deterministic seeded hash, so the same
  budget always yields the same cohort and the per-patient split assignment
  stays balanced.
* Annotation XML files are tiny and always taken in full.
* Headroom is reserved for artifacts (index, checkpoints, heatmaps).

Requires the AWS CLI on PATH. No AWS account or credentials are needed --
the bucket is public and every call uses --no-sign-request.

    python scripts/00_fetch_data.py --plan --budget-gb 280
    python scripts/00_fetch_data.py --download --budget-gb 280
    python scripts/00_fetch_data.py --manifest
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

BUCKET = "s3://camelyon-dataset"
PREFIX = "CAMELYON16"

# The bucket holds four parallel trees. Only two of them are wanted:
#
#   CAMELYON16/images/             399 files, 700.8 GB  <- the WSIs
#   CAMELYON16/annotations/        160 files,   0.1 GB  <- lesion polygons
#   CAMELYON16/masks/              399 files,   8.8 GB  <- rasterised labels
#   CAMELYON16/background_tissue/  400 files,   0.4 GB  <- tissue masks, 8x down
#
# masks/ and background_tissue/ are also .tif and also named normal_*/tumor_*,
# so a filename-only filter happily downloads them and then stage 01 tries to
# segment a label mask. They are single-channel, not RGB, and the pipeline
# computes both of these itself, so they are excluded outright.
IMAGES_PREFIX = "CAMELYON16/images/"
ANNOTATIONS_PREFIX = "CAMELYON16/annotations/"
EXCLUDE_SUFFIXES = ("_mask", "_tissue")
REGION = "us-west-2"
GB = 1024 ** 3


@dataclass(frozen=True)
class Obj:
    key: str
    size: int

    @property
    def name(self) -> str:
        return self.key.rsplit("/", 1)[-1]

    @property
    def stem(self) -> str:
        return self.name.rsplit(".", 1)[0]

    @property
    def kind(self) -> str:
        n = self.name.lower()
        if n.startswith("tumor"):
            return "tumor"
        if n.startswith("normal"):
            return "normal"
        if n.startswith("test"):
            return "test"
        return "other"

    @property
    def is_slide(self) -> bool:
        if not self.key.startswith(IMAGES_PREFIX):
            return False
        if not self.name.lower().endswith((".tif", ".tiff", ".svs")):
            return False
        return not self.stem.lower().endswith(EXCLUDE_SUFFIXES)

    @property
    def is_annotation(self) -> bool:
        return (self.key.startswith(ANNOTATIONS_PREFIX)
                and self.name.lower().endswith(".xml"))

    @property
    def is_excluded_raster(self) -> bool:
        """A mask or background-tissue raster: same naming, wrong content."""
        return (self.name.lower().endswith((".tif", ".tiff"))
                and (self.stem.lower().endswith(EXCLUDE_SUFFIXES)
                     or not self.key.startswith(IMAGES_PREFIX)))


# --------------------------------------------------------------------------
def require_aws() -> str:
    exe = shutil.which("aws")
    if exe:
        return exe
    sys.exit(
        "The AWS CLI is not on PATH.\n\n"
        "Install it with ONE of:\n"
        "  curl 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' "
        "-o /tmp/awscliv2.zip \\\n"
        "    && unzip -q /tmp/awscliv2.zip -d /tmp && sudo /tmp/aws/install\n"
        "  sudo snap install aws-cli --classic\n"
        "  pip install awscli        (inside your conda env)\n\n"
        "Note: 'sudo apt install awscli' no longer works on Ubuntu 24.04."
    )


def list_bucket(cache: Path | None = None) -> list[Obj]:
    """`aws s3 ls --recursive`, cached so planning does not re-list."""
    if cache and cache.exists():
        raw = cache.read_text()
    else:
        aws = require_aws()
        print(f"listing {BUCKET}/{PREFIX}/ ...", flush=True)
        raw = subprocess.run(
            [aws, "s3", "ls", f"{BUCKET}/{PREFIX}/", "--recursive",
             "--no-sign-request", "--region", REGION],
            check=True, capture_output=True, text=True).stdout
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(raw)
    return parse_listing(raw)


def parse_listing(raw: str) -> list[Obj]:
    out: list[Obj] = []
    for line in raw.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) < 4:
            continue
        try:
            size = int(parts[2])
        except ValueError:
            continue
        out.append(Obj(parts[3], size))
    return out


# --------------------------------------------------------------------------
def _shuffled(objs: list[Obj], seed: int) -> list[Obj]:
    """Deterministic order, so the same budget always yields the same cohort."""
    return sorted(objs, key=lambda o: hashlib.md5(
        f"{seed}:{o.name}".encode()).hexdigest())


def plan(objs: list[Obj], budget_gb: float, reserve_gb: float = 40.0,
         seed: int = 1337, normal_per_tumor: float = 1.0
         ) -> tuple[list[Obj], dict]:
    """Choose a subset that keeps both classes represented.

    Selection interleaves tumor and normal slides to hold roughly
    ``normal_per_tumor`` normals per tumor. Taking all tumor slides first
    would exhaust a tight budget before a single negative slide is fetched.
    """
    slides = [o for o in objs if o.is_slide]
    anns = [o for o in objs if o.is_annotation]

    tumor = _shuffled([o for o in slides if o.kind == "tumor"], seed)
    normal = _shuffled([o for o in slides if o.kind == "normal"], seed)

    budget = int((budget_gb - reserve_gb) * GB)
    if budget <= 0:
        sys.exit(f"budget {budget_gb} GB does not exceed the {reserve_gb} GB reserve")

    chosen = list(anns)
    used = sum(o.size for o in anns)
    ti = ni = 0

    # Take whichever class is currently under-represented against the target.
    while ti < len(tumor) or ni < len(normal):
        want_normal = (ni < normal_per_tumor * max(ti, 1)) and ni < len(normal)
        pool, i = (normal, ni) if want_normal else (tumor, ti)
        if i >= len(pool):                      # that class exhausted
            pool, i = (tumor, ti) if want_normal else (normal, ni)
            if i >= len(pool):
                break
            want_normal = not want_normal
        o = pool[i]
        if used + o.size > budget:
            break
        chosen.append(o); used += o.size
        if want_normal:
            ni += 1
        else:
            ti += 1

    stats = {
        "tumor_available": len(tumor), "tumor_selected": ti,
        "tumor_skipped": len(tumor) - ti,
        "normal_available": len(normal), "normal_selected": ni,
        "normal_skipped": len(normal) - ni,
        "annotations": len(anns), "bytes": used, "budget_bytes": budget,
        "total_bucket_bytes": sum(o.size for o in slides),
    }
    return chosen, stats


def report(objs: list[Obj], chosen: list[Obj], stats: dict,
           budget_gb: float, reserve_gb: float) -> None:
    slides = [o for o in objs if o.is_slide]
    by_kind: dict[str, list[Obj]] = {}
    for o in slides:
        by_kind.setdefault(o.kind, []).append(o)

    excluded = [o for o in objs if o.is_excluded_raster]
    if excluded:
        by_pre: dict[str, list[Obj]] = {}
        for o in excluded:
            by_pre.setdefault("/".join(o.key.split("/")[:-1]), []).append(o)
        print("\nEXCLUDED (not WSIs -- label/tissue rasters the pipeline "
              "computes itself)")
        for d in sorted(by_pre):
            g = by_pre[d]
            print(f"  {d:38s} {len(g):5d} files  "
                  f"{sum(o.size for o in g)/GB:7.2f} GB  SKIPPED")

    print("\nWHAT IS IN THE BUCKET (images/ only)")
    print(f"  {'class':10s} {'slides':>7s} {'total':>10s} {'mean':>9s}")
    for k in sorted(by_kind):
        g = by_kind[k]
        tot = sum(o.size for o in g)
        print(f"  {k:10s} {len(g):7d} {tot/GB:9.1f}G {tot/len(g)/GB:8.2f}G")
    print(f"  {'TOTAL':10s} {len(slides):7d} "
          f"{stats['total_bucket_bytes']/GB:9.1f}G")

    print(f"\nPLAN  (budget {budget_gb:.0f} GB, reserving {reserve_gb:.0f} GB "
          f"for artifacts)")
    print(f"  tumor slides      {stats['tumor_selected']:4d} "
          f"of {stats['tumor_available']}")
    print(f"  normal slides     {stats['normal_selected']:4d} "
          f"of {stats['normal_available']}")
    print(f"  annotation files  {stats['annotations']:4d}")
    print(f"  download size     {stats['bytes']/GB:8.1f} GB "
          f"of {stats['budget_bytes']/GB:.1f} GB usable")

    ratio = stats["normal_selected"] / max(stats["tumor_selected"], 1)
    print(f"\n  normal:tumor ratio {ratio:.2f} "
          f"(full dataset {stats['normal_available']/max(stats['tumor_available'],1):.2f})")

    warn = []
    if stats["tumor_selected"] < 40:
        warn.append(f"only {stats['tumor_selected']} tumor slides. These carry "
                    "every positive label; below ~40 the val and test splits "
                    "get very few lesions and FROC becomes noisy.")
    if stats["normal_selected"] < 20:
        warn.append(f"only {stats['normal_selected']} normal slides. Slide-level "
                    "AUC and the false-positives-per-normal-slide metric both "
                    "need tumor-free slides in the test split.")
    if warn:
        print()
        for w in warn:
            print(f"  WARNING: {w}")
    est = stats["tumor_selected"] + stats["normal_selected"]
    print(f"\n  cohort: {est} slides. Split roughly 70/15/15 by patient ->"
          f" ~{int(est*0.15)} test slides.")


# --------------------------------------------------------------------------
def download(chosen: list[Obj], dest: Path) -> None:
    aws = require_aws()
    dest.mkdir(parents=True, exist_ok=True)
    total = sum(o.size for o in chosen)
    done = 0
    for i, o in enumerate(sorted(chosen, key=lambda o: o.key), 1):
        target = dest / o.name
        if target.exists() and target.stat().st_size == o.size:
            done += o.size
            continue
        print(f"[{i:4d}/{len(chosen)}] {o.name:34s} "
              f"{o.size/GB:6.2f}G   {done/total:5.1%} done", flush=True)
        subprocess.run(
            [aws, "s3", "cp", f"{BUCKET}/{o.key}", str(target),
             "--no-sign-request", "--region", REGION, "--only-show-errors"],
            check=True)
        done += o.size
    print(f"\ncomplete: {len(chosen)} objects, {total/GB:.1f} GB in {dest}")


def manifest(dest: Path, out: Path) -> None:
    """Checksum every downloaded slide. Slow, but a partially-corrupt TIFF
    often still opens and reads plausible pixels from its undamaged tiles."""
    lines = []
    files = sorted(p for p in dest.iterdir() if p.is_file())
    for i, p in enumerate(files, 1):
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {p.name}")
        print(f"[{i}/{len(files)}] {p.name}", flush=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} files)")


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(prog="00_fetch_data")
    ap.add_argument("--plan", action="store_true", help="show the plan, download nothing")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--fetch-reference-tissue", action="store_true",
                    help="download CAMELYON16/background_tissue rasters for the "
                         "slides already present locally (~1 MB each). These are "
                         "the challenge's own tissue masks and turn tissue-"
                         "detection QC from eyeballing into a measured IoU.")
    ap.add_argument("--reference-dest", default="data/reference_tissue")
    ap.add_argument("--budget-gb", type=float, default=280.0)
    ap.add_argument("--reserve-gb", type=float, default=40.0,
                    help="disk held back for index, checkpoints and heatmaps")
    ap.add_argument("--dest", default="data/raw/camelyon16")
    ap.add_argument("--listing-cache", default="data/.s3_listing.txt")
    ap.add_argument("--from-listing", default=None,
                    help="plan from a saved listing file instead of calling S3")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--normal-per-tumor", type=float, default=1.0,
                    help="target ratio of normal to tumor slides (default 1.0)")
    a = ap.parse_args()

    dest = Path(a.dest)

    if a.fetch_reference_tissue:
        have = {p.stem for p in dest.glob("*.tif")
                if not p.stem.endswith(("_mask", "_tissue"))}
        if not have:
            sys.exit(f"no slides found in {dest}")
        objs = (parse_listing(Path(a.from_listing).read_text())
                if a.from_listing else list_bucket(Path(a.listing_cache)))
        want = [o for o in objs
                if o.key.startswith("CAMELYON16/background_tissue/")
                and o.stem.rsplit("_", 1)[0] in have]
        print(f"{len(have)} local slides -> {len(want)} reference rasters, "
              f"{sum(o.size for o in want)/1024**2:.0f} MB")
        download(want, Path(a.reference_dest))
        return 0

    if a.manifest:
        manifest(dest, Path("data/raw_manifest.sha256"))
        return 0
    if not (a.plan or a.download):
        ap.print_help(); return 1

    if a.from_listing:
        objs = parse_listing(Path(a.from_listing).read_text())
    else:
        objs = list_bucket(Path(a.listing_cache))
    if not objs:
        sys.exit("bucket listing was empty")

    chosen, stats = plan(objs, a.budget_gb, a.reserve_gb, a.seed,
                     a.normal_per_tumor)
    report(objs, chosen, stats, a.budget_gb, a.reserve_gb)

    if a.download:
        print()
        download(chosen, dest)
        print("\nNext: python scripts/00_fetch_data.py --manifest")
    else:
        print("\n(plan only -- re-run with --download to fetch)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
