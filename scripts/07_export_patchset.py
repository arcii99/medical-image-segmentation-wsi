#!/usr/bin/env python
"""Export a portable, self-contained patch set for training on a remote GPU.

Why this exists. The pipeline normally reads pixels lazily from the slides on
every epoch (ADR-005), which makes the 230 GB archive a runtime dependency of
training. That is the right design on a machine that holds the archive, and
impossible on Colab or Kaggle.

This freezes a balanced subset into a few gigabytes of JPEG patches that can be
uploaded once and trained on repeatedly.

What is deliberately given up:
  * Per-epoch coordinate jitter. The sampler normally nudges each patch by up
    to +/-128 px every epoch, so the model never sees the identical crop twice.
    Here the crops are fixed, which is a genuine reduction in augmentation.
  * The ability to re-index at a different patch size or stride without
    re-exporting.
  * Hard-negative mining over the full normal pool -- only the exported
    normals are available.

What is kept: the class balance, the split assignment, the physical scale
resolution, and every label. Photometric augmentation still runs at train time.

    python scripts/07_export_patchset.py --plan
    python scripts/07_export_patchset.py --export --out artifacts/patchset
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import index as idx  # noqa: E402
from src.io import annotations as ann  # noqa: E402
from src.io.slide import open_slide  # noqa: E402
from src.utils.config import load  # noqa: E402
from src.utils.envcheck import assert_environment  # noqa: E402
from src.utils.geometry import needs_resample  # noqa: E402

MB = 1024 ** 2


# --------------------------------------------------------------------------
def select_all(patches: pd.DataFrame) -> pd.DataFrame:
    """Export the entire index -- no sampling at all.

    Worth the extra disk when there is disk to spare. A balanced subset forces
    the sampler to draw from a pool that was already sampled, and leaves hard
    negative mining with nothing to mine. The full index restores both: the
    sampler behaves exactly as it does against the slide archive, and mining
    sees every normal patch.

    The only thing still given up is per-epoch coordinate jitter, and
    --jitter-copies recovers part of that.
    """
    sel = patches[patches.split.isin(("train", "val"))].copy()
    sel["key"] = [f"{r.slide_id}_{r.x0}_{r.y0}" for r in sel.itertuples()]
    sel["jitter"] = 0
    return sel.reset_index(drop=True)


def add_jitter_copies(sel: pd.DataFrame, n_copies: int, thr: float,
                      jitter_px: int, seed: int) -> pd.DataFrame:
    """Extra randomly-offset crops of the TUMOR patches only.

    Jitter matters most where data is scarcest. Tumor patches are ~6% of the
    index, so copying those is cheap, while copying negatives would multiply
    the export for little benefit -- there are already 237,000 of them.
    """
    if n_copies <= 0:
        sel["jx"] = 0; sel["jy"] = 0
        return sel
    rng = np.random.default_rng(seed)
    # Give the originals explicit zero offsets. Concatenating frames where
    # only some carry jx/jy fills the rest with NaN, which then fails on
    # int() at read time.
    sel = sel.copy(); sel["jx"] = 0; sel["jy"] = 0
    tumor = sel[(sel.tumor_frac > thr) & (sel.jitter == 0)]
    extra = []
    for k in range(1, n_copies + 1):
        c = tumor.copy()
        c["jx"] = rng.integers(-jitter_px, jitter_px + 1, len(c))
        c["jy"] = rng.integers(-jitter_px, jitter_px + 1, len(c))
        c["jitter"] = k
        c["key"] = [f"{r.slide_id}_{r.x0}_{r.y0}_j{k}" for r in c.itertuples()]
        extra.append(c)
    out = pd.concat([sel] + extra, ignore_index=True)
    out["jx"] = out["jx"].fillna(0).astype(int)
    out["jy"] = out["jy"].fillna(0).astype(int)
    return out


def select(patches: pd.DataFrame, slides: pd.DataFrame, cfg,
           normal_per_tumor: float, max_per_split: dict[str, int],
           seed: int) -> pd.DataFrame:
    """Pick which patches to export, preserving class balance per split."""
    thr = float(cfg.patching.tumor_threshold)
    rng = np.random.default_rng(seed)
    out = []
    for split in ("train", "val"):
        p = patches[patches.split == split]
        tumor = p[p.tumor_frac > thr]
        bound = p[(p.tumor_frac > 0) & (p.tumor_frac <= thr)]
        normal = p[p.tumor_frac == 0]

        n_norm = int(len(tumor) * normal_per_tumor)
        # Spread negatives across slides rather than taking whole slides:
        # a few slides' worth of negatives teaches the model those slides
        # rather than what normal tissue looks like in general.
        #
        # cumcount() instead of groupby().apply(): pandas 3.0 removed the
        # include_groups argument, and a rank filter is both faster and
        # version-independent.
        normal = normal.sample(frac=1.0, random_state=seed)
        n_slides = max(normal.slide_id.nunique(), 1)
        per_slide = max(1, n_norm // n_slides)
        normal = normal[normal.groupby("slide_id").cumcount() < per_slide]
        if len(normal) > n_norm:
            normal = normal.sample(n_norm, random_state=seed)

        sel = pd.concat([tumor, bound, normal])
        cap = max_per_split.get(split, 0)
        if cap and len(sel) > cap:
            # Cap by scaling both classes, never by truncating one.
            frac = cap / len(sel)
            sel = pd.concat([
                tumor.sample(max(1, int(len(tumor) * frac)), random_state=seed),
                bound.sample(max(0, int(len(bound) * frac)), random_state=seed),
                normal.sample(max(1, int(len(normal) * frac)), random_state=seed)])
        out.append(sel.assign(split=split))
    sel = pd.concat(out).reset_index(drop=True)
    sel["key"] = [f"{r.slide_id}_{r.x0}_{r.y0}" for r in sel.itertuples()]
    sel["jitter"] = 0
    return sel


def estimate_bytes(n: int, jpeg_kb: float = 78.0, mask_kb: float = 3.5) -> float:
    return n * (jpeg_kb + mask_kb) * 1024


# --------------------------------------------------------------------------
def _export_slide(args) -> tuple[str, list[dict], bytes]:
    """Encode every selected patch of one slide. Returns tar-ready members."""
    slide_id, path, xml, rows, size, level, residual, quality = args
    members, blob_parts = [], []
    with open_slide(path, slide_id=slide_id) as r:
        geom = ann.load(xml, slide_id, r.level_dims[0]) if xml else \
            ann.load(None, slide_id)
        ds = r.level_downsamples[level]
        for x0, y0, tumor_frac, key, jx, jy in rows:
            # Jittered copies are read at an offset, so they are genuinely
            # different crops rather than duplicates of the same pixels.
            x0 = max(0, x0 + int(jx * ds)); y0 = max(0, y0 + int(jy * ds))
            img = r.read_region(x0, y0, level, size, size)
            mask = geom.rasterize(x0, y0, size, size, ds)
            if needs_resample(residual):
                d = (int(round(size / residual)),) * 2
                img = cv2.resize(img, d, interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, d, interpolation=cv2.INTER_NEAREST)
                img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (size, size),
                                  interpolation=cv2.INTER_NEAREST)
            ok_i, enc_i = cv2.imencode(
                ".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            ok_m, enc_m = cv2.imencode(".png", (mask > 0).astype(np.uint8) * 255,
                                       [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
            if not (ok_i and ok_m):
                continue
            members.append({"key": key, "img": len(enc_i), "msk": len(enc_m),
                            "tumor_frac": float(tumor_frac)})
            blob_parts.append((f"{key}.jpg", enc_i.tobytes()))
            blob_parts.append((f"{key}.png", enc_m.tobytes()))
    return slide_id, members, blob_parts


def main() -> int:
    assert_environment()
    ap = argparse.ArgumentParser(prog="07_export_patchset")
    ap.add_argument("--config", nargs="+",
                    default=["configs/base.yaml", "configs/data_camelyon16.yaml"])
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--out", default="artifacts/patchset")
    ap.add_argument("--all", action="store_true",
                    help="export the ENTIRE index, no sampling (~20 GB for "
                         "254k patches). Restores the sampler and hard "
                         "negative mining; needs disk on both ends.")
    ap.add_argument("--jitter-copies", type=int, default=0,
                    help="extra randomly-offset crops per TUMOR patch, to "
                         "recover some of the per-epoch jitter an export loses")
    ap.add_argument("--normal-per-tumor", type=float, default=3.0)
    ap.add_argument("--max-train", type=int, default=0, help="0 = no cap")
    ap.add_argument("--max-val", type=int, default=8000)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()

    cfg = load(a.config, a.set)
    in_dir = Path(cfg.paths.index)
    patches = idx.load(in_dir / "patches.parquet")
    slides = pd.read_parquet(in_dir / "slides.parquet")

    thr = float(cfg.patching.tumor_threshold)
    if a.all:
        sel = select_all(patches)
        print("\nMODE: full index -- no sampling. The sampler and hard "
              "negative mining will behave exactly as they do against the "
              "slide archive.")
    else:
        sel = select(patches, slides, cfg, a.normal_per_tumor,
                     {"train": a.max_train, "val": a.max_val}, int(cfg.seed))
    if not a.jitter_copies:
        sel["jx"] = 0; sel["jy"] = 0
    if a.jitter_copies:
        before = len(sel)
        sel = add_jitter_copies(sel, a.jitter_copies, thr,
                                int(cfg.patching.jitter), int(cfg.seed))
        print(f"  + {len(sel)-before:,} jittered tumor copies "
              f"({a.jitter_copies} per tumor patch, +/-"
              f"{int(cfg.patching.jitter)} px)")
    print(f"\nSELECTED {len(sel):,} patches")
    for split in ("train", "val"):
        s = sel[sel.split == split]
        t = int((s.tumor_frac > thr).sum())
        print(f"  {split:6s} {len(s):7,}  tumor {t:6,} ({t/max(len(s),1):5.1%})"
              f"  slides {s.slide_id.nunique():3d}")
    est = estimate_bytes(len(sel))
    print(f"\n  estimated size ~{est/MB/1024:.2f} GB at JPEG q{a.quality}")
    print(f"  shards of {a.shard_size:,} -> "
          f"{int(np.ceil(len(sel)/a.shard_size))} tar files")
    if est / MB / 1024 > 60:
        print("\n  WARNING: this will not fit Colab's ~107 GB local disk once")
        print("  extracted alongside checkpoints. Reduce the export.")
    elif est / MB / 1024 > 12:
        print("\n  NOTE: over ~12 GB. Fine if Drive has room and the upload")
        print("  is a one-off; tight on free Drive (15 GB).")

    if not a.export:
        print("\n(plan only -- re-run with --export)")
        return 0

    out = Path(a.out); (out / "shards").mkdir(parents=True, exist_ok=True)
    meta = slides.set_index("slide_id")
    jobs = []
    for sid, g in sel.groupby("slide_id"):
        m = meta.loc[sid]
        xml = Path(m.path).with_suffix(".xml")
        jobs.append((sid, m.path, xml if xml.exists() else None,
                     [(int(r.x0), int(r.y0), float(r.tumor_frac), r.key,
                       int(r.jx), int(r.jy))
                      for r in g.itertuples()],
                     int(g["size"].iloc[0]), int(m.level_w),
                     float(m.residual_scale), a.quality))

    shard_i, in_shard, tar = 0, 0, None
    written, total_bytes = [], 0

    def _open_shard(i):
        return tarfile.open(out / "shards" / f"shard_{i:04d}.tar", "w")

    tar = _open_shard(shard_i)
    runner = (ProcessPoolExecutor(a.workers).map if a.workers
              else map)
    for n_done, (sid, members, blobs) in enumerate(
            runner(_export_slide, jobs), 1):
        for name, payload in blobs:
            info = tarfile.TarInfo(name); info.size = len(payload)
            import io
            tar.addfile(info, io.BytesIO(payload))
            total_bytes += len(payload)
        for m in members:
            m["slide_id"] = sid; m["shard"] = shard_i
            written.append(m)
        in_shard += len(members)
        if in_shard >= a.shard_size:
            tar.close(); shard_i += 1; in_shard = 0; tar = _open_shard(shard_i)
        print(f"  [{n_done:3d}/{len(jobs)}] {sid:18s} "
              f"{len(members):5d} patches  {total_bytes/MB/1024:6.2f} GB",
              flush=True)
    tar.close()

    man = pd.DataFrame(written).merge(
        sel[["key", "split", "slide_id", "x0", "y0"]], on=["key", "slide_id"],
        how="left")
    man.to_parquet(out / "manifest.parquet", index=False)
    (out / "export_info.json").write_text(json.dumps({
        "n_patches": len(man), "patch_size": int(sel["size"].iloc[0]),
        "jpeg_quality": a.quality, "normal_per_tumor": a.normal_per_tumor,
        "tumor_threshold": thr, "index_hash": idx.index_hash(
            in_dir / "patches.parquet"), "seed": int(cfg.seed),
        "bytes": total_bytes,
    }, indent=2))
    print(f"\nwrote {out}  ({total_bytes/MB/1024:.2f} GB, "
          f"{shard_i+1} shards, {len(man):,} patches)")
    print("\nUpload the whole directory. See docs/COLAB.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
