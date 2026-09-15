#!/usr/bin/env python
"""Measure patch-read latency (gate V1.5).

Also the way to settle "can I just train off a Google Drive mount?" with a
measurement instead of an argument. Point it at a slide on the mount and
compare against local disk.

    python scripts/bench_read.py --slide data/raw/camelyon16/tumor_016.tif
    python scripts/bench_read.py --slide /content/drive/MyDrive/wsi/tumor_016.tif
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.slide import open_slide  # noqa: E402

GPU_PATCHES_PER_SEC = 36.0     # UNet-EffB0, batch 16, bf16, T4 (Architecture 5)


def main() -> int:
    ap = argparse.ArgumentParser(prog="bench_read")
    ap.add_argument("--slide", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8,
                    help="assumed parallel readers, for the throughput estimate")
    a = ap.parse_args()

    with open_slide(a.slide) as r:
        level = a.level if a.level is not None else r.level_for_mpp(0.5)[0]
        lw, lh = r.level_dims[level]
        ds = r.level_downsamples[level]
        rng = np.random.default_rng(0)
        xs = rng.integers(0, max(lw - a.size, 1), a.n)
        ys = rng.integers(0, max(lh - a.size, 1), a.n)

        print(f"slide {Path(a.slide).name}  level {level}  "
              f"{lw}x{lh}  mpp {r.meta.mpp_at(level):.3f}")
        print(f"reading {a.n} random {a.size}px windows ...", flush=True)

        # Warm up: the first read pays for index parsing and, on a network
        # mount, connection setup. Including it would flatter or distort.
        r.read_region(int(xs[0] * ds), int(ys[0] * ds), level, a.size, a.size)

        t = []
        t0 = time.perf_counter()
        for x, y in zip(xs, ys):
            s = time.perf_counter()
            r.read_region(int(x * ds), int(y * ds), level, a.size, a.size)
            t.append((time.perf_counter() - s) * 1000)
        wall = time.perf_counter() - t0

    t.sort()
    p = lambda q: t[min(int(len(t) * q), len(t) - 1)]  # noqa: E731
    print(f"\n  p50 {p(0.50):8.1f} ms")
    print(f"  p90 {p(0.90):8.1f} ms")
    print(f"  p99 {p(0.99):8.1f} ms")
    print(f"  mean {statistics.mean(t):7.1f} ms   total {wall:.1f} s")

    single = 1000.0 / statistics.mean(t)
    est = single * a.workers
    print(f"\n  1 reader  ~{single:8.1f} patches/s")
    print(f"  {a.workers} readers ~{est:8.1f} patches/s  (assumes linear scaling)")
    print(f"  GPU appetite ~{GPU_PATCHES_PER_SEC:.0f} patches/s")

    if est >= 3 * GPU_PATCHES_PER_SEC:
        print("\n  VERDICT: comfortably GPU-bound. Reading is not the problem.")
    elif est >= GPU_PATCHES_PER_SEC:
        print("\n  VERDICT: marginal. Reading roughly keeps up; any hiccup "
              "stalls the GPU.")
    else:
        print(f"\n  VERDICT: IO-BOUND. The GPU would idle {1 - est/GPU_PATCHES_PER_SEC:.0%} "
              "of the time.")
        print("  Training from this location is not worth the GPU hours.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
