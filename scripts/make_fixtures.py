#!/usr/bin/env python
"""Generate deterministic synthetic pyramidal WSIs + annotations for tests.

These fixtures exist to make the verification gates runnable without the
400-slide CAMELYON16 archive.  Each is built to trip a specific historical
defect if it ever comes back:

* ``synth_tumor``  -- a known 2.00 mm^2 lesion with an exclusion island, so
  annotation area and post-processing recovery are checkable to 4 decimals
  (gates V3.1, V9.2).
* ``synth_normal`` -- tissue, no annotation.
* ``synth_edge``   -- dimensions **not** a multiple of the stride, an alpha=0
  border region, and pyramid levels sized ``ceil(w/2)`` so the downsample
  factors are 1.9996..., not 2.0 (BUG-001, BUG-003, BUG-012).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import tifffile

MPP0 = 0.25          # um/px at level 0
TILE = 256


def _tissue_field(h: int, w: int, seed: int, coverage: float = 0.35) -> np.ndarray:
    """Smooth blobby 'tissue' region on a white glass background."""
    rng = np.random.default_rng(seed)
    small = rng.random((max(h // 64, 4), max(w // 64, 4)))
    import cv2
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (0, 0), max(h, w) / 120.0)
    return big > np.quantile(big, 1.0 - coverage)


def _render(h: int, w: int, seed: int, coverage: float = 0.35):
    import cv2
    rng = np.random.default_rng(seed)
    tissue = _tissue_field(h, w, seed, coverage)
    img = np.full((h, w, 3), 243, np.uint8)                      # glass
    img += rng.integers(-3, 4, (h, w, 3), dtype=np.int16).astype(np.uint8) * 0
    # H&E-ish: purple nuclei speckle over pink stroma
    stroma = np.dstack([
        np.full((h, w), 196, np.uint8),
        np.full((h, w), 138, np.uint8),
        np.full((h, w), 186, np.uint8)]).astype(np.float32)
    noise = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.5)
    stroma -= (noise[..., None] * 60)
    img[tissue] = np.clip(stroma, 0, 255).astype(np.uint8)[tissue]
    return img, tissue


def _pyramid(base: np.ndarray, levels: int = 6, min_dim: int = 48) -> list[np.ndarray]:
    """Levels sized ceil(w/2), so downsample factors are NOT powers of two.

    Six levels rather than three: a real WSI has 6-9, which puts the ~8 um/px
    thumbnail level within reach.  With only three levels the thumbnail
    resolves at 1.0 um/px, and the physically-specified 40 um closing radius
    becomes a 40-pixel structuring element that merges every tissue blob --
    correct behaviour from the algorithm, wrong conditions from the fixture.
    """
    import cv2
    out = [base]
    for _ in range(levels - 1):
        h, w = out[-1].shape[:2]
        if min(h, w) // 2 < min_dim:
            break
        out.append(cv2.resize(out[-1], ((w + 1) // 2, (h + 1) // 2),
                              interpolation=cv2.INTER_AREA))
    return out


_COMPRESSION: str | None = None


def _pick_compression() -> str:
    """Choose a tile codec that actually works in this environment.

    JPEG is preferred because real SVS slides use it, so the fixtures exercise
    the same decode path as production data.  But JPEG encoding goes through
    imagecodecs, which ships compiled extensions linked against numpy's C API.
    A numpy/imagecodecs version mismatch -- common in conda environments where
    pip upgraded one but not the other -- raises

        ValueError: numpy.dtype size changed ... Expected 96 ..., got 88

    That is an environment problem, not a reason the test fixtures should be
    unbuildable.  Probe once and fall back to deflate, which is slower to read
    and produces larger files but needs no compiled codec.
    """
    global _COMPRESSION
    if _COMPRESSION is not None:
        return _COMPRESSION
    probe = np.zeros((TILE, TILE, 3), np.uint8)
    try:
        import io
        with tifffile.TiffWriter(io.BytesIO(), bigtiff=True) as tw:
            tw.write(probe, tile=(TILE, TILE), photometric="rgb",
                     compression="jpeg")
        _COMPRESSION = "jpeg"
    except Exception as e:
        print(f"  ! JPEG tile encoding unavailable ({type(e).__name__}: "
              f"{str(e)[:70]})")
        print("  ! falling back to deflate; fixtures will be larger but valid.")
        print("  ! to get JPEG back:  pip install -U imagecodecs")
        _COMPRESSION = "deflate"
    return _COMPRESSION


def _write(path: Path, pyr: list[np.ndarray]) -> None:
    res = (1e4 / MPP0, 1e4 / MPP0)     # px per cm
    comp = _pick_compression()
    with tifffile.TiffWriter(path, bigtiff=True) as tw:
        tw.write(pyr[0], tile=(TILE, TILE), photometric="rgb", compression=comp,
                 resolution=res, resolutionunit="CENTIMETER", subifds=len(pyr) - 1)
        for lvl in pyr[1:]:
            tw.write(lvl, tile=(TILE, TILE), photometric="rgb", compression=comp,
                     resolution=res, resolutionunit="CENTIMETER", subfiletype=1)


def _annotation_xml(path: Path, outer_px, holes_px) -> None:
    root = ET.Element("ASAP_Annotations")
    anns = ET.SubElement(root, "Annotations")
    for i, (poly, group) in enumerate(
            [(p, "_0") for p in outer_px] + [(p, "_2") for p in holes_px]):
        a = ET.SubElement(anns, "Annotation", Name=f"Annotation {i}",
                          Type="Polygon", PartOfGroup=group, Color="#F4FA58")
        cs = ET.SubElement(a, "Coordinates")
        for j, (x, y) in enumerate(poly):
            ET.SubElement(cs, "Coordinate", Order=str(j), X=f"{x:.1f}", Y=f"{y:.1f}")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _rect(cx, cy, w, h):
    return [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
            (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]


def build(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = out_dir / "reference_tissue"; ref.mkdir(exist_ok=True)
    import imageio.v3 as iio

    # ---- synth_tumor: exact 2.00 mm^2 lesion ------------------------
    # 2.00 mm^2 net = 2.40 mm^2 outer - 0.40 mm^2 exclusion, in level-0 px.
    # The slide must be large enough to hold it: at 0.25 um/px a 2.4 mm^2
    # square is 6197 px on a side, so 8192^2 (2.048 mm across, 4.19 mm^2) is
    # the minimum sensible canvas.  A smaller canvas trips the annotation
    # bbox guard in io.annotations.load, which is the correct behaviour.
    import cv2
    h = w = 8192
    img, tissue = _render(h, w, seed=1, coverage=0.30)
    px_per_mm = 1000.0 / MPP0                         # 4000 px per mm
    side_outer = np.sqrt(2.40) * px_per_mm            # mm^2 -> px side
    side_hole = np.sqrt(0.40) * px_per_mm
    cx = cy = w / 2.0
    outer = _rect(cx, cy, side_outer, side_outer)
    hole = _rect(cx, cy, side_hole, side_hole)

    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(np.array(outer)).astype(np.int32)], 1)
    cv2.fillPoly(m, [np.round(np.array(hole)).astype(np.int32)], 0)
    lesion = m.astype(bool)
    tissue = tissue | lesion                          # lesion must sit on tissue
    img[tissue & ~lesion] = img[tissue & ~lesion]
    base = np.clip(img.astype(np.int16) - np.array([40, 60, 30], np.int16), 0, 255)
    img[lesion] = base[lesion].astype(np.uint8)
    _write(out_dir / "synth_tumor.tif", _pyramid(img))
    _annotation_xml(out_dir / "synth_tumor.xml", [outer], [hole])
    iio.imwrite(ref / "synth_tumor.png",
                (cv2.resize(tissue.astype(np.uint8), (w // 8, h // 8),
                            interpolation=cv2.INTER_NEAREST) * 255))
    del base, m

    # ---- synth_normal -----------------------------------------------
    img, tissue = _render(4096, 4096, seed=2, coverage=0.28)
    _write(out_dir / "synth_normal.tif", _pyramid(img))
    iio.imwrite(ref / "synth_normal.png",
                (cv2.resize(tissue.astype(np.uint8), (512, 512),
                            interpolation=cv2.INTER_NEAREST) * 255))

    # ---- synth_edge: non-aligned dims + alpha-0 style black border ---
    h, w = 2001, 1503                                  # not multiples of 256
    img, tissue = _render(h, w, seed=3, coverage=0.30)
    img[:, -180:] = 0                                  # simulates alpha=0 region
    img[-140:, :] = 0
    _write(out_dir / "synth_edge.tif", _pyramid(img))
    iio.imwrite(ref / "synth_edge.png",
                (cv2.resize(tissue.astype(np.uint8), (w // 8, h // 8),
                            interpolation=cv2.INTER_NEAREST) * 255))

    print(f"wrote fixtures to {out_dir}")
    for p in sorted(out_dir.glob("*.tif")):
        print(f"  {p.name:20s} {p.stat().st_size/1e6:6.2f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/fixtures/mini")
    build(Path(ap.parse_args().out))
