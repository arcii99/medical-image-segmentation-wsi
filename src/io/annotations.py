"""Annotation parsing and windowed rasterisation.

Tumor geometry is kept as **vector polygons in level-0 coordinates** for the
whole pipeline and rasterised only inside a 512x512 window at the moment a
patch is read.  Rasterising a full slide mask would cost ~4.7 GB per slide;
rasterising a window costs 256 KB (Architecture contract A3).

Two things this module exists to get right:

* CAMELYON XML uses annotation **groups**: ``_0``/``_1`` are tumor, ``_2``
  marks non-tumor islands *inside* them.  Unioning all polygons labels fat and
  stroma as tumor (BUG-008).
* Patch scoring uses exact polygon **area**, not rasterisation -- 40x faster
  over three million windows and exact to within 0.5% (gate V3.4).
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from shapely import STRtree
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

log = logging.getLogger(__name__)

__all__ = ["TumorGeometry", "load", "AnnotationError"]

TUMOR_GROUPS = ("_0", "_1", "tumor", "metastases")
EXCLUSION_GROUPS = ("_2", "exclusion", "normal")


class AnnotationError(ValueError):
    pass


# --------------------------------------------------------------------------
@dataclass
class TumorGeometry:
    """Tumor polygons in level-0 pixel coordinates, exclusions already removed."""

    slide_id: str
    geom: MultiPolygon
    n_groups: dict[str, int]
    n_invalid_before_repair: int

    def __post_init__(self) -> None:
        parts = list(getattr(self.geom, "geoms", [self.geom])) if not self.geom.is_empty else []
        self._parts: list[Polygon] = [p for p in parts if not p.is_empty]
        self._tree = STRtree(self._parts) if self._parts else None

    # -- summary -------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return not self._parts

    @property
    def n_lesions(self) -> int:
        return len(self._parts)

    def area_px(self) -> float:
        return float(self.geom.area)

    def area_mm2(self, mpp_level0: float) -> float:
        return self.area_px() * (mpp_level0 ** 2) / 1e6

    def bounds(self) -> tuple[float, float, float, float] | None:
        return tuple(self.geom.bounds) if self._parts else None  # type: ignore[return-value]

    def lesions(self) -> list[Polygon]:
        """Individual lesions, for lesion-level FROC matching (ADR-013)."""
        return list(self._parts)

    # -- queries -------------------------------------------------------
    def _clip(self, window: Polygon) -> list[Polygon]:
        if self._tree is None:
            return []
        out = []
        for idx in self._tree.query(window):
            inter = self._parts[int(idx)].intersection(window)
            if not inter.is_empty:
                out.append(inter)
        return out

    def area_fraction(self, x0: int, y0: int, w_l0: float, h_l0: float) -> float:
        """Fraction of a level-0 window covered by tumor, by exact polygon area.

        This is what labels the patch index; it never touches pixels.
        """
        if self._tree is None:
            return 0.0
        win = box(x0, y0, x0 + w_l0, y0 + h_l0)
        covered = sum(p.area for p in self._clip(win))
        return float(min(1.0, covered / max(win.area, 1e-9)))

    def rasterize(
        self,
        x0: int,
        y0: int,
        w: int,
        h: int,
        downsample: float,
    ) -> np.ndarray:
        """Rasterise the tumor mask for one window.

        Parameters
        ----------
        x0, y0 : int
            Window origin in level-0 pixels.
        w, h : int
            Output size in pixels at the target level.
        downsample : float
            The target level's downsample factor, taken from the slide
            (INV-2 -- never ``2 ** level``).

        Returns
        -------
        np.ndarray
            ``(h, w)`` uint8 in ``{0, 1}``.
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        if self._tree is None:
            return mask

        win = box(x0, y0, x0 + w * downsample, y0 + h * downsample)
        clipped = self._clip(win)
        if not clipped:
            return mask

        for poly in clipped:
            for part in getattr(poly, "geoms", [poly]):
                if not isinstance(part, Polygon) or part.is_empty:
                    continue
                cv2.fillPoly(mask, [_ring_to_px(part.exterior, x0, y0, downsample)], 1)
                for ring in part.interiors:  # holes from exclusion subtraction
                    cv2.fillPoly(mask, [_ring_to_px(ring, x0, y0, downsample)], 0)
        return mask


def _ring_to_px(ring, x0: int, y0: int, downsample: float) -> np.ndarray:
    """Level-0 polygon ring -> integer pixel coordinates local to the window."""
    xy = np.asarray(ring.coords, dtype=np.float64)
    xy[:, 0] = (xy[:, 0] - x0) / downsample
    xy[:, 1] = (xy[:, 1] - y0) / downsample
    return np.round(xy).astype(np.int32)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def load(
    xml_path: str | Path | None,
    slide_id: str,
    level0_dims: tuple[int, int] | None = None,
) -> TumorGeometry:
    """Parse a CAMELYON-style annotation file.

    A missing path yields an empty geometry -- that is the correct
    representation of a normal slide, not an error.
    """
    empty = MultiPolygon([])
    if xml_path is None:
        return TumorGeometry(slide_id, empty, {}, 0)
    xml_path = Path(xml_path)
    if not xml_path.exists():
        return TumorGeometry(slide_id, empty, {}, 0)

    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as e:
        raise AnnotationError(f"{slide_id}: malformed XML: {e}") from e

    tumor_polys: list[Polygon] = []
    excl_polys: list[Polygon] = []
    counts: dict[str, int] = {}
    n_invalid = 0

    for ann in root.iter("Annotation"):
        group = (ann.get("PartOfGroup") or ann.get("Name") or "").strip()
        coords = _coords_of(ann)
        if len(coords) < 3:
            continue
        poly = Polygon(coords)
        if not poly.is_valid:
            n_invalid += 1
            poly = poly.buffer(0)  # repairs self-intersections; common in CAMELYON
        if poly.is_empty:
            continue
        counts[group or "<none>"] = counts.get(group or "<none>", 0) + 1

        key = group.lower()
        if key in EXCLUSION_GROUPS:
            excl_polys.append(poly)
        elif key in TUMOR_GROUPS or not group:
            tumor_polys.append(poly)
        else:
            log.warning("%s: unrecognised annotation group %r, treating as tumor",
                        slide_id, group)
            tumor_polys.append(poly)

    geom = _combine(tumor_polys, excl_polys)

    if level0_dims is not None and not geom.is_empty:
        w0, h0 = level0_dims
        minx, miny, maxx, maxy = geom.bounds
        if minx < -1 or miny < -1 or maxx > w0 + 1 or maxy > h0 + 1:
            raise AnnotationError(
                f"{slide_id}: annotation bbox ({minx:.0f},{miny:.0f},"
                f"{maxx:.0f},{maxy:.0f}) outside level-0 dims {w0}x{h0}; "
                "annotations are probably not in level-0 coordinates"
            )

    return TumorGeometry(slide_id, geom, counts, n_invalid)


def _coords_of(ann: ET.Element) -> list[tuple[float, float]]:
    pts: list[tuple[int, float, float]] = []
    for c in ann.iter("Coordinate"):
        try:
            pts.append((int(c.get("Order", len(pts))),
                        float(c.get("X")), float(c.get("Y"))))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    pts.sort(key=lambda t: t[0])
    return [(x, y) for _, x, y in pts]


def _combine(tumor: Sequence[Polygon], exclusion: Sequence[Polygon]) -> MultiPolygon:
    """Union the tumor groups, then subtract the exclusion group (BUG-008)."""
    if not tumor:
        return MultiPolygon([])
    geom = unary_union(list(tumor))
    if exclusion:
        geom = geom.difference(unary_union(list(exclusion)))
    geom = geom.buffer(0)
    if geom.is_empty:
        return MultiPolygon([])
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    return MultiPolygon([g for g in geom.geoms if isinstance(g, Polygon)])


# --------------------------------------------------------------------------
# CLI (gate V3.1)
# --------------------------------------------------------------------------
def _cli() -> None:
    import argparse

    from src.io.slide import open_slide

    p = argparse.ArgumentParser(prog="src.io.annotations")
    p.add_argument("--info", required=True, help="path to annotation XML")
    p.add_argument("--slide", required=True, help="matching slide file")
    args = p.parse_args()

    with open_slide(args.slide) as r:
        dims = r.level_dims[0]
        mpp = r.meta.mpp

    pre = load(args.info, Path(args.slide).stem, None)
    counts = ", ".join(f"{k}={v}" for k, v in sorted(pre.n_groups.items())) or "none"
    print(f"groups: {counts}")
    print(f"polygons parsed: {sum(pre.n_groups.values())}    "
          f"invalid before buffer(0): {pre.n_invalid_before_repair}    "
          f"invalid after: {0 if pre.geom.is_valid else 1}")
    print(f"tumor area (post-exclusion): {pre.area_mm2(mpp):.4f} mm^2")
    print(f"lesions: {pre.n_lesions}")
    b = pre.bounds()
    inside = b is None or (b[0] >= -1 and b[1] >= -1 and b[2] <= dims[0] + 1
                           and b[3] <= dims[1] + 1)
    print(f"bbox within level-0 dims: {inside}")


if __name__ == "__main__":
    _cli()
