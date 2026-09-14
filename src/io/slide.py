"""Slide reading facade (ADR-001).

Every pixel in this project enters through :class:`SlideReader`.  The facade
owns three normalisations so that no caller has to remember them:

1. RGBA -> RGB composited onto **white** (BUG-001).  Regions outside the
   scanned area have ``A = 0`` and undefined RGB; slicing off alpha turns
   "nothing here" into pure black, which every tissue detector reads as dense
   tissue.
2. MPP resolution from vendor properties with a TIFF-tag fallback, raising
   rather than guessing.
3. The ``read_region(location_level0, level, size_at_level)`` contract, which
   is asserted rather than assumed.

The reader is deliberately **not picklable with a live handle**: libtiff and
openslide handles carry mutable seek state and are not fork-safe.  See
BUG-002 and :func:`src.data.dataset.worker_init`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from src.utils.geometry import (
    Window,
    assert_level0_coords,
    resolve_level_for_mpp,
)

log = logging.getLogger(__name__)

Backend = Literal["tiffslide", "openslide", "cucim"]
BACKEND_CHAIN: tuple[Backend, ...] = ("tiffslide", "openslide", "cucim")


class SlideReadError(RuntimeError):
    """Raised when a slide cannot be opened or a region cannot be read."""


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SlideMeta:
    slide_id: str
    path: Path
    backend: Backend
    level_count: int
    level_dims: tuple[tuple[int, int], ...]      # (w, h) per level
    level_downsamples: tuple[float, ...]
    mpp_x: float
    mpp_y: float
    vendor: str = "unknown"
    properties: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def mpp(self) -> float:
        """Isotropic MPP. Anisotropy beyond 2% is a scanner fault, not a feature."""
        if abs(self.mpp_x - self.mpp_y) / max(self.mpp_x, self.mpp_y) > 0.02:
            raise SlideReadError(
                f"{self.slide_id}: anisotropic pixels "
                f"({self.mpp_x:.4f} x {self.mpp_y:.4f} um/px)"
            )
        return 0.5 * (self.mpp_x + self.mpp_y)

    def mpp_at(self, level: int) -> float:
        return self.mpp * float(self.level_downsamples[level])


# --------------------------------------------------------------------------
# Backend availability
# --------------------------------------------------------------------------
def available_backends() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in BACKEND_CHAIN:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[name] = None
    return out


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------
class SlideReader:
    """Random-access reader for a pyramidal whole-slide image."""

    def __init__(self, path: str | Path, backend: Backend | None = None,
                 slide_id: str | None = None):
        self.path = Path(path)
        self._slide_id = slide_id or self.path.stem
        self._requested_backend = backend
        self._handle: Any | None = None
        self._meta: SlideMeta | None = None
        self._open()

    # -- lifecycle ------------------------------------------------------
    def _open(self) -> None:
        chain = (self._requested_backend,) if self._requested_backend else BACKEND_CHAIN
        errors: list[str] = []
        for name in chain:
            try:
                handle, meta = _OPENERS[name](self.path, self._slide_id)
            except ImportError as e:
                errors.append(f"{name}: not installed ({e})")
                continue
            except Exception as e:  # noqa: BLE001 - backend-specific failures vary
                errors.append(f"{name}: {type(e).__name__}: {e}")
                continue
            self._handle, self._meta = handle, meta
            log.debug("opened %s with %s", self.path.name, name)
            return
        raise SlideReadError(
            f"could not open {self.path} with any backend:\n  " + "\n  ".join(errors)
        )

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:  # noqa: BLE001
                pass
            self._handle = None

    def __enter__(self) -> "SlideReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # BUG-002: never carry a handle across a process boundary.
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open()

    # -- metadata -------------------------------------------------------
    @property
    def meta(self) -> SlideMeta:
        assert self._meta is not None
        return self._meta

    @property
    def slide_id(self) -> str:
        return self.meta.slide_id

    @property
    def level_dims(self) -> tuple[tuple[int, int], ...]:
        return self.meta.level_dims

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        return self.meta.level_downsamples

    def level_for_mpp(self, target_mpp: float, tolerance: float = 0.05
                      ) -> tuple[int, float]:
        """(level, residual_scale) for a target microns-per-pixel. See ADR-002."""
        return resolve_level_for_mpp(
            self.meta.level_downsamples, self.meta.mpp, target_mpp, tolerance
        )

    # -- reading --------------------------------------------------------
    def read_region(self, x0: int, y0: int, level: int, w: int, h: int,
                    retries: int = 1) -> np.ndarray:
        """Read a window.

        Parameters
        ----------
        x0, y0 : int
            Top-left corner in **level-0** pixels (INV-1).
        level : int
            Pyramid level to read from.
        w, h : int
            Size in pixels **at ``level``**.

        Returns
        -------
        np.ndarray
            ``(h, w, 3)`` uint8, RGB, alpha composited onto white,
            C-contiguous.
        """
        assert_level0_coords(x0, y0)
        if not 0 <= level < self.meta.level_count:
            raise ValueError(f"level {level} out of range 0..{self.meta.level_count-1}")

        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                raw = self._handle.read_region((int(x0), int(y0)), int(level),
                                               (int(w), int(h)))
                arr = np.asarray(raw)
                return _rgba_to_rgb_on_white(arr)
            except Exception as e:  # noqa: BLE001
                last = e
                if attempt < retries:
                    log.warning("read_region retry on %s at (%d,%d): %s",
                                self.slide_id, x0, y0, e)
                    self.close()
                    self._open()
        raise SlideReadError(
            f"{self.slide_id}: read_region({x0},{y0},L{level},{w}x{h}) failed"
        ) from last

    def read_window(self, win: Window) -> np.ndarray:
        return self.read_region(win.x0, win.y0, win.level, win.w, win.h)

    def read_level(self, level: int) -> np.ndarray:
        """Read an entire pyramid level. Only safe for thumbnail levels."""
        w, h = self.meta.level_dims[level]
        if w * h > 80_000_000:
            raise SlideReadError(
                f"refusing to materialise level {level} ({w}x{h} = "
                f"{w*h/1e6:.0f} Mpx); use a coarser level"
            )
        return self.read_region(0, 0, level, w, h)

    def thumbnail_level(self, target_mpp: float = 8.0) -> int:
        level, _ = self.level_for_mpp(target_mpp, tolerance=float("inf"))
        return level

    def __repr__(self) -> str:
        m = self.meta
        return (f"SlideReader({m.slide_id!r}, backend={m.backend}, "
                f"levels={m.level_count}, mpp={m.mpp:.4f})")


# --------------------------------------------------------------------------
# Normalisation helpers
# --------------------------------------------------------------------------
def _rgba_to_rgb_on_white(arr: np.ndarray) -> np.ndarray:
    """Composite RGBA onto white and return a contiguous uint8 RGB array.

    BUG-001: ``arr[..., :3]`` leaves zero-alpha pixels black, which the
    inverted-grayscale tissue branch reads as maximally dense tissue.  Glass is
    white, so white is the correct background.
    """
    if arr.ndim != 3:
        raise ValueError(f"expected HxWxC, got shape {arr.shape}")
    if arr.shape[2] == 3:
        return np.ascontiguousarray(arr.astype(np.uint8, copy=False))
    if arr.shape[2] != 4:
        raise ValueError(f"expected 3 or 4 channels, got {arr.shape[2]}")

    rgba = arr.astype(np.float32, copy=False)
    a = rgba[..., 3:4] / 255.0
    rgb = rgba[..., :3] * a + 255.0 * (1.0 - a)
    return np.ascontiguousarray(np.clip(rgb, 0, 255).round().astype(np.uint8))


def _philips_mpp(description: str) -> tuple[float, float] | None:
    """Extract microns-per-pixel from a Philips DP ImageDescription block.

    CAMELYON16's images are Philips scanner exports. They carry no TIFF
    XResolution tag at all -- the pixel spacing lives inside an XML document
    stashed in ImageDescription:

        <DataObject ObjectType="DPUfsImport">
          <Attribute Name="PIM_DP_SCANNED_IMAGES" ...>
            <Array>
              <DataObject ObjectType="DPScannedImage">
                <Attribute Name="PIM_DP_IMAGE_TYPE" ...>WSI</Attribute>
                <Attribute Name="DICOM_PIXEL_SPACING" ...>"0.00025" "0.00025"</Attribute>

    The file contains several scanned images -- the WSI itself plus a macro
    photograph and a label photograph -- each with its own spacing, so the
    WSI entry must be selected rather than taking the first match. Spacing is
    in millimetres; we return microns.
    """
    import xml.etree.ElementTree as _ET  # noqa: PLC0415

    try:
        root = _ET.fromstring(description)
    except _ET.ParseError:
        return None

    def _spacings(node) -> list[float]:
        vals: list[float] = []
        for attr in node.iter("Attribute"):
            if "PIXEL_SPACING" not in (attr.get("Name") or "").upper():
                continue
            for tok in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                                  attr.text or ""):
                v = float(tok)
                if v > 0:
                    vals.append(v)
        return vals

    wsi: list[float] = []
    for obj in root.iter("DataObject"):
        kind = ""
        for attr in obj.findall("Attribute"):
            if (attr.get("Name") or "").upper().endswith("IMAGE_TYPE"):
                kind = (attr.text or "").strip().upper()
        if kind == "WSI":
            wsi.extend(_spacings(obj))

    # Fall back to the finest spacing anywhere in the document: the macro and
    # label photographs are far coarser than the WSI, so the minimum is the
    # WSI's own spacing.
    vals = wsi or _spacings(root)
    if not vals:
        return None
    mm = min(vals)
    if not 1e-5 < mm < 1e-2:            # 0.01 - 10 um/px, sanity band
        log.warning("Philips pixel spacing %.6g mm is outside the plausible "
                    "range; ignoring", mm)
        return None
    um = mm * 1000.0
    return um, um


def _resolve_mpp(props: dict[str, Any]) -> tuple[float, float]:
    """Extract microns-per-pixel, or raise.

    Order: vendor-neutral openslide keys -> Aperio ``MPP`` -> TIFF resolution
    tags.  Never falls back to a default; ADR-002 treats a missing MPP as a
    hard failure because a wrong physical scale is silently destructive.
    """
    def _f(v: Any) -> float | None:
        try:
            f = float(v)
            return f if f > 0 else None
        except (TypeError, ValueError):
            return None

    mx = _f(props.get("openslide.mpp-x"))
    my = _f(props.get("openslide.mpp-y"))
    if mx and my:
        return mx, my

    aperio = _f(props.get("aperio.MPP"))
    if aperio:
        return aperio, aperio

    # tiffslide derives these from the TIFF resolution tags when present.
    tsx, tsy = _f(props.get("tiffslide.mpp-x")), _f(props.get("tiffslide.mpp-y"))
    if tsx and tsy:
        return tsx, tsy

    # Philips DP exports (all of CAMELYON16) have no resolution tag; the
    # spacing is inside the ImageDescription XML.
    desc = props.get("tiff.ImageDescription") or props.get("tiffslide.comment")
    if desc and "DPUfsImport" in str(desc):
        got = _philips_mpp(str(desc))
        if got:
            return got

    # TIFF resolution tags: pixels per RESOLUTIONUNIT.
    unit = str(props.get("tiff.ResolutionUnit", "")).lower()
    xres, yres = _f(props.get("tiff.XResolution")), _f(props.get("tiff.YResolution"))
    if xres and yres:
        if unit in ("centimeter", "cm", "3"):
            return 10_000.0 / xres, 10_000.0 / yres
        if unit in ("inch", "2"):
            return 25_400.0 / xres, 25_400.0 / yres

    raise SlideReadError(
        "no microns-per-pixel metadata found (checked openslide.mpp-*, "
        "aperio.MPP, tiffslide.mpp-*, Philips ImageDescription, "
        f"tiff.*Resolution); available keys: {sorted(props)[:12]}"
    )


def _normalise_downsamples(
    level_dims: Sequence[tuple[int, int]], reported: Sequence[float] | None
) -> tuple[float, ...]:
    """Prefer the file's reported factors; otherwise derive from dimensions.

    Either way the result is float and generally *not* a power of two (INV-2).
    """
    if reported is not None and len(reported) == len(level_dims):
        return tuple(float(d) for d in reported)
    w0, h0 = level_dims[0]
    return tuple(0.5 * (w0 / w + h0 / h) for w, h in level_dims)


# --------------------------------------------------------------------------
# Backend openers
# --------------------------------------------------------------------------
def _open_tiffslide(path: Path, slide_id: str) -> tuple[Any, SlideMeta]:
    import tiffslide  # noqa: PLC0415

    h = tiffslide.TiffSlide(str(path))
    props = dict(h.properties)
    dims = tuple((int(w), int(h_)) for w, h_ in h.level_dimensions)
    mx, my = _resolve_mpp(props)
    meta = SlideMeta(
        slide_id=slide_id, path=path, backend="tiffslide",
        level_count=h.level_count, level_dims=dims,
        level_downsamples=_normalise_downsamples(dims, h.level_downsamples),
        mpp_x=mx, mpp_y=my,
        vendor=str(props.get("tiffslide.vendor")
                   or props.get("openslide.vendor") or "generic-tiff"),
        properties=props,
    )
    return h, meta


def _open_openslide(path: Path, slide_id: str) -> tuple[Any, SlideMeta]:
    import openslide  # noqa: PLC0415

    h = openslide.OpenSlide(str(path))
    props = dict(h.properties)
    dims = tuple((int(w), int(h_)) for w, h_ in h.level_dimensions)
    mx, my = _resolve_mpp(props)
    meta = SlideMeta(
        slide_id=slide_id, path=path, backend="openslide",
        level_count=h.level_count, level_dims=dims,
        level_downsamples=_normalise_downsamples(dims, h.level_downsamples),
        mpp_x=mx, mpp_y=my,
        vendor=str(props.get("openslide.vendor", "unknown")),
        properties=props,
    )
    return h, meta


def _open_cucim(path: Path, slide_id: str) -> tuple[Any, SlideMeta]:
    from cucim import CuImage  # noqa: PLC0415

    img = CuImage(str(path))
    res = img.resolutions
    dims = tuple((int(w), int(h_)) for w, h_ in res["level_dimensions"])
    props = dict(img.metadata or {})
    mx, my = _resolve_mpp(props)

    class _Shim:
        """Adapt CuImage to the read_region signature used above."""

        def __init__(self, im: Any):
            self._im = im

        def read_region(self, location: tuple[int, int], level: int,
                        size: tuple[int, int]) -> np.ndarray:
            return np.asarray(
                self._im.read_region(location=location, level=level, size=size)
            )

        def close(self) -> None:
            self._im.close()

    meta = SlideMeta(
        slide_id=slide_id, path=path, backend="cucim",
        level_count=int(res["level_count"]), level_dims=dims,
        level_downsamples=_normalise_downsamples(dims, res.get("level_downsamples")),
        mpp_x=mx, mpp_y=my, vendor="cucim", properties=props,
    )
    return _Shim(img), meta


_OPENERS = {
    "tiffslide": _open_tiffslide,
    "openslide": _open_openslide,
    "cucim": _open_cucim,
}


def open_slide(path: str | Path, backend: Backend | None = None,
               slide_id: str | None = None) -> SlideReader:
    return SlideReader(path, backend=backend, slide_id=slide_id)


# --------------------------------------------------------------------------
# CLI (verification gates V0.2, V1.1)
# --------------------------------------------------------------------------
def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(prog="src.io.slide")
    p.add_argument("--selftest", action="store_true", help="gate V0.2")
    p.add_argument("--info", type=str, default=None, help="gate V1.1")
    p.add_argument("--target-mpp", type=float, default=0.5)
    args = p.parse_args()

    if args.selftest:
        avail = available_backends()
        for name in BACKEND_CHAIN:
            v = avail[name]
            print(f"{name:10s}: {'available  (' + v + ')' if v else 'unavailable'}")
        chain = [n for n in BACKEND_CHAIN if avail[n]]
        if not chain:
            raise SystemExit("FAIL: no slide backend available")
        print("backend chain:", " -> ".join(chain))
        return

    if args.info:
        with open_slide(args.info) as r:
            m = r.meta
            lvl, resid = r.level_for_mpp(args.target_mpp)
            print(f"{'slide_id':18s}{m.slide_id}")
            print(f"{'backend':18s}{m.backend}")
            print(f"{'vendor':18s}{m.vendor}")
            print(f"{'level_count':18s}{m.level_count}")
            print(f"{'level_dims':18s}{list(m.level_dims)}")
            print(f"{'level_downsamples':18s}"
                  f"{[round(d, 5) for d in m.level_downsamples]}")
            print(f"{'mpp':18s}{m.mpp_x:.4f} x {m.mpp_y:.4f}")
            print(f"{'working level':18s}L{lvl} @ {m.mpp_at(lvl):.4f} um/px "
                  f"(residual {resid:.4f})")
        return

    p.print_help()


if __name__ == "__main__":
    _cli()
