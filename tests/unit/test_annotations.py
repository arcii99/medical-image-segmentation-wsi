"""BUG-008: group _2 marks non-tumor islands and must be subtracted."""
import numpy as np
import pytest

from src.io import annotations as ann


def test_group_2_is_subtracted(fixtures):
    g = ann.load(fixtures / "synth_tumor.xml", "synth_tumor")
    assert g.n_groups.get("_0") == 1 and g.n_groups.get("_2") == 1
    assert g.area_mm2(0.25) == pytest.approx(2.0, abs=2e-3)
    assert g.geom.is_valid


def test_exclusion_is_a_hole_in_the_raster(fixtures):
    g = ann.load(fixtures / "synth_tumor.xml", "synth_tumor")
    cx = 8192 // 2
    m = g.rasterize(cx - 256, cx - 256, 512, 512, downsample=1.0)
    assert m.sum() == 0, "slide centre is inside the exclusion polygon"
    off = g.rasterize(cx + 1400, cx - 256, 512, 512, downsample=1.0)
    assert off.sum() > 0, "offset window should be inside the tumor ring"


def test_area_fraction_matches_raster(fixtures):
    """Gate V3.4: the vector shortcut must agree with rasterisation."""
    g = ann.load(fixtures / "synth_tumor.xml", "synth_tumor")
    x0 = y0 = 8192 // 2 + 1200
    vec = g.area_fraction(x0, y0, 512, 512)
    ras = g.rasterize(x0, y0, 512, 512, downsample=1.0).mean()
    assert abs(vec - ras) < 0.005


def test_missing_annotation_is_an_empty_geometry_not_an_error():
    g = ann.load(None, "synth_normal")
    assert g.is_empty and g.area_px() == 0.0
    assert g.rasterize(0, 0, 64, 64, 1.0).sum() == 0
