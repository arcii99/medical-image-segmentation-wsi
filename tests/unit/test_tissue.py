"""BUG-004 / BUG-005: guards against degenerate Otsu and pen ink."""
import numpy as np
import pytest

from src.preprocess.tissue import TissueDetectionError, tissue_mask


def _glass(shape=(400, 400), seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(243, 3, (*shape, 3)), 0, 255).astype(np.uint8)


def test_all_glass_raises():
    with pytest.raises(TissueDetectionError):
        tissue_mask(_glass(), mpp=8.0)


def test_black_border_does_not_become_tissue():
    """The BUG-001 failure mode, seen from the tissue detector's side."""
    img = _glass()
    img[:, -120:] = 0
    res = tissue_mask(img, mpp=8.0, strict=False)
    assert res.tissue_frac < 0.10, "near-black region must be rejected as artifact"


def test_ink_is_rejected():
    img = _glass()
    img[50:150, 50:150] = (20, 200, 60)        # saturated green marker
    res = tissue_mask(img, mpp=8.0, strict=False)
    ink_region = res.mask[50:150, 50:150]
    assert ink_region.mean() < 0.02


def test_real_tissue_is_detected():
    img = _glass()
    img[100:300, 100:300] = (196, 138, 186)    # H&E-ish
    res = tissue_mask(img, mpp=8.0, strict=False)
    assert 0.15 < res.tissue_frac < 0.45


# --- contract A2 lower bound is PHYSICAL AREA, not a fraction (BUG-023) ---

def _thumb(frac, h=3416, w=1504):
    """A real CAMELYON16 thumbnail footprint: 23.4 x 53.1 mm."""
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(243, 3, (h, w, 3)), 0, 255).astype(np.uint8)
    side = int((frac * h * w) ** 0.5)
    img[:side, :side] = (196, 138, 186)
    return img


MPP_L6 = 15.5579


def test_sparse_but_real_lymph_node_is_accepted():
    """The lowest tissue fraction in the real cohort was 0.0114 -- about
    14 mm^2, a genuine node. A fractional floor of 0.01 rejected slides
    like this one."""
    res = tissue_mask(_thumb(0.0114), MPP_L6, strict=True)
    assert res.tissue_mm2 > 5.0
    assert res.tissue_frac < 0.02


def test_genuinely_blank_slide_is_still_rejected():
    with pytest.raises(TissueDetectionError, match="mm\\^2"):
        tissue_mask(_thumb(0.0008), MPP_L6, strict=True)


def test_area_is_reported_in_physical_units():
    res = tissue_mask(_thumb(0.10), MPP_L6, strict=False)
    expected = res.mask.sum() * MPP_L6 ** 2 / 1e6
    assert abs(res.tissue_mm2 - expected) < 1e-6


def test_guard_scales_with_resolution_not_pixels():
    """Same physical tissue, two thumbnail resolutions -> same mm^2."""
    a = tissue_mask(_thumb(0.10, 3416, 1504), MPP_L6, strict=False)
    b = tissue_mask(_thumb(0.10, 1708, 752), MPP_L6 * 2, strict=False)
    assert abs(a.tissue_mm2 - b.tissue_mm2) / a.tissue_mm2 < 0.10
