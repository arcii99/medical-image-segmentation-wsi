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
