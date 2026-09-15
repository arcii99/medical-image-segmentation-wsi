"""Post-processing must filter by the same measure evaluation scores by."""
import numpy as np
import pytest

from src.eval import froc as F
from src.infer.postproc import PostProcConfig, apply, threshold_from_ckpt

MPP = 2.0   # heatmap microns per pixel


def _heat(shapes, size=500):
    h = np.zeros((size, size), np.float32)
    for (y0, y1, x0, x1) in shapes:
        h[y0:y1, x0:x1] = 0.9
    return h


def test_elongated_evaluable_lesion_survives():
    """400 x 40 um: axis 400 um (evaluable) but only 0.016 mm^2 of area."""
    _, les = apply(_heat([(50, 250, 50, 70)]), 0.5, MPP, PostProcConfig())
    assert len(les) == 1
    assert les[0].major_axis_um > F.ITC_MAX_AXIS_UM
    assert les[0].area_mm2 < 0.02, "fixture must exercise the area/axis gap"


def test_area_only_filter_would_have_deleted_it():
    cfg = PostProcConfig(min_lesion_axis_um=0.0, min_lesion_mm2=0.02)
    _, les = apply(_heat([(50, 250, 50, 70)]), 0.5, MPP, cfg)
    assert les == [], "documents the behaviour this change removes"


def test_speckle_is_still_removed():
    _, les = apply(_heat([(300, 305, 300, 305)]), 0.5, MPP, PostProcConfig())
    assert les == []


def test_filter_threshold_is_below_the_evaluability_boundary():
    """No lesion FROC would score may ever be filtered out."""
    assert PostProcConfig().min_lesion_axis_um < F.ITC_MAX_AXIS_UM


def test_axis_matches_the_evaluation_measure():
    """A predicted square and a GT square of equal size must measure equal."""
    from shapely.geometry import box
    _, les = apply(_heat([(100, 200, 100, 200)]), 0.5, MPP, PostProcConfig())
    gt = F.major_axis_um(box(0, 0, 100, 100), MPP)
    assert les[0].major_axis_um == pytest.approx(gt, rel=0.05)


def test_threshold_must_come_from_the_checkpoint():
    with pytest.raises(KeyError, match="no fitted threshold"):
        threshold_from_ckpt({})
    assert threshold_from_ckpt({"threshold": 0.42}) == 0.42
