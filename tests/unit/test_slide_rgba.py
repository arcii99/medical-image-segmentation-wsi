"""BUG-001: zero-alpha regions composite to white, not black."""
import pytest
import numpy as np

from src.io.slide import _rgba_to_rgb_on_white


def test_zero_alpha_becomes_white():
    rgba = np.zeros((32, 32, 4), np.uint8)
    rgba[..., :3] = 17
    rgba[..., 3] = 255
    rgba[:16, :16, 3] = 0                     # unscanned quadrant
    out = _rgba_to_rgb_on_white(rgba)
    assert out.shape == (32, 32, 3) and out.dtype == np.uint8
    assert (out[:16, :16] == 255).all(), "alpha=0 must read as glass, not tissue"
    assert (out[16:, 16:] == 17).all()


def test_rgb_passthrough():
    rgb = np.full((8, 8, 3), 42, np.uint8)
    assert (_rgba_to_rgb_on_white(rgb) == 42).all()


# --- Philips DP exports (all of CAMELYON16) carry no TIFF resolution tag ---

PHILIPS_DESC = """<?xml version="1.0" encoding="UTF-8" ?>
<DataObject ObjectType="DPUfsImport">
  <Attribute Name="DICOM_MANUFACTURER" PMSVR="IString">PHILIPS</Attribute>
  <Attribute Name="PIM_DP_SCANNED_IMAGES" PMSVR="IDataObjectArray">
    <Array>
      <DataObject ObjectType="DPScannedImage">
        <Attribute Name="PIM_DP_IMAGE_TYPE" PMSVR="IString">MACROIMAGE</Attribute>
        <Attribute Name="DICOM_PIXEL_SPACING" PMSVR="IDoubleArray">"0.0221" "0.0221"</Attribute>
      </DataObject>
      <DataObject ObjectType="DPScannedImage">
        <Attribute Name="PIM_DP_IMAGE_TYPE" PMSVR="IString">WSI</Attribute>
        <Attribute Name="DICOM_PIXEL_SPACING" PMSVR="IDoubleArray">"0.000243" "0.000243"</Attribute>
      </DataObject>
    </Array>
  </Attribute>
</DataObject>"""


def test_philips_wsi_spacing_is_preferred_over_macro_image():
    from src.io.slide import _philips_mpp
    mx, my = _philips_mpp(PHILIPS_DESC)
    assert mx == pytest.approx(0.243, abs=1e-4)
    assert my == pytest.approx(0.243, abs=1e-4)


def test_philips_description_routed_through_resolve_mpp():
    from src.io.slide import _resolve_mpp
    mx, _ = _resolve_mpp({"tiff.ImageDescription": PHILIPS_DESC})
    assert mx == pytest.approx(0.243, abs=1e-4)


def test_implausible_spacing_is_rejected_not_used():
    from src.io.slide import _philips_mpp
    assert _philips_mpp(PHILIPS_DESC.replace("0.000243", "5.0")) is None


def test_no_mpp_anywhere_still_raises():
    from src.io.slide import SlideReadError, _resolve_mpp
    with pytest.raises(SlideReadError, match="no microns-per-pixel"):
        _resolve_mpp({"tiff.ImageDescription": "<unrelated/>"})
