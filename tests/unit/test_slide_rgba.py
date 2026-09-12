"""BUG-001: zero-alpha regions composite to white, not black."""
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
