"""BUG-003: downsample factors are never exact powers of two."""
import numpy as np
import pytest

from src.utils.geometry import (block_downsample, clamp_window, l0_to_level,
                                mm2_to_px, resolve_level_for_mpp, um_to_px)

DS = [1.0, 1.9992175, 3.9984350]


def test_non_power_of_two_downsample():
    naive = 105_000 / 2 ** 1
    actual = l0_to_level((105_000, 0), DS, 1)[0]
    assert abs(actual - naive) > 20, "fixture must exercise the drift"
    assert actual == pytest.approx(105_000 / DS[1])


def test_resolve_level_by_mpp_not_index():
    # Scanner A: level 0 is 40x -> 20x is level 1.
    assert resolve_level_for_mpp(DS, 0.25, 0.50)[0] == 1
    # Scanner B: level 0 is already ~20x -> 20x is level 0.
    lvl, resid = resolve_level_for_mpp([1.0, 2.0], 0.50, 0.50)
    assert lvl == 0 and resid == pytest.approx(1.0)


def test_missing_mpp_raises_rather_than_guessing():
    for bad in (None, 0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            resolve_level_for_mpp(DS, bad, 0.5)


def test_physical_units_scale_with_resolution():
    assert mm2_to_px(0.02, 2.0) == 5_000
    assert mm2_to_px(0.02, 0.5) == 80_000
    assert um_to_px(10, 2.0) == 5 and um_to_px(10, 0.5) == 20


def test_clamp_keeps_window_inside():
    dims = [(5003, 7001), (2502, 3501)]
    x, y = clamp_window(99_999, 99_999, 1, 512, dims, DS)
    assert 0 <= x <= (2502 - 512) * DS[1] and 0 <= y <= (3501 - 512) * DS[1]


def test_block_downsample_is_mean_not_subsample():
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = block_downsample(a, 2)
    assert out.shape == (2, 2)
    assert out[0, 0] == pytest.approx(np.mean([0, 1, 4, 5]))
    with pytest.raises(ValueError):
        block_downsample(np.zeros((5, 5), np.float32), 2)
