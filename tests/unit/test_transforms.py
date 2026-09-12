"""BUG-011 / BUG-013: photometric transforms must never touch the mask."""
import numpy as np

from src.data.transforms import TrainTransform, photometric


def test_mask_binary_after_full_pipeline():
    rng = np.random.default_rng(0)
    img = rng.integers(120, 230, (64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), np.uint8); mask[10:30, 10:30] = 1
    tf = TrainTransform(0)
    for _ in range(500):
        _, m = tf(img, mask)
        assert set(np.unique(m)) <= {0.0, 1.0}


def test_photometric_signature_cannot_accept_a_mask():
    import inspect
    params = list(inspect.signature(photometric).parameters)
    assert "mask" not in params, "photometric must be image-only by construction"
