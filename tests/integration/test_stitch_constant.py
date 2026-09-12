"""Gates V8.3-V8.5: numerical correctness of gigapixel reconstruction."""
import numpy as np
import pytest

from src.infer.sliding_window import PlannerConfig, plan_positions
from src.infer.stitch import CoverageError, GaussianStitcher, StitchConfig

DIMS = [(5003, 7001), (2502, 3501)]          # deliberately not stride-aligned
DS = [1.0, 5003 / 2502]


def _run(const=0.37):
    coords, summary = plan_positions(DIMS, DS, 1, np.ones((351, 251), bool),
                                     PlannerConfig())
    st = GaussianStitcher((7001, 5003), DS[1], StitchConfig(chunk=256))
    for i in range(0, len(coords), 16):
        c = coords[i:i + 16]
        st.add(np.full((len(c), 512, 512), const, np.float32), c)
    return st, coords, summary


def test_constant_field_reconstructs_exactly():
    st, _, _ = _run(0.37)
    heat = st.finalize(dtype=np.float32)
    den = np.asarray(st.den[:, :])
    inner = heat[den >= 0.05]
    assert np.abs(inner - 0.37).max() < 1e-3


def test_no_uncovered_tissue_on_non_aligned_dims():
    st, _, _ = _run()
    frac, _ = st.coverage_report(np.ones((351, 251), bool))
    assert frac == 0.0


def test_accumulators_must_be_float32():
    with pytest.raises(TypeError, match="float32"):
        GaussianStitcher((1024, 1024), 2.0, StitchConfig(),
                         store_factory=lambda s, d, c, n: np.zeros(s, np.float16))


def test_hole_is_detected():
    st = GaussianStitcher((2048, 2048), 2.0, StitchConfig())
    st.add(np.full((1, 512, 512), 0.5, np.float32), [(0, 0)])
    with pytest.raises(CoverageError):
        st.finalize(tissue=np.ones((256, 256), bool))


def test_seams_are_suppressed():
    st, _, _ = _run(0.37)
    heat = st.finalize(dtype=np.float32)
    den = np.asarray(st.den[:, :])
    cov = den >= 0.05
    g = np.abs(np.diff(heat, axis=1))[cov[:, 1:]]
    assert g.max() < 1e-4, "constant field must reconstruct without grid seams"
