"""End-to-end on synthetic fixtures: read -> tissue -> index -> plan -> stitch."""
import numpy as np
import pytest

from src.infer.sliding_window import PlannerConfig, plan_positions
from src.infer.stitch import GaussianStitcher, StitchConfig
from src.io import annotations as ann
from src.io.slide import open_slide
from src.preprocess.patching import PatchConfig, iter_patches
from src.preprocess.tissue import tissue_mask


def test_metadata_and_coordinate_contract(fixtures):
    with open_slide(fixtures / "synth_edge.tif") as r:
        assert r.meta.level_count >= 4, "fixture needs a deep enough pyramid"
        # BUG-003 fixture property: factors are close to, but never exactly,
        # powers of two -- and the error grows with level.
        for lvl in range(1, r.meta.level_count):
            nominal = 2.0 ** lvl
            actual = r.level_downsamples[lvl]
            assert actual != nominal
            assert abs(actual - nominal) / nominal < 0.02
        assert r.meta.mpp == pytest.approx(0.25, abs=1e-3)

        # location is level-0, size is at `level`
        a = r.read_region(512, 512, 1, 128, 128)
        assert a.shape == (128, 128, 3) and a.dtype == np.uint8


def test_tissue_then_index_then_plan(fixtures):
    with open_slide(fixtures / "synth_tumor.tif") as r:
        lt = r.thumbnail_level(8.0)
        thumb = r.read_level(lt)
        res = tissue_mask(thumb, r.meta.mpp_at(lt), strict=False)
        assert 0.05 < res.tissue_frac < 0.95

        lw, _ = r.level_for_mpp(0.5)
        geom = ann.load(fixtures / "synth_tumor.xml", "synth_tumor",
                        r.level_dims[0])
        recs = list(iter_patches("synth_tumor", r.level_dims, r.level_downsamples,
                                 lw, res.mask, geom, PatchConfig()))
        assert recs, "tissue gating removed every patch"
        assert all(x.tissue_frac >= 0.10 for x in recs)
        assert any(x.tumor_frac > 0.05 for x in recs), "no tumor patches indexed"
        assert max(x.tumor_frac for x in recs) <= 1.0

        coords, summary = plan_positions(r.level_dims, r.level_downsamples, lw,
                                         res.mask, PlannerConfig())
        assert 0 < summary["kept"] <= summary["grid"]


def test_full_slide_reconstruction_covers_all_tissue(fixtures):
    with open_slide(fixtures / "synth_edge.tif") as r:
        lt = r.thumbnail_level(8.0)
        res = tissue_mask(r.read_level(lt), r.meta.mpp_at(lt), strict=False)
        lw, _ = r.level_for_mpp(0.5)
        cfg = PlannerConfig(patch_size=256, stride=128)
        coords, _ = plan_positions(r.level_dims, r.level_downsamples, lw,
                                   res.mask, cfg)
        if not coords:
            pytest.skip("fixture has no tissue at the working level")
        h0, w0 = r.level_dims[0][1], r.level_dims[0][0]
        st = GaussianStitcher((h0, w0), r.level_downsamples[lw],
                              StitchConfig(patch_size=256, heat_downsample=4,
                                           chunk=128))
        for i in range(0, len(coords), 8):
            c = coords[i:i + 8]
            st.add(np.full((len(c), 256, 256), 0.42, np.float32), c)
        frac, _ = st.coverage_report(res.mask)
        assert frac == 0.0
        heat = st.finalize(tissue=res.mask, dtype=np.float32)
        den = np.asarray(st.den[:, :])
        assert np.abs(heat[den >= 0.05] - 0.42).max() < 1e-3
