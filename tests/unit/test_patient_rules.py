"""Patient-ID derivation must not collapse unrelated slides into one case."""
import pandas as pd
import pytest

from src.data.index import assert_splits_usable
from src.utils.splits import assign, patient_id


def test_camelyon16_slide_is_its_own_patient():
    assert patient_id("tumor_001", "slide_id") == "tumor_001"
    assert patient_id("normal_088", "slide_id") == "normal_088"


def test_camelyon16_splits_are_not_degenerate():
    """The bug: a prefix rule put every tumor slide in one split."""
    names = [f"tumor_{i:03d}" for i in range(1, 111)] + \
            [f"normal_{i:03d}" for i in range(1, 161)]
    by_kind = {}
    for n in names:
        by_kind.setdefault(n.split("_")[0], set()).add(
            assign(patient_id(n, "slide_id")))
    assert by_kind["tumor"] == {"train", "val", "test"}
    assert by_kind["normal"] == {"train", "val", "test"}


def test_camelyon17_groups_nodes_of_one_patient():
    nodes = [f"patient_017_node_{i}" for i in range(5)]
    pids = {patient_id(n, "camelyon17") for n in nodes}
    assert pids == {"patient_017"}
    assert len({assign(p) for p in pids}) == 1


def test_unmatched_patient_rule_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="did not match"):
        patient_id("no-underscores-here", "camelyon17")


def _idx(rows):
    return pd.DataFrame(rows, columns=["slide_id", "split", "tumor_frac"])


def test_split_usability_gate_catches_the_original_bug():
    bad = _idx([("tumor_001", "train", 0.4), ("tumor_002", "train", 0.3),
                ("normal_001", "test", 0.0), ("normal_002", "test", 0.0)])
    with pytest.raises(ValueError, match="split 'val' is empty"):
        assert_splits_usable(bad)


def test_split_usability_gate_catches_tumor_free_test_set():
    bad = _idx([("a", "train", 0.4), ("b", "val", 0.3), ("c", "test", 0.0)])
    with pytest.raises(ValueError, match="test.*no tumor"):
        assert_splits_usable(bad)


def test_healthy_index_passes():
    ok = _idx([("a", "train", 0.4), ("b", "val", 0.3), ("c", "test", 0.2)])
    assert_splits_usable(ok)


# --- stratified assignment: proportions must hold WITHIN each class ---

from src.utils.splits import assign_stratified  # noqa: E402


def _cohort(n_tumor=61, n_normal=70):
    d = {f"tumor_{i:03d}": "tumor" for i in range(n_tumor)}
    d.update({f"normal_{i:03d}": "normal" for i in range(n_normal)})
    return d


def _counts(mapping, classes):
    out = {}
    for pid, split in mapping.items():
        out.setdefault(classes[pid], {}).setdefault(split, 0)
        out[classes[pid]][split] += 1
    return out


def test_each_class_hits_its_target_proportions():
    c = _cohort()
    got = _counts(assign_stratified(c), c)
    for cls, n in (("tumor", 61), ("normal", 70)):
        assert abs(got[cls]["train"] / n - 0.70) < 0.06
        assert abs(got[cls]["val"] / n - 0.15) < 0.06
        assert abs(got[cls]["test"] / n - 0.15) < 0.06


def test_fixes_the_observed_two_tumor_validation_draw():
    """Per-slide hashing gave val=2 tumor on the real cohort (P = 0.0034)."""
    c = _cohort()
    got = _counts(assign_stratified(c), c)
    assert got["tumor"]["val"] >= 6, "validation needs enough tumor to fit tau"


def test_every_split_populated_for_a_three_patient_class():
    c = {f"t{i}": "tumor" for i in range(3)}
    assert set(assign_stratified(c).values()) == {"train", "val", "test"}


def test_is_deterministic_and_seed_sensitive():
    c = _cohort()
    assert assign_stratified(c, 1337) == assign_stratified(c, 1337)
    assert assign_stratified(c, 1337) != assign_stratified(c, 99)


def test_no_patient_receives_two_splits():
    c = _cohort()
    m = assign_stratified(c)
    assert set(m) == set(c)
    assert all(v in ("train", "val", "test") for v in m.values())


# --- CAMELYON16 ITC convention (BUG-025) ---

from shapely.geometry import box  # noqa: E402

from src.eval import froc as F  # noqa: E402


def test_major_axis_uses_rotated_rectangle_not_bbox():
    """A diagonal lesion must not be inflated by an axis-aligned box."""
    from shapely.affinity import rotate
    p = box(0, 0, 100, 10)
    assert F.major_axis_um(p, 1.0) == pytest.approx(100, abs=1)
    assert F.major_axis_um(rotate(p, 45), 1.0) == pytest.approx(100, abs=1)


def test_clinical_classification_boundaries():
    assert F.classify(box(0, 0, 3000, 3000), 1.0) == "macro"
    assert F.classify(box(0, 0, 400, 400), 1.0) == "micro"
    assert F.classify(box(0, 0, 50, 50), 1.0) == "itc"


def test_itc_excluded_from_denominator():
    polys = [box(0, 0, 10, 10), box(100, 100, 900, 900)]
    ev, itc = F.split_evaluable(polys, 1.0)
    assert len(ev) == 1 and len(itc) == 1


class _Les:
    def __init__(self, cy, cx, score=0.9):
        self.centroid_yx = (cy, cx)
        self.score_max = score


def test_prediction_on_an_itc_is_not_a_false_positive():
    gt = [box(0, 0, 40, 40)]                     # 40 um -> ITC at mpp 1.0
    dets, n_ev, n_itc = F.match([_Les(20, 20)], gt, "s", 1.0)
    assert n_ev == 0 and n_itc == 1
    assert dets == [], "a hit on an ITC must be dropped, not charged as an FP"


def test_real_miss_is_still_a_false_positive():
    gt = [box(0, 0, 900, 900)]
    dets, n_ev, _ = F.match([_Les(5000, 5000)], gt, "s", 1.0)
    assert n_ev == 1 and len(dets) == 1 and dets[0].hit is False


def test_duplicate_hits_on_one_lesion_count_once():
    gt = [box(0, 0, 900, 900)]
    dets, n_ev, _ = F.match([_Les(100, 100), _Les(200, 200), _Les(300, 300)],
                            gt, "s", 1.0)
    assert n_ev == 1
    assert sum(d.hit for d in dets) == 1
    assert sum(not d.hit for d in dets) == 0, "extra hits are dropped, not FPs"


# --- config merge order (BUG-027) ---

from pathlib import Path as _Path  # noqa: E402

from src.utils.config import load as _load  # noqa: E402


def _resolve(user):
    """Mirrors the default-insertion logic in scripts/03_train.py."""
    paths = list(user)
    if not any("model" in _Path(p).stem for p in paths):
        paths.insert(0, "configs/model_unet_effb0.yaml")
    return _load(paths)


BASE = ["configs/base.yaml", "configs/data_camelyon16.yaml"]


def test_trailing_config_wins():
    """cpu_smoke last must actually apply -- appending the model config after
    it silently restored batch 16 and the run was OOM-killed."""
    cfg = _resolve(BASE + ["configs/model_unet_effb0.yaml",
                           "configs/cpu_smoke.yaml"])
    assert cfg.train.batch_size == 4
    assert cfg.hw.device == "cpu"
    assert cfg.data.augment is False


def test_model_config_inserted_when_absent():
    cfg = _resolve(BASE)
    assert cfg.model.name == "unet_effb0"
    assert cfg.train.batch_size == 16       # the documented GPU default


def test_patchset_config_survives():
    cfg = _resolve(BASE + ["configs/model_unet_effb0.yaml",
                           "configs/patchset.yaml"])
    assert cfg.data.source == "patchset"
    assert cfg.model.name == "unet_effb0"


def test_cli_set_overrides_every_file():
    cfg = _load(BASE + ["configs/model_unet_effb0.yaml",
                        "configs/cpu_smoke.yaml"], ["train.batch_size=2"])
    assert cfg.train.batch_size == 2
