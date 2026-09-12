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
