"""BUG-007: splits are per patient, stable, and never overlap."""
import collections

import pandas as pd
import pytest

from src.data.index import assert_no_split_leakage
from src.utils.splits import assign


def test_assignment_is_stable():
    assert assign("patient_007") == assign("patient_007")


def test_distribution_is_roughly_70_15_15():
    c = collections.Counter(assign(f"p{i:04d}") for i in range(4000))
    total = sum(c.values())
    assert 0.66 < c["train"] / total < 0.74
    assert 0.11 < c["val"] / total < 0.19
    assert 0.11 < c["test"] / total < 0.19


def test_no_slide_in_two_splits():
    ok = pd.DataFrame({"slide_id": ["a", "b"], "split": ["train", "val"]})
    assert_no_split_leakage(ok)
    bad = pd.DataFrame({"slide_id": ["a", "a"], "split": ["train", "val"]})
    with pytest.raises(ValueError, match="leakage"):
        assert_no_split_leakage(bad)
