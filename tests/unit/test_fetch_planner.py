"""The subset planner must never starve one class to fill the budget."""
import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "fetch", Path(__file__).resolve().parents[2] / "scripts" / "00_fetch_data.py")
fetch = importlib.util.module_from_spec(spec)
sys.modules["fetch"] = fetch
spec.loader.exec_module(fetch)

GB = 1024 ** 3


def _listing(n_tumor=111, n_normal=160, size_gb=2.5):
    rows = []
    for i in range(1, n_tumor + 1):
        rows.append(f"2018-01-01 00:00:00 {int(size_gb*GB):>12d} "
                    f"CAMELYON16/images/tumor_{i:03d}.tif")
        rows.append(f"2018-01-01 00:00:00 {50_000:>12d} "
                    f"CAMELYON16/annotations/tumor_{i:03d}.xml")
    for i in range(1, n_normal + 1):
        rows.append(f"2018-01-01 00:00:00 {int(size_gb*GB):>12d} "
                    f"CAMELYON16/images/normal_{i:03d}.tif")
    return fetch.parse_listing("\n".join(rows))


def test_parses_sizes_and_classes():
    objs = _listing(2, 3)
    slides = [o for o in objs if o.is_slide]
    assert len(slides) == 5
    assert sum(o.is_annotation for o in objs) == 2
    assert {o.kind for o in slides} == {"tumor", "normal"}


def test_both_classes_present_on_a_tight_budget():
    """The original bug: all-tumor-first filled the budget with no negatives."""
    _, st = fetch.plan(_listing(), budget_gb=140, reserve_gb=40)
    assert st["tumor_selected"] > 0
    assert st["normal_selected"] > 0, "a cohort with no normals has no clean negatives"


def test_ratio_is_respected():
    for target in (0.5, 1.0, 1.5):
        _, st = fetch.plan(_listing(), budget_gb=240, reserve_gb=40,
                           normal_per_tumor=target)
        got = st["normal_selected"] / st["tumor_selected"]
        assert abs(got - target) < 0.15, f"target {target}, got {got:.2f}"


def test_stays_within_budget():
    chosen, st = fetch.plan(_listing(), budget_gb=140, reserve_gb=40)
    assert st["bytes"] <= st["budget_bytes"]
    assert sum(o.size for o in chosen) == st["bytes"]


def test_selection_is_deterministic():
    a, _ = fetch.plan(_listing(), budget_gb=200, reserve_gb=40, seed=1337)
    b, _ = fetch.plan(_listing(), budget_gb=200, reserve_gb=40, seed=1337)
    assert [o.key for o in a] == [o.key for o in b]
    c, _ = fetch.plan(_listing(), budget_gb=200, reserve_gb=40, seed=99)
    assert [o.key for o in a] != [o.key for o in c]


def test_annotations_are_always_included():
    chosen, st = fetch.plan(_listing(), budget_gb=60, reserve_gb=40)
    assert sum(o.is_annotation for o in chosen) == st["annotations"] == 111


def test_impossible_budget_exits():
    with pytest.raises(SystemExit):
        fetch.plan(_listing(), budget_gb=30, reserve_gb=40)


# --- CAMELYON16 bucket has four parallel trees; only images/ holds WSIs ---

REAL_LAYOUT = """\
2018-01-01 00:00:00   1892341760 CAMELYON16/images/tumor_001.tif
2018-01-01 00:00:00   1992341760 CAMELYON16/images/normal_001.tif
2018-01-01 00:00:00     22341760 CAMELYON16/masks/tumor_001_mask.tif
2018-01-01 00:00:00     22341760 CAMELYON16/masks/normal_001_mask.tif
2018-01-01 00:00:00      1034176 CAMELYON16/background_tissue/tumor_001_tissue.tif
2018-01-01 00:00:00      1034176 CAMELYON16/background_tissue/normal_001_tissue.tif
2018-01-01 00:00:00       271234 CAMELYON16/annotations/tumor_001.xml
2018-01-01 00:00:00         1024 CAMELYON16/README.md
"""


def test_only_images_prefix_counts_as_a_slide():
    objs = fetch.parse_listing(REAL_LAYOUT)
    slides = [o for o in objs if o.is_slide]
    assert {o.stem for o in slides} == {"tumor_001", "normal_001"}, \
        "masks/ and background_tissue/ rasters must not be treated as WSIs"


def test_mask_and_tissue_rasters_are_flagged_excluded():
    objs = fetch.parse_listing(REAL_LAYOUT)
    excluded = {o.stem for o in objs if o.is_excluded_raster}
    assert excluded == {"tumor_001_mask", "normal_001_mask",
                        "tumor_001_tissue", "normal_001_tissue"}


def test_annotations_come_from_the_annotations_prefix():
    objs = fetch.parse_listing(REAL_LAYOUT)
    anns = [o for o in objs if o.is_annotation]
    assert [o.stem for o in anns] == ["tumor_001"]


def test_plan_never_selects_a_mask_raster():
    objs = fetch.parse_listing(REAL_LAYOUT)
    chosen, _ = fetch.plan(objs, budget_gb=50, reserve_gb=0)
    assert not any(o.is_excluded_raster for o in chosen)
