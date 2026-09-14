"""Patient-level split assignment (ADR-010).

Assigned once in stage 01 and persisted. Deriving splits at training time is
how BUG-007 happened: patch-level splits leak near-duplicate neighbours across
train and val and inflate validation Dice by 15-25 points.
"""
from __future__ import annotations

import hashlib
import re

__all__ = ["assign", "assign_stratified", "patient_id", "SPLITS",
           "PATIENT_RULES"]

SPLITS = ("train", "val", "test")

# How to get a patient identifier out of a slide filename.
#
# This MUST be set per cohort. Guessing is dangerous: a rule that merges
# unrelated slides into one "patient" forces them all into the same split.
# On CAMELYON16 the old prefix-stripping heuristic mapped all 110 tumor
# slides to patient "tumor" and all 160 normal slides to patient "normal",
# which put every tumor slide in train, every normal slide in test, left
# validation empty, and produced a test set containing no tumor at all.
# Training would have run to completion and the numbers would have been
# meaningless.
PATIENT_RULES = {
    # CAMELYON16: one slide per case -> the slide IS the patient.
    "slide_id": None,
    # CAMELYON17: patient_017_node_2.tif -> patient_017
    "camelyon17": r"^(patient_\d+)",
}


def patient_id(slide_id: str, rule: str = "slide_id") -> str:
    """Derive a patient identifier from a slide filename stem.

    ``rule`` is either a key of PATIENT_RULES or a regex with one capture
    group. An unmatched regex raises rather than silently falling back --
    a wrong patient grouping is invisible until evaluation and by then a
    full training run has been wasted.
    """
    pattern = PATIENT_RULES.get(rule, rule)
    if pattern is None:
        return slide_id
    m = re.match(pattern, slide_id)
    if not m or not m.groups():
        raise ValueError(
            f"patient rule {rule!r} did not match slide {slide_id!r}. "
            "Set splits.patient_from correctly for this cohort "
            "(use 'slide_id' when each slide is its own case)."
        )
    return m.group(1)


def assign(patient_id: str, seed: int = 1337,
           bounds: tuple[int, int] = (70, 85)) -> str:
    """Stable hash-based split. Adding slides never reshuffles existing ones."""
    h = hashlib.md5(f"{seed}:{patient_id}".encode()).hexdigest()
    bucket = int(h[:8], 16) % 100
    lo, hi = bounds
    if bucket < lo:
        return "train"
    return "val" if bucket < hi else "test"


def assign_stratified(
    patient_class: dict[str, str],
    seed: int = 1337,
    bounds: tuple[int, int] = (70, 85),
) -> dict[str, str]:
    """Assign splits so each class hits the target proportions exactly.

    ``patient_class`` maps patient id -> class label (e.g. "tumor"/"normal").

    Why this exists. Independent per-patient hashing (:func:`assign`) gives the
    right proportions *in expectation*, but a small cohort can draw badly. On
    the first real CAMELYON16 subset, 61 tumor patients produced a validation
    set with 2 tumor slides against an expected 9.2 -- a 0.34% outcome. Two
    tumor slides cannot support threshold fitting or early stopping, so the
    run would have been governed by noise.

    Ranking within each class removes the variance: patients are ordered by a
    seeded hash (so the choice is arbitrary but reproducible) and cut at the
    proportion boundaries.

    Trade-off against ADR-010. Pure hashing has the property that adding
    slides never moves an existing slide between splits. Ranking gives that
    up: a new patient can shift the cut points. This is acceptable because the
    split is written into ``slides.parquet`` on the first run and read from
    there afterwards -- but re-running stage 01 on a *grown* cohort will
    produce a different assignment, and any model trained on the old one must
    be re-evaluated, not compared across.
    """
    lo, hi = bounds
    by_class: dict[str, list[str]] = {}
    for pid, cls in patient_class.items():
        by_class.setdefault(cls, []).append(pid)

    out: dict[str, str] = {}
    for cls, pids in by_class.items():
        ordered = sorted(pids, key=lambda p: hashlib.md5(
            f"{seed}:{cls}:{p}".encode()).hexdigest())
        n = len(ordered)
        n_train = round(n * lo / 100)
        n_val = round(n * (hi - lo) / 100)
        # Guarantee at least one patient per split once the class has three.
        if n >= 3:
            n_val = max(n_val, 1)
            n_train = min(n_train, n - n_val - 1)
            n_train = max(n_train, 1)
        for i, pid in enumerate(ordered):
            out[pid] = ("train" if i < n_train
                        else "val" if i < n_train + n_val else "test")
    return out
