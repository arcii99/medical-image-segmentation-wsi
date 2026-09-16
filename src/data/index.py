"""Patch index IO -- the pipeline's narrow waist (Architecture contract A4)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

SCHEMA = {
    "slide_id": "string", "x0": "int32", "y0": "int32", "level": "int8",
    "size": "int16", "tissue_frac": "float32", "tumor_frac": "float32",
    "split": "string",
}


def save(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    df.astype({k: v for k, v in SCHEMA.items() if k in df}).to_parquet(
        path, index=False, compression="zstd")
    return path


def load(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = set(SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(
            f"patch index missing columns {sorted(missing)}; splits must be "
            "assigned in stage 01 and persisted, never derived at train time "
            "(ADR-010 / BUG-007)")
    return df


def index_hash(path: str | Path, n: int = 16) -> str:
    """Content hash, recorded in the checkpoint so a model and its data pair."""
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def assert_splits_usable(df: pd.DataFrame, tumor_threshold: float = 0.05) -> None:
    """Every split must exist and be trainable/evaluable.

    Catches a whole family of silent split-construction errors -- most
    importantly a patient rule that collapses many slides into one identifier
    and so forces them all into the same split. That failure produces an
    index that looks fine, trains to completion, and yields a meaningless
    evaluation (see src/utils/splits.PATIENT_RULES).
    """
    problems: list[str] = []
    present = set(df["split"].unique())
    for name in ("train", "val", "test"):
        if name not in present:
            problems.append(f"split {name!r} is empty")

    n_patients = df["slide_id"].nunique()
    if n_patients < 3:
        problems.append(f"only {n_patients} distinct slides in the index")

    for name in ("train", "val"):
        sub = df[df["split"] == name]
        if len(sub) and not (sub["tumor_frac"] > tumor_threshold).any():
            problems.append(
                f"split {name!r} contains no tumor patches "
                "(sampling and threshold fitting both need them)")
    test = df[df["split"] == "test"]
    if len(test) and not (test["tumor_frac"] > tumor_threshold).any():
        problems.append("split 'test' contains no tumor patches; "
                        "FROC and AUC would be undefined")

    if problems:
        counts = df.groupby("split").slide_id.nunique().to_dict()
        raise ValueError(
            "unusable split structure:\n  - " + "\n  - ".join(problems)
            + f"\nslides per split: {counts}"
            + "\nCheck splits.patient_from in your data config.")


def assert_no_split_leakage(df: pd.DataFrame) -> None:
    """Gate V4.2. Blocking -- never waived."""
    groups = {k: set(g.slide_id) for k, g in df.groupby("split")}
    for a in groups:
        for b in groups:
            if a < b and (shared := groups[a] & groups[b]):
                raise ValueError(
                    f"slide-level split leakage: {len(shared)} slides in both "
                    f"{a!r} and {b!r}, e.g. {sorted(shared)[:3]}")
