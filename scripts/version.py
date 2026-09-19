#!/usr/bin/env python
"""Report which build of the project is actually on disk.

Exists because "did the update get applied?" turned out to be a real and
repeatedly costly question: several debugging rounds were spent on symptoms
that were simply the previous version still running.

IMPORTANT LIMITATION. This script ships inside the tree it inspects, so it
can only check for versions it already knows about. A tree that is entirely
one release behind reports its own VERSION as consistent and says nothing is
wrong -- which is exactly what happened at 0.4.0 while 0.5.0 was current.

The reliable signal is therefore the ``src`` tree hash printed at the bottom.
Compare it against the hash quoted with the release you meant to install. The
checklist above only tells you the tree is internally consistent.
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Features introduced by version, each identified by something checkable on
# disk rather than by a string someone could forget to bump.
MARKERS = [
    ("0.2.0", "explicit patient rules + split-usability gate",
     [("src/utils/splits.py", "PATIENT_RULES"),
      ("src/data/index.py", "assert_splits_usable")]),
    ("0.3.0", "Philips MPP parsing + images/-only fetch",
     [("src/io/slide.py", "_philips_mpp"),
      ("scripts/00_fetch_data.py", "IMAGES_PREFIX")]),
    ("0.4.0", "stratified splits + physical tissue guard + montage QC",
     [("src/utils/splits.py", "assign_stratified"),
      ("src/preprocess/tissue.py", "min_tissue_mm2"),
      ("scripts/qc_tissue_montage.py", "render_cell")]),
    ("0.5.0", "CAMELYON16 ITC convention in FROC + index audit",
     [("src/eval/froc.py", "ITC_MAX_AXIS_UM"),
      ("scripts/qc_index.py", "major_axis_um")]),
    ("0.6.0", "post-processing filters by major axis, matching evaluation",
     [("src/infer/postproc.py", "min_lesion_axis_um"),
      ("src/infer/postproc.py", "_major_axes")]),
    ("0.7.0", "portable patchset export + CPU path",
     [("scripts/07_export_patchset.py", "jitter-copies"),
      ("src/data/patchset.py", "PatchSetDataset"),
      ("configs/cpu_smoke.yaml", "overfit_batches"),
      ("update.sh", "NOTHING TO DO"),
      ("update.sh", ".update.sh.incoming")]),
    ("0.8.0", "training path verified end to end (gate V6.2 passed)",
     [("scripts/03_train.py", "Insert the default FIRST"),
      ("scripts/03_train.py", "likely to be OOM-killed"),
      ("scripts/03_train.py", "wiring is sound"),
      ("scripts/03_train.py", "OVERFIT MODE"),
      ("scripts/03_train.py", "overfit criterion met"),
      ("scripts/03_train.py", "GATE V6.2"),
      ("scripts/03_train.py", "max_val_batches"),
      ("src/train/metrics.py", "def confidence"),
      ("docs/COLAB.md", "Why not just put the slides on Drive")]),
    ("0.8.1", "environment preflight (wrong conda env named directly)",
     [("src/utils/envcheck.py", "assert_environment"),
      ("scripts/_common.py", "assert_environment")]),
]


# Tree hashes of published releases, so a stale tree can be identified even
# when its own marker list predates the release it is missing.
KNOWN_HASHES = {
    "0.9.9": "17f1ac908c8b6f2b",
    "0.9.8": "6257afd32a4bebfc",
    "0.9.7": "fb31de282fbe239e",
    "0.9.6": "ea0efbf9bc831dc4",
    "0.9.5": "1d09c2ed9b99811b",
    "0.9.4": "8375d4fec76ee9d7",
    "0.9.3": "b7b4ef99aa4646db",
    "0.9.2": "47249ea2e23d56aa",
    "0.9.1": "db3437ea73fb6f2f",
    # Hashes are over src + scripts + configs + update.sh + Makefile +
    # pyproject + VERSION, excluding this file. Entries marked "old scheme"
    # were computed over narrower sets during 0.4-0.7 and are kept only to
    # identify a stale tree.
    "0.8.2": "7b9c2016792a843e",
    "0.8.6": "6a1f19e1217f7315",
    "0.8.1": "d0bd2b78d7a92102",
    "0.8.2": "7b9c2016792a843e",
    "0.8.6": "6a1f19e1217f7315",
    "0.8.1": "a2ba17dd32d79089",
    "0.8.2": "7b9c2016792a843e",
    "0.8.6": "6a1f19e1217f7315",
    "0.8.1": "2207c1e9a6201432",
    "0.9.0": "c95e9afe17619bfb",
    "0.8.0": "9abc903764072f14",
    "0.4.0 (old scheme)": "4dc9dde93baf0408",
    "0.5.0 (old scheme)": "c6a27035dd31fd51",
    "0.6.0 (old scheme)": "8f480e32048247e9",
    "0.7.0 (old scheme)": "76602045f53100df",
}


def tree_hash() -> tuple[str, int]:
    """The authoritative tree hash.

    Exposed as a function, and writable via --record, because every time the
    packaging step reimplemented this logic it drifted: a one-file difference
    in the glob produced a different digest, and the recorded value then
    disagreed with what the tool itself reported. One implementation, one
    answer.
    """
    extra = [ROOT / "update.sh", ROOT / "VERSION", ROOT / "Makefile",
             ROOT / "pyproject.toml", ROOT / ".importlinter"]
    files = sorted(
        q for q in
        list((ROOT / "src").rglob("*.py"))
        + list((ROOT / "scripts").rglob("*.py"))
        + list((ROOT / "configs").rglob("*.yaml"))
        + [e for e in extra if e.exists()]
        if q.resolve() != pathlib.Path(__file__).resolve())
    h = hashlib.sha256()
    for q in files:
        h.update(q.relative_to(ROOT).as_posix().encode())
        h.update(q.read_bytes())
    return h.hexdigest()[:16], len(files)


def record() -> int:
    """Write the current tree hash into KNOWN_HASHES for the current VERSION."""
    ver = (ROOT / "VERSION").read_text().strip()
    digest, n = tree_hash()
    src = pathlib.Path(__file__)
    s = src.read_text()
    s = re.sub(rf'    "{re.escape(ver)}": "[0-9a-f]*",\n', "", s)
    s = s.replace("KNOWN_HASHES = {\n", f'KNOWN_HASHES = {{\n    "{ver}": "{digest}",\n')
    src.write_text(s)
    after, _ = tree_hash()
    print(f"recorded {ver} = {digest} over {n} files")
    print("stable" if after == digest else f"UNSTABLE -- now {after}")
    return 0 if after == digest else 1


def main() -> int:
    declared = (ROOT / "VERSION").read_text().strip() \
        if (ROOT / "VERSION").exists() else "<no VERSION file>"
    print(f"declared version : {declared}")
    print(f"project root     : {ROOT}")
    try:
        sys.path.insert(0, str(ROOT))
        from src.utils.envcheck import assert_environment, describe
        print("  " + describe().replace("\n", "\n  "))
        probs = assert_environment(strict=False)
        for pr in probs:
            print(f"  ENVIRONMENT PROBLEM: {pr}")
    except Exception as e:  # noqa: BLE001
        print(f"  (environment check unavailable: {e})")
    print()
    print("  (this file ships with the tree, so it cannot detect a release")
    print("   newer than itself -- compare the tree hash below)\n")

    worst = None
    for ver, desc, checks in MARKERS:
        missing = []
        for rel, needle in checks:
            p = ROOT / rel
            if not p.exists():
                missing.append(f"{rel} (file absent)")
            elif needle not in p.read_text(errors="ignore"):
                missing.append(f"{rel} (missing '{needle}')")
        ok = not missing
        print(f"  [{'x' if ok else ' '}] {ver}  {desc}")
        for m in missing:
            print(f"        - {m}")
        if not ok and worst is None:
            worst = ver

    print()
    digest, n_files = tree_hash()
    print(f"tree hash        : {digest}  ({n_files} files)")
    print("  known release hashes:")
    for ver, hsh in sorted(KNOWN_HASHES.items()):
        mark = "  <-- this tree" if hsh == digest else ""
        print(f"    {ver:8s} {hsh}{mark}")
    if digest not in KNOWN_HASHES.values():
        print("    (this tree matches no published release -- locally modified,")
        print("     or a release whose hash is not recorded here)")

    if worst:
        print(f"\nINCOMPLETE -- the {worst} changes are not all present.")
        print("Re-extract the tarball over this directory, then re-run this.")
        return 1
    print(f"\nCONSISTENT with VERSION {declared}.")
    print("This does NOT mean it is the latest -- check the tree hash above.")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(record() if "--record" in sys.argv else main())
