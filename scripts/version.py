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
      ("scripts/qc_tissue_montage.py", "render_cell"),
      ("scripts/01_build_tissue_masks.py", "slide_class")]),
    ("0.5.0", "CAMELYON16 ITC convention in FROC + index audit",
     [("src/eval/froc.py", "ITC_MAX_AXIS_UM"),
      ("src/eval/froc.py", "major_axis_um"),
      ("scripts/qc_index.py", "ITC_MAX_AXIS_UM")]),
]


# Tree hashes of published releases, so a stale tree can be identified even
# when its own marker list predates the release it is missing.
KNOWN_HASHES = {
    "0.4.0": "4dc9dde93baf0408",
    "0.5.0": "c6a27035dd31fd51",
}


def main() -> int:
    declared = (ROOT / "VERSION").read_text().strip() \
        if (ROOT / "VERSION").exists() else "<no VERSION file>"
    print(f"declared version : {declared}")
    print(f"project root     : {ROOT}\n")
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
    files = sorted(p for p in (ROOT / "src").rglob("*.py"))
    h = hashlib.sha256()
    for p in files:
        h.update(p.read_bytes())
    digest = h.hexdigest()[:16]
    print(f"src/ tree hash   : {digest}  ({len(files)} files)")
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
    raise SystemExit(main())
