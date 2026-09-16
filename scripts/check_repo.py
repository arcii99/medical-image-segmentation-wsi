#!/usr/bin/env python
"""Verify the git repository actually contains the whole project.

Exists because a bare `data/` line in .gitignore silently excluded
`src/data/` -- five modules absent from every commit, discovered only when a
clone on another machine failed its version check. Git reports nothing for a
file it was told to ignore, so the omission is invisible locally.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MUST_TRACK = ["src", "scripts", "configs", "tests", "docs", "notebooks"]


def main() -> int:
    try:
        tracked = set(subprocess.check_output(
            ["git", "ls-files"], cwd=ROOT, text=True).split())
    except Exception as e:  # noqa: BLE001
        print(f"not a git repository, or git unavailable: {e}")
        return 1

    missing: list[str] = []
    for top in MUST_TRACK:
        d = ROOT / top
        if not d.exists():
            continue
        for f in d.rglob("*"):
            if not f.is_file() or "__pycache__" in f.parts:
                continue
            if f.suffix in (".pyc", ".tif", ".xml", ".png"):
                continue
            rel = f.relative_to(ROOT).as_posix()
            if rel not in tracked:
                missing.append(rel)

    print(f"tracked files: {len(tracked)}")
    if missing:
        print(f"\nNOT TRACKED BY GIT ({len(missing)}):")
        for m in sorted(missing)[:40]:
            why = subprocess.run(["git", "check-ignore", "-v", m], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip()
            print(f"  {m}" + (f"\n      ignored by {why.split(chr(9))[0]}" if why else ""))
        print("\nA clone of this repository would be INCOMPLETE.")
        return 1
    print("\nevery source file is tracked -- a clone would be complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
