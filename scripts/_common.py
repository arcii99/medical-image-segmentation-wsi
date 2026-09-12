"""Shared CLI plumbing for the stage scripts."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def base_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog)
    p.add_argument("--config", nargs="+",
                   default=["configs/base.yaml", "configs/data_camelyon16.yaml"])
    p.add_argument("--set", action="append", default=[], metavar="key=value")
    return p


def git_state() -> tuple[str, bool]:
    """(sha, dirty). Recorded in every artifact so results stay attributable."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                             cwd=ROOT, text=True).strip())
        return sha, dirty
    except Exception:
        return "nogit", True


def discover(raw_dir: Path, exts=(".tif", ".tiff", ".svs", ".ndpi")) -> list[Path]:
    return sorted(p for p in Path(raw_dir).rglob("*") if p.suffix.lower() in exts)
