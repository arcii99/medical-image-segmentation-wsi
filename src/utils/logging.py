"""Console + JSONL logging. Metrics go to JSONL so gates can grep them."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

_FMT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"


def setup(run_dir: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(_FMT, "%H:%M:%S"))
    root.addHandler(h)
    if run_dir:
        run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(run_dir / "run.log")
        fh.setFormatter(logging.Formatter(_FMT))
        root.addHandler(fh)
    return root


class MetricWriter:
    """Append-only JSONL metric sink."""

    def __init__(self, run_dir: str | Path, name: str = "metrics.jsonl"):
        self.path = Path(run_dir) / name
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()

    def log(self, **kv: Any) -> None:
        kv.setdefault("wall_s", round(time.time() - self._t0, 3))
        with self.path.open("a") as fh:
            fh.write(json.dumps(kv, default=str) + "\n")
