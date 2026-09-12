"""Report assembly and attribution enforcement (ADR-014)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RefusalError(RuntimeError):
    """Raised when a number would be produced that cannot be attributed."""


def assert_reportable(ckpt: dict, allow_dirty: bool = False) -> None:
    if ckpt.get("dirty") and not allow_dirty:
        raise RefusalError(
            f"checkpoint was produced from a dirty tree "
            f"(git_sha {ckpt.get('git_sha', '?')[:7]}-dirty). "
            "Re-run with --allow-dirty to produce non-reportable numbers.")


def assert_attrs_match(attrs: dict, ckpt: dict) -> None:
    """A heatmap and the checkpoint that scores it must be the same model."""
    a, b = attrs.get("model_run_id"), ckpt.get("run_id")
    if a and b and a != b:
        raise RefusalError(f"heatmap was produced by run {a!r}, not {b!r}")
    ta, tb = attrs.get("threshold"), ckpt.get("threshold")
    if ta is not None and tb is not None and abs(float(ta) - float(tb)) > 1e-9:
        raise RefusalError(f"heatmap threshold {ta} != checkpoint threshold {tb}")


def write(out_dir: str | Path, metrics: dict[str, Any],
          config_snapshot: dict[str, Any] | None = None) -> Path:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    if config_snapshot is not None:
        (out / "config_snapshot.json").write_text(
            json.dumps(config_snapshot, indent=2, default=str))
    return out / "metrics.json"
