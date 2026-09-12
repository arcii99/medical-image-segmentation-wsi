"""Config composition, freezing, and hashing (ADR-014).

`utils.config` is the only module allowed to read YAML or the environment.
Everything else receives a frozen `ResolvedConfig` by value, so the config
hash that names the run directory is honest.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

__all__ = ["ResolvedConfig", "load", "config_hash"]


class ResolvedConfig(Mapping):
    """Immutable, attribute-accessible, hashable config tree."""

    __slots__ = ("_d",)

    def __init__(self, d: Mapping[str, Any]):
        object.__setattr__(self, "_d", {
            k: ResolvedConfig(v) if isinstance(v, Mapping) else
               tuple(v) if isinstance(v, list) else v
            for k, v in d.items()
        })

    def __getattr__(self, k: str) -> Any:
        try:
            return self._d[k]
        except KeyError as e:
            raise AttributeError(f"no config key {k!r} (have {sorted(self._d)})") from e

    def __getitem__(self, k: str) -> Any: return self._d[k]
    def __iter__(self): return iter(self._d)
    def __len__(self) -> int: return len(self._d)
    def __setattr__(self, *_): raise TypeError("ResolvedConfig is immutable")
    def __repr__(self) -> str: return f"ResolvedConfig({self.to_dict()!r})"

    def get(self, k: str, default: Any = None) -> Any:
        v = self._d.get(k, default)
        return v

    def to_dict(self) -> dict[str, Any]:
        return {k: v.to_dict() if isinstance(v, ResolvedConfig) else
                   list(v) if isinstance(v, tuple) else v
                for k, v in self._d.items()}


def _deep_merge(base: dict, over: Mapping) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, Mapping) else v
    return out


def _coerce(s: str) -> Any:
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def load(paths: Sequence[str | Path], overrides: Sequence[str] = ()) -> ResolvedConfig:
    """Compose YAML files left-to-right, then apply `key.sub=value` overrides."""
    merged: dict[str, Any] = {}
    for p in paths:
        with open(p) as fh:
            merged = _deep_merge(merged, yaml.safe_load(fh) or {})

    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        key, _, raw = ov.partition("=")
        node = merged
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(raw.strip())

    root = os.environ.get("WSI_DATA_ROOT")
    if root:
        merged.setdefault("paths", {})["root"] = root

    # Batch-size/LR coupling is expressed here so it cannot drift (ADR-011).
    tr = merged.get("train", {})
    if tr.get("lr_scale_with_batch", False) and "batch_size" in tr:
        ref = tr.get("lr_reference_batch", 16)
        for k in ("lr_decoder", "lr_encoder"):
            if k in tr:
                tr[k] = float(tr[k]) * tr["batch_size"] / ref
    return ResolvedConfig(merged)


def config_hash(cfg: ResolvedConfig | Mapping, n: int = 8) -> str:
    d = cfg.to_dict() if isinstance(cfg, ResolvedConfig) else dict(cfg)
    blob = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:n]


def _cli() -> None:
    import argparse
    p = argparse.ArgumentParser(prog="src.utils.config")
    p.add_argument("configs", nargs="+")
    p.add_argument("--print-hash", action="store_true")
    p.add_argument("--set", action="append", default=[])
    a = p.parse_args()
    cfg = load(a.configs, a.set)
    print(config_hash(cfg) if a.print_hash else json.dumps(cfg.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    _cli()
