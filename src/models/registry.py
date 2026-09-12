"""Model registry.

Indirection over segmentation_models_pytorch so the coupling to that API lives
in exactly one file (ADR-007 Consequences).
"""
from __future__ import annotations

from typing import Any, Callable

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def build(cfg) -> Any:
    name = str(cfg.name if hasattr(cfg, "name") else cfg["name"])
    if name not in _REGISTRY:
        import src.models.unet_effb0  # noqa: F401  (populate registry)
        try:
            import src.models.mask2former  # noqa: F401
        except ImportError:
            pass
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; have {sorted(_REGISTRY)}")
    kwargs = {k: v for k, v in (cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)).items()
              if k != "name"}
    return _REGISTRY[name](**kwargs)


def _cli() -> None:
    import argparse
    import torch
    p = argparse.ArgumentParser(prog="src.models.registry")
    p.add_argument("--summary", required=True)
    p.add_argument("--size", type=int, default=512)
    a = p.parse_args()
    m = build({"name": a.summary})
    x = torch.zeros(2, 3, a.size, a.size)
    with torch.no_grad():
        y = m(x)
    total = sum(p.numel() for p in m.parameters())
    train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    enc = sum(p.numel() for n, p in m.named_parameters() if n.startswith("encoder"))
    print(f"input  {list(x.shape)} -> output {list(y.shape)}")
    print(f"params total {total:,}   trainable {train:,}   encoder {enc:,}")


if __name__ == "__main__":
    _cli()
