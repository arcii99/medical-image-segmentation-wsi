"""Cosine schedule with linear warmup (ADR-011).

Warmup matters here specifically because the balanced sampler makes early
batches unrepresentative of the slide-level class prior.
"""
from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup(optimizer, warmup_epochs: int, total_epochs: int,
                       min_factor: float = 1e-6 / 3e-4):
    def fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        prog = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * prog))
    return LambdaLR(optimizer, fn)


def param_groups(model, lr_encoder: float, lr_decoder: float, weight_decay: float):
    """Discriminative LR, with BN and bias excluded from weight decay."""
    groups = {"enc_decay": [], "enc_nodecay": [], "dec_decay": [], "dec_nodecay": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        enc = name.startswith("encoder")
        nodecay = p.ndim <= 1 or name.endswith(".bias")
        groups[f"{'enc' if enc else 'dec'}_{'nodecay' if nodecay else 'decay'}"].append(p)
    return [
        {"params": groups["enc_decay"], "lr": lr_encoder, "weight_decay": weight_decay},
        {"params": groups["enc_nodecay"], "lr": lr_encoder, "weight_decay": 0.0},
        {"params": groups["dec_decay"], "lr": lr_decoder, "weight_decay": weight_decay},
        {"params": groups["dec_nodecay"], "lr": lr_decoder, "weight_decay": 0.0},
    ]
