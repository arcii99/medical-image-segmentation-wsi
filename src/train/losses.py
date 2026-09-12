"""Losses (ADR-006).

BCE + Dice, not pure Dice.  On an all-negative patch Dice's gradient is
degenerate -- it pushes p -> 0 with a magnitude independent of how wrong p is
elsewhere -- and with a 1:3 sampler ~75% of patches are empty, so the model
collapses to predicting zero (BUG-006).  BCE supplies a well-behaved per-pixel
gradient there; Dice supplies the overlap signal where there is something to
overlap.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SoftDiceLoss", "BCEDiceLoss", "FocalLoss", "build_loss"]


class SoftDiceLoss(nn.Module):
    """Per-image soft Dice.

    Two details that matter:

    * ``smooth`` appears in numerator and denominator, so dice(empty, empty)
      is exactly 1.0 rather than NaN.
    * Reduction is **per image, then mean** -- never over the flattened batch.
      Batch-flattened Dice lets one large lesion dominate fifteen empty
      patches and hides exactly the failure mode we care about.
    """

    def __init__(self, smooth: float = 1.0, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits) if self.from_logits else logits
        p = p.float().flatten(1)
        t = target.float().flatten(1)
        inter = (p * t).sum(1)
        denom = p.sum(1) + t.sum(1)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Ablation only; underperformed BCE+Dice by ~0.9 Dice in a 10-epoch pilot."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * target + (1 - p) * (1 - target)
        at = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (at * (1 - pt).pow(self.gamma) * bce).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, w_bce: float = 0.5, w_dice: float = 0.5,
                 pos_weight: float = 2.0, smooth: float = 1.0):
        super().__init__()
        self.w_bce, self.w_dice = w_bce, w_dice
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.dice = SoftDiceLoss(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Reduce in fp32: a Dice sum over 262k pixels in bf16 loses precision (W2).
        bce = F.binary_cross_entropy_with_logits(
            logits.float(), target.float(), pos_weight=self.pos_weight.to(logits.device))
        return self.w_bce * bce + self.w_dice * self.dice(logits.float(), target)


def build_loss(cfg) -> nn.Module:
    name = str(getattr(cfg, "name", "bce_dice")).lower()
    if name == "bce_dice":
        return BCEDiceLoss(cfg.get("w_bce", 0.5), cfg.get("w_dice", 0.5),
                           cfg.get("pos_weight", 2.0))
    if name == "dice":
        return SoftDiceLoss()
    if name == "focal":
        return FocalLoss(cfg.get("gamma", 2.0), cfg.get("alpha", 0.25))
    raise ValueError(f"unknown loss {name!r}")
