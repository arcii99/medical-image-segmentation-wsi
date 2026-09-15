"""Confusion-count metrics and threshold sweeping.

Counts accumulate on the GPU as int64; pulling per-batch python floats is both
slow and a source of drift.  The threshold sweep here is what produces the tau
stored in the checkpoint (ADR-006) -- five hops downstream depend on it, and
any of them defaulting to 0.5 is a silent 5-10 point F1 error.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)


@dataclass
class ConfusionAccumulator:
    thresholds: np.ndarray = field(default_factory=lambda: THRESHOLDS)
    tp: torch.Tensor | None = None
    fp: torch.Tensor | None = None
    fn: torch.Tensor | None = None
    # Overlap alone hides a model that is right but unconfident: Dice can be
    # 0.9999 while every probability sits at 0.52. That state looks solved and
    # is not -- the margin is what a decision threshold has to work with, and
    # it is what makes tau stable.
    _p_sum_pos: torch.Tensor | None = None
    _p_sum_neg: torch.Tensor | None = None
    _n_pos: torch.Tensor | None = None
    _n_neg: torch.Tensor | None = None

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        p = torch.sigmoid(logits.detach().float())
        t = (target.detach().float() > 0.5)
        if self.tp is None:
            z = lambda: torch.zeros(len(self.thresholds), dtype=torch.long, device=p.device)
            self.tp, self.fp, self.fn = z(), z(), z()
            s0 = lambda: torch.zeros((), dtype=torch.float64, device=p.device)
            self._p_sum_pos, self._p_sum_neg = s0(), s0()
            self._n_pos, self._n_neg = s0(), s0()

        self._p_sum_pos += p[t].double().sum()
        self._p_sum_neg += p[~t].double().sum()
        self._n_pos += t.sum()
        self._n_neg += (~t).sum()
        for i, th in enumerate(self.thresholds):
            pred = p > float(th)
            self.tp[i] += (pred & t).sum()
            self.fp[i] += (pred & ~t).sum()
            self.fn[i] += (~pred & t).sum()

    def dice_per_threshold(self) -> np.ndarray:
        tp = self.tp.double(); fp = self.fp.double(); fn = self.fn.double()
        return (2 * tp / (2 * tp + fp + fn).clamp(min=1)).cpu().numpy()

    def best(self) -> tuple[float, float]:
        """(best_threshold, best_dice)."""
        d = self.dice_per_threshold()
        i = int(np.argmax(d))
        return float(self.thresholds[i]), float(d[i])

    def confidence(self) -> dict[str, float]:
        """Mean predicted probability on each class, and the gap between them."""
        if self._n_pos is None:
            return {"p_pos": float("nan"), "p_neg": float("nan"),
                    "margin": float("nan")}
        pp = float(self._p_sum_pos / self._n_pos.clamp(min=1))
        pn = float(self._p_sum_neg / self._n_neg.clamp(min=1))
        return {"p_pos": pp, "p_neg": pn, "margin": pp - pn}

    def summary(self) -> dict[str, float]:
        th, dice = self.best()
        i = int(np.argmin(np.abs(self.thresholds - th)))
        tp, fp, fn = (int(x[i]) for x in (self.tp, self.fp, self.fn))
        iou = tp / max(tp + fp + fn, 1)
        return {"dice": dice, "iou": iou, "threshold": th,
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1), **self.confidence()}
