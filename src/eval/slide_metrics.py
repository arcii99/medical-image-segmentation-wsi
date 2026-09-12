"""Pixel- and slide-level metrics."""
from __future__ import annotations

import numpy as np


def dice_iou(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None):
    p = np.asarray(pred).astype(bool); g = np.asarray(gt).astype(bool)
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        p, g = p & m, g & m
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    return {"dice": dice, "iou": iou, "tp": tp, "fp": fp, "fn": fn}


def slide_score(heat: np.ndarray, tissue: np.ndarray | None = None) -> float:
    h = np.asarray(heat, dtype=np.float32)
    if tissue is not None and tissue.shape == h.shape:
        h = h[tissue.astype(bool)]
    return float(h.max()) if h.size else 0.0


def roc_auc(labels, scores) -> float:
    """Rank-based AUC; no sklearn dependency."""
    y = np.asarray(labels).astype(int); s = np.asarray(scores, dtype=float)
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.size); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
