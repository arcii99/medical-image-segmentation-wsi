"""Class-balanced sampling with hard-negative mining (ADR-006)."""
from __future__ import annotations

import logging

import numpy as np

try:
    from torch.utils.data import Sampler
except ImportError:
    class Sampler:                                    # type: ignore[no-redef]
        pass

log = logging.getLogger(__name__)


class BalancedSampler(Sampler):
    """Draw 1 tumor : `neg_per_pos` normal, plus a trickle of boundary patches.

    Boundary patches (0 < tumor_frac <= threshold) get their own small quota;
    folding them into the negatives would teach the model that faint evidence
    is negative.
    """

    def __init__(self, index, neg_per_pos: int = 3, tumor_threshold: float = 0.05,
                 boundary_rate: float = 0.10, hard_cap: float = 0.15, seed: int = 1337):
        tf = index.tumor_frac.to_numpy()
        self.pos = np.flatnonzero(tf > tumor_threshold)
        self.bnd = np.flatnonzero((tf > 0) & (tf <= tumor_threshold))
        self.neg = np.flatnonzero(tf == 0)
        if self.pos.size == 0:
            raise ValueError("no tumor patches in the index; check annotations (V4.4)")
        self.neg_per_pos = neg_per_pos
        self.boundary_rate = boundary_rate
        self.hard_cap = hard_cap
        self.rng = np.random.default_rng(seed)
        self.hard: np.ndarray = np.array([], dtype=int)
        log.info("sampler pools: %d tumor / %d boundary / %d normal",
                 self.pos.size, self.bnd.size, self.neg.size)

    def __len__(self) -> int:
        return int(self.pos.size * (1 + self.neg_per_pos + self.boundary_rate))

    def __iter__(self):
        n_pos = self.pos.size
        n_neg = n_pos * self.neg_per_pos
        n_bnd = int(n_pos * self.boundary_rate)

        neg_pool = self.neg
        if self.hard.size:
            n_hard = min(self.hard.size, int(n_neg * self.hard_cap))
            hard = self.rng.choice(self.hard, n_hard, replace=False)
            rest = self.rng.choice(neg_pool, n_neg - n_hard, replace=neg_pool.size < n_neg)
            negs = np.concatenate([hard, rest])
        else:
            negs = self.rng.choice(neg_pool, n_neg, replace=neg_pool.size < n_neg)

        parts = [self.pos, negs]
        if n_bnd and self.bnd.size:
            parts.append(self.rng.choice(self.bnd, n_bnd, replace=self.bnd.size < n_bnd))
        out = np.concatenate(parts)
        self.rng.shuffle(out)
        return iter(out.tolist())

    def set_hard_negatives(self, idx: np.ndarray) -> None:
        """Promote false-positive-heavy normals for the next few epochs."""
        self.hard = np.asarray(idx, dtype=int)
        log.info("hard-negative pool: %d patches", self.hard.size)
