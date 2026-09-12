"""Seeding. Full determinism costs ~20% throughput, so warn_only (ADR-011)."""
from __future__ import annotations

import os
import random

import numpy as np

BASE_SEED = 1337


def seed_everything(seed: int = BASE_SEED, deterministic: bool = True) -> int:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass
    return seed
