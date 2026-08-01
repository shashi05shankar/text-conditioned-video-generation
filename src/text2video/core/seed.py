"""Deterministic seeding.

Every experiment records its seed so runs can be reproduced. `seed_everything`
covers Python, NumPy and PyTorch (CPU + CUDA).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed all RNGs used in this project.

    Args:
        seed: the seed value, recorded in the run log.
        deterministic: if True, force cuDNN into deterministic mode. This makes
            results bit-reproducible on GPU at some cost to speed, so it is off by
            default for real training runs and turned on for tests.

    Returns:
        The seed, so callers can log exactly what was used.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a distinct, reproducible seed.

    Without this, forked workers can share a NumPy RNG state and generate
    identical "random" samples -- a classic silent data-pipeline bug.
    """
    base_seed = torch.initial_seed() % (2**32)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)
