"""Fixed random seeds for every source of randomness the experiments use."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seeds Python, NumPy and, when it is installed, PyTorch.

    Python's hash seed cannot be changed from inside a running interpreter. Start
    experiments through scripts/run.sh, which sets PYTHONHASHSEED=0. gensim, which
    Graph2Vec is built on, also needs a single worker thread to be repeatable.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def hash_seed_is_fixed() -> bool:
    """True when the interpreter was started with a fixed PYTHONHASHSEED."""
    value = os.environ.get("PYTHONHASHSEED", "random")
    return value != "random"
