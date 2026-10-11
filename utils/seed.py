"""
Random seed utilities for reproducible training.

Sets all relevant random seeds (Python, NumPy, PyTorch, CUDA) and
configures CUDA determinism settings.
"""

import random
import os
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Integer seed value
        deterministic: If True, enable CUDA determinism (slower but fully reproducible).

    Note:
        Full CUDA determinism significantly slows training. For most purposes,
        setting the seed without deterministic=True is sufficient.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # PyTorch 2.0+ deterministic algorithms
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    print(f"[Seed] Set random seed to {seed} (deterministic={deterministic})")
