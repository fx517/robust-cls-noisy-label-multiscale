"""
Learning rate schedulers for fine-tuning.

Supports cosine annealing with warmup, step decay, and plateau-based scheduling.
"""

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    StepLR,
    ReduceLROnPlateau,
    OneCycleLR,
    _LRScheduler,
)


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine annealing with linear warmup.

    - Linear warmup from warmup_lr to base_lr over warmup_epochs
    - Cosine decay from base_lr to min_lr over remaining epochs
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        warmup_lr: float = 1e-6,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            progress = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.warmup_lr + (base_lr - self.warmup_lr) * progress
                for base_lr in self.base_lrs
            ]

        # Cosine decay
        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_decay
            for base_lr in self.base_lrs
        ]


def build_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    warmup_epochs: int = 5,
    total_epochs: int = 100,
    warmup_lr: float = 1e-6,
    min_lr: float = 1e-7,
    step_size: int = 30,
    step_gamma: float = 0.1,
    steps_per_epoch: int = None,
) -> _LRScheduler:
    """
    Build a learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: "cosine", "step", or "plateau"
        warmup_epochs: Warmup duration (cosine only)
        total_epochs: Total training epochs (cosine only)
        warmup_lr: Initial warmup LR
        min_lr: Minimum LR after decay
        step_size: Step size for step scheduler
        step_gamma: Decay factor for step scheduler

    Returns:
        LR scheduler instance
    """
    if scheduler_type == "onecycle":
        # OneCycleLR: per-batch stepping for fast convergence
        total_steps = total_epochs * (steps_per_epoch or 100)
        return OneCycleLR(
            optimizer,
            max_lr=[pg["lr"] for pg in optimizer.param_groups],
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            final_div_factor=1e4,
        )

    if scheduler_type == "cosine":
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            warmup_lr=warmup_lr,
            min_lr=min_lr,
        )

    elif scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=step_size,
            gamma=step_gamma,
        )

    elif scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=step_gamma,
            patience=10,
            min_lr=min_lr,
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
