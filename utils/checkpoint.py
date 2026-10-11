"""
Checkpoint saving and loading utilities.

Handles model state, optimizer state, scheduler state, and training
metadata for seamless training resumption.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    epoch: int = 0,
    best_acc: float = 0.0,
    **extra,
) -> None:
    """
    Save a training checkpoint.

    Args:
        path: Output file path
        model: Model to save
        optimizer: Optimizer state (optional)
        scheduler: LR scheduler state (optional)
        epoch: Current epoch number
        best_acc: Best accuracy so far
        **extra: Additional data to save (train_losses, val_accs, etc.)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_acc": best_acc,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    checkpoint.update(extra)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        path: Checkpoint file path
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to map tensors to
        strict: Whether to strictly match state dict keys

    Returns:
        Dict with checkpoint metadata (epoch, best_acc, etc.)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        k: v
        for k, v in checkpoint.items()
        if k not in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict")
    }


def get_latest_checkpoint(checkpoint_dir: str, pattern: str = "checkpoint_epoch_*.pth") -> Optional[str]:
    """
    Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory to search
        pattern: Glob pattern to match checkpoint files

    Returns:
        Path to the latest checkpoint, or None if none found
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = sorted(
        checkpoint_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
    )

    if checkpoints:
        # Prefer best_model.pth over epoch checkpoints
        best = checkpoint_dir / "best_model.pth"
        if best.exists():
            return str(best)
        return str(checkpoints[-1])

    return None
