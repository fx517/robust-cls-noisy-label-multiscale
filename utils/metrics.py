"""
Metrics computation utilities.

Includes accuracy calculation, per-class metrics, and the AverageMeter
utility for tracking running statistics during training.
"""

from typing import List, Dict, Optional

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


class AverageMeter:
    """
    Computes and stores the average and current value.

    Useful for tracking running means of loss, accuracy, etc. during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def __repr__(self) -> str:
        return f"{self.avg:.4f}"


def topk_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: tuple = (1, 5),
) -> List[float]:
    """
    Compute top-k accuracy for the given predictions.

    Args:
        output: (B, C) model logits
        target: (B,) ground-truth labels
        topk: Tuple of k values (e.g., (1, 5))

    Returns:
        List of accuracy values for each k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append((correct_k / batch_size).item())
        return results


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    average: str = "macro",
) -> Dict:
    """
    Compute comprehensive classification metrics.

    Args:
        predictions: (N,) predicted class indices
        labels: (N,) ground-truth class indices
        class_names: Optional list of class names
        average: Averaging method for multi-class metrics

    Returns:
        Dict with accuracy, precision, recall, f1, per-class breakdown
    """
    metrics = {}

    # Overall accuracy
    metrics["accuracy"] = accuracy_score(labels, predictions)

    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=average, zero_division=0
    )
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1_score"] = f1

    # Per-class metrics
    per_class = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    if class_names:
        metrics["per_class"] = {}
        for i, name in enumerate(class_names):
            if i < len(per_class[0]):
                metrics["per_class"][name] = {
                    "precision": per_class[0][i],
                    "recall": per_class[1][i],
                    "f1": per_class[2][i],
                    "support": per_class[3][i],
                }

    return metrics


def confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    # Normalize per row (true class)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
    return cm.astype(np.float32) / row_sums
