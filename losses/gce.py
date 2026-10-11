"""
Generalized Cross-Entropy (GCE) Loss.

Reference:
    Zhang & Sabuncu, "Generalized Cross Entropy Loss for Training Deep Neural
    Networks with Noisy Labels", NeurIPS 2018.

The GCE loss is a generalization between MAE (L1) and CE (Cross-Entropy):

    L_q(f(x), y) = (1 - f_y(x)^q) / q

where q ∈ (0, 1] controls the noise robustness:
- q → 0: behaves like standard CE (emphasizes hard examples)
- q = 1: behaves like MAE (treats all examples equally, noise robust)
- Intermediate q (e.g., 0.7): balances noise robustness and convergence speed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCELoss(nn.Module):
    """
    Generalized Cross-Entropy loss for noisy-label learning.

    The loss smoothly interpolates between CE (q→0, fast convergence but
    noise-sensitive) and MAE (q=1, noise-robust but slow convergence).

    Args:
        q: Interpolation parameter. q ∈ (0, 1].
           Lower q: closer to CE, faster convergence, more noise-sensitive
           Higher q: closer to MAE, slower convergence, more noise-robust
           Recommended: 0.7
        num_classes: Number of classes (for truncation)
        apply_softmax: Whether to apply softmax to inputs
    """

    def __init__(
        self,
        q: float = 0.7,
        num_classes: int = 400,
        apply_softmax: bool = True,
    ):
        super().__init__()

        assert 0.0 < q <= 1.0, f"q must be in (0, 1], got {q}"
        self.q = q
        self.num_classes = num_classes
        self.apply_softmax = apply_softmax

    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GCE loss.

        Args:
            outputs: (B, C) model logits
            labels: (B,) ground-truth labels (integers)

        Returns:
            GCE loss (scalar), gradient-truncated for noise robustness
        """
        if self.apply_softmax:
            probs = F.softmax(outputs, dim=1)
        else:
            probs = outputs

        # Clamp for numerical stability
        probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)

        if labels.dim() > 1 and labels.dtype in (torch.float16, torch.float32, torch.float64):
            # Soft labels (from MixUp/CutMix)
            pred_probs = (probs * labels).sum(dim=1)
        else:
            # Hard labels
            pred_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

        # GCE formula: (1 - p_y^q) / q
        loss = (1.0 - torch.pow(pred_probs, self.q)) / self.q

        return loss.mean()


class TruncatedGCELoss(nn.Module):
    """
    Truncated GCE: applies GCE only to samples with prediction below threshold.

    This prevents the loss from being dominated by already-well-classified samples.
    Samples with p_y > threshold get zero loss.
    """

    def __init__(
        self,
        q: float = 0.7,
        num_classes: int = 400,
        threshold: float = 0.1,
    ):
        super().__init__()

        self.q = q
        self.num_classes = num_classes
        self.threshold = threshold

    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        probs = F.softmax(outputs, dim=1)
        probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)

        if labels.dim() > 1:
            pred_probs = (probs * labels).sum(dim=1)
        else:
            pred_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

        # GCE per sample
        gce = (1.0 - torch.pow(pred_probs, self.q)) / self.q

        # Truncation mask: only apply to low-confidence samples
        mask = (pred_probs < 1.0 - self.threshold).float()
        gce = gce * mask

        # Normalize by number of active samples
        active = mask.sum()
        if active > 0:
            return gce.sum() / active
        return gce.mean()
