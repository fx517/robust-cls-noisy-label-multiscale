"""
Double-Softmax Loss (DSPT: Intrinsic Gradient Suppression for Label-Noise).

Reference:
    Li et al., "Intrinsic Gradient Suppression for Label-Noise Prompt Tuning
    in Vision-Language Models", ICML 2026. arXiv:2605.00591

Key insight:
    Apply two sequential softmax normalizations to the logits
    (p = softmax(z), q = softmax(p)), then compute the cross-entropy on q.
    This "saturation region" adaptively suppresses the gradient of high-error
    noisy samples to nearly 0 (because they conflict with CLIP's strong prior
    and produce extreme gradients), while retaining useful gradients for
    informative samples. Hyperparameter-free, bounded, drop-in, and a
    particularly good fit for VLMs with strong priors.

    L = -log( softmax(softmax(z))_y )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleSoftmaxLoss(nn.Module):
    """
    Double-softmax cross-entropy loss for noisy-label fine-tuning.

    Two sequential softmax normalizations:
        p = softmax(outputs)      # 1st: logits -> probabilities
        q = softmax(p)            # 2nd: renormalize the probabilities (saturation region)
        loss = -log(q[y])         # cross-entropy

    Properties (from the original paper):
        - Bounded: log(1 + (C-1)/e) <= L <= log(e + C - 1)
        - Gradient tends to zero for high-error noisy samples (suppresses memorization of noisy labels)
        - Gradient is retained for clean/uncertain but informative samples

    Args:
        num_classes: Number of classes (used only for record-keeping)
    """

    def __init__(self, num_classes: int = 400):
        super().__init__()
        self.num_classes = num_classes

    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            outputs: (B, C) model logits
            labels: (B,) hard labels, or (B, C) soft labels

        Returns:
            double-softmax cross-entropy loss (scalar)
        """
        # First softmax -> probabilities p
        p = F.softmax(outputs, dim=1)

        # Second softmax (sequential probability normalization) -> q
        q = F.softmax(p, dim=1)
        q = torch.clamp(q, 1e-7, 1.0)
        log_q = torch.log(q)

        soft = labels.dim() > 1 and labels.dtype in (
            torch.float16, torch.float32, torch.float64,
        )
        if soft:
            # Soft labels (MixUp/CutMix): CE form
            return -(labels * log_q).sum(dim=1).mean()
        else:
            # Hard labels
            return -log_q.gather(1, labels.unsqueeze(1)).squeeze(1).mean()