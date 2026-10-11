"""
Early-Learning Regularization (ELR) Loss.

Reference:
    Liu et al., "Early-Learning Regularization Prevents Memorization of Noisy Labels"
    NeurIPS 2020. https://arxiv.org/abs/2007.00151

Key insight:
    DNNs trained on noisy labels first learn clean patterns ("early learning phase")
    before eventually memorizing the noisy labels. ELR prevents memorization by
    penalizing predictions that diverge from an EMA of early predictions.

The loss: L = CE(y_pred, y_true) + λ * log(1 - <target_ema, y_pred>)

where target_ema maintains an exponential moving average of model predictions
for each training sample. This EMA captures the model's "early" beliefs about
each sample, which are more likely to be correct (since clean patterns are
learned first). When the model later tries to memorize a wrong label, the
ELR term penalizes it because the EMA target disagrees.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ELRLoss(nn.Module):
    """
    Early-Learning Regularization loss for noisy-label training.

    Maintains a per-sample EMA of prediction probabilities. For clean samples,
    predictions and EMA agree → low regularization. For noisy samples,
    the model's attempt to memorize conflicts with the EMA → high penalty.

    Args:
        num_examples: Total number of training samples
        num_classes: Number of classification classes
        lmbda: Regularization strength (λ). Higher = stronger noise resistance.
               Typical values: 1-10, tune based on noise level.
        beta: EMA momentum for target estimation. Range: 0.5-0.99.
              Higher beta = slower adaptation. Robust across values.
    """

    def __init__(
        self,
        num_examples: int,
        num_classes: int = 400,
        lmbda: float = 3.0,
        beta: float = 0.7,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.lmbda = lmbda
        self.beta = beta

        # Per-sample EMA target: (N, C) tensor
        # Registered as buffer so it moves with .cuda() / .to(device)
        self.register_buffer(
            "target",
            torch.zeros(num_examples, num_classes),
        )

        # Track update counts per sample (for debugging)
        self.register_buffer(
            "update_count",
            torch.zeros(num_examples, dtype=torch.long),
        )

    def forward(
        self,
        indices: torch.Tensor,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ELR loss for a batch.

        Args:
            indices: (B,) sample indices in the dataset (0 <= idx < num_examples)
            outputs: (B, C) model logits
            labels: (B,) ground-truth labels (integers or soft targets)

        Returns:
            Combined CE + ELR regularization loss (scalar)
        """
        # Check for soft labels (from MixUp/CutMix)
        is_soft = labels.dtype in (torch.float16, torch.float32, torch.float64) and labels.dim() > 1

        # Get predicted probabilities
        y_pred = F.softmax(outputs, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)

        # Detach for EMA update (stop gradient through target)
        y_pred_detach = y_pred.detach()

        # --- Update per-sample EMA targets ---
        # target[idx] = beta * target[idx] + (1-beta) * y_pred
        with torch.no_grad():
            # Normalize detach predictions for stability
            y_pred_normalized = y_pred_detach / y_pred_detach.sum(dim=1, keepdim=True)

            # The target buffer already sits on GPU with the criterion; update in place to avoid host<->device sync
            idx = indices.to(self.target.device)
            self.target[idx] = (
                self.beta * self.target[idx]
                + (1 - self.beta) * y_pred_normalized.to(
                    device=self.target.device, dtype=self.target.dtype
                )
            )
            self.update_count[idx] += 1

        # --- Cross-entropy loss ---
        if is_soft:
            # Soft labels (MixUp/CutMix): compute KL divergence
            # CE(p, q) = -sum(p * log(q)) where p is soft target, q is prediction
            ce_loss = -(labels * torch.log(y_pred)).sum(dim=1).mean()
        else:
            # Standard CE with label smoothing for additional regularization
            ce_loss = F.cross_entropy(outputs, labels, label_smoothing=0.05)

        # --- ELR regularization term ---
        # log(1 - <target, y_pred>), averaged over batch
        target_pred = self.target[indices.to(self.target.device)].to(y_pred.device)  # (B, C)
        inner_prod = (target_pred * y_pred).sum(dim=1)  # (B,)
        elr_reg = torch.log(1.0 - inner_prod + 1e-8).mean()

        # Combined loss
        total_loss = ce_loss + self.lmbda * elr_reg

        return total_loss

    def get_targets(self, indices: torch.Tensor) -> torch.Tensor:
        """Get the current EMA target for given sample indices."""
        return self.target[indices.to(self.target.device)].to(indices.device)

    def get_all_targets(self) -> torch.Tensor:
        """Get the full target matrix (N, C)."""
        return self.target.clone()

    def detect_noisy_labels(self, given_labels: torch.Tensor) -> torch.Tensor:
        """
        After training, detect potentially noisy samples.

        A sample is considered noisy if:
        argmax(EMA_target) != given_label

        Args:
            given_labels: (N,) original labels from the dataset

        Returns:
            noisy_mask: (N,) boolean tensor (True = estimated noisy)
        """
        predicted_labels = self.target.argmax(dim=1)
        return predicted_labels != given_labels.to(self.target.device)

    def reset_targets(self):
        """Reset EMA targets (useful for restarting training)."""
        self.target.zero_()
        self.update_count.zero_()
