"""
Combined loss strategies for noisy-label fine-grained recognition.

Provides flexible combinations of ELR, GCE, and auxiliary losses including
self-distillation and contrastive learning components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .elr import ELRLoss
from .gce import GCELoss, TruncatedGCELoss
from .double_softmax import DoubleSoftmaxLoss


class CombinedLoss(nn.Module):
    """
    Combinable loss wrapper supporting ELR, GCE, self-distillation, and more.

    Usage:
        loss_fn = CombinedLoss(
            num_examples=43087, num_classes=400,
            loss_type="elr+gce", elr_lambda=3.0, gce_q=0.7,
        )
        for images, labels, indices in loader:
            logits, features = model(images, return_features=True)
            loss = loss_fn(indices, logits, labels, features=features)
    """

    def __init__(
        self,
        num_examples: int,
        num_classes: int = 400,
        loss_type: str = "elr",
        # ELR params
        elr_lambda: float = 3.0,
        elr_beta: float = 0.7,
        elr_weight: float = 1.0,
        # GCE params
        gce_q: float = 0.7,
        gce_weight: float = 0.5,
        gce_truncated: bool = False,
        gce_threshold: float = 0.1,
        # Double-softmax (DSPT)
        ds_weight: float = 1.0,
        # Self-distillation
        use_distill: bool = False,
        distill_temperature: float = 4.0,
        distill_weight: float = 0.5,
        # Label smoothing
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.loss_type = loss_type
        self.use_distill = use_distill
        self.distill_temperature = distill_temperature
        self.distill_weight = distill_weight
        self.label_smoothing = label_smoothing
        self.elr_weight = elr_weight
        self.gce_weight = gce_weight
        self.ds_weight = ds_weight

        # Initialize sub-losses
        self.elr = None
        self.gce = None
        self.ds = None

        if "elr" in loss_type:
            self.elr = ELRLoss(
                num_examples=num_examples,
                num_classes=num_classes,
                lmbda=elr_lambda,
                beta=elr_beta,
            )

        if "gce" in loss_type:
            if gce_truncated:
                self.gce = TruncatedGCELoss(
                    q=gce_q,
                    num_classes=num_classes,
                    threshold=gce_threshold,
                )
            else:
                self.gce = GCELoss(
                    q=gce_q,
                    num_classes=num_classes,
                )

        if "ds" in loss_type or "dsp" in loss_type:
            self.ds = DoubleSoftmaxLoss(num_classes=num_classes)

    def forward(
        self,
        indices: torch.Tensor,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor = None,
        teacher_logits: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            indices: (B,) sample indices for ELR
            outputs: (B, C) model logits
            labels: (B,) or (B, C) ground-truth (hard or soft)
            features: (B, D) visual features (for contrastive loss, optional)
            teacher_logits: (B, C) teacher model logits for distillation

        Returns:
            Combined loss scalar
        """
        is_soft = labels.dtype in (torch.float16, torch.float32, torch.float64) and labels.dim() > 1
        total_loss = torch.tensor(0.0, device=outputs.device)

        # --- ELR loss ---
        if self.elr is not None:
            if not is_soft:
                # ELR needs integer labels for cross-entropy component
                elr_loss = self.elr(indices, outputs, labels)
                total_loss = total_loss + self.elr_weight * elr_loss
            else:
                # With soft labels, use GCE or CE as base
                pass

        # --- GCE loss ---
        if self.gce is not None:
            if is_soft:
                # GCE with soft labels: use CE as fallback
                probs = F.softmax(outputs, dim=1)
                probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
                gce_loss = -(labels * torch.log(probs)).sum(dim=1).mean()
            else:
                gce_loss = self.gce(outputs, labels)
            total_loss = total_loss + self.gce_weight * gce_loss

        # --- Double-Softmax loss (DSPT) ---
        if self.ds is not None:
            ds_loss = self.ds(outputs, labels)
            total_loss = total_loss + self.ds_weight * ds_loss

        # --- Plain cross-entropy fallback ---
        if self.elr is None and self.gce is None and self.ds is None:
            if is_soft:
                probs = F.softmax(outputs, dim=1)
                probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
                total_loss = -(labels * torch.log(probs)).sum(dim=1).mean()
            else:
                total_loss = F.cross_entropy(outputs, labels, label_smoothing=self.label_smoothing)

        # --- Self-distillation loss ---
        if self.use_distill and teacher_logits is not None:
            distill_loss = self._distillation_loss(outputs, teacher_logits)
            total_loss = total_loss + self.distill_weight * distill_loss

        return total_loss

    def _distillation_loss(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """KL divergence distillation loss."""
        student_soft = F.log_softmax(
            student_logits / self.distill_temperature, dim=1
        )
        teacher_soft = F.softmax(
            teacher_logits / self.distill_temperature, dim=1
        )
        return F.kl_div(
            student_soft, teacher_soft, reduction="batchmean"
        ) * (self.distill_temperature ** 2)

    def detect_noisy_labels(self, given_labels: torch.Tensor) -> torch.Tensor:
        """Use ELR targets to detect noisy labels (requires ELR)."""
        if self.elr is not None:
            return self.elr.detect_noisy_labels(given_labels)
        return torch.zeros_like(given_labels, dtype=torch.bool)

    def get_elr_targets(self) -> torch.Tensor:
        """Get full ELR target matrix (requires ELR)."""
        if self.elr is not None:
            return self.elr.get_all_targets()
        return torch.tensor([])


def get_loss_function(
    loss_type: str = "elr",
    num_examples: int = 43087,
    num_classes: int = 400,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a loss function.

    Args:
        loss_type: "ce", "elr", "gce", "elr+gce", "truncated_gce"
        num_examples: Number of training samples (for ELR)
        num_classes: Number of classes
        **kwargs: Additional loss parameters

    Returns:
        Loss module
    """
    if loss_type == "ce":
        return lambda idx, out, lbl: F.cross_entropy(out, lbl)

    if loss_type == "elr":
        return ELRLoss(
            num_examples=num_examples,
            num_classes=num_classes,
            lmbda=kwargs.get("elr_lambda", 3.0),
            beta=kwargs.get("elr_beta", 0.7),
        )

    if loss_type == "gce":
        return GCELoss(
            q=kwargs.get("gce_q", 0.7),
            num_classes=num_classes,
        )

    if loss_type == "truncated_gce":
        return TruncatedGCELoss(
            q=kwargs.get("gce_q", 0.7),
            num_classes=num_classes,
            threshold=kwargs.get("gce_threshold", 0.1),
        )

    # Combined
    return CombinedLoss(
        num_examples=num_examples,
        num_classes=num_classes,
        loss_type=loss_type,
        **kwargs,
    )
