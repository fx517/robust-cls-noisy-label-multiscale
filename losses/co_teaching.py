"""
Co-Teaching for Robust Learning with Noisy Labels.

Reference:
    Han et al., "Co-teaching: Robust Training of Deep Neural Networks
    with Extremely Noisy Labels", NeurIPS 2018.
    https://arxiv.org/abs/1804.06872

Core idea:
    Train two networks simultaneously. Each network selects samples with
    the smallest loss (cleanest samples) and feeds them to the OTHER network.
    This prevents both networks from memorizing the same noisy patterns
    — they have different learning trajectories and thus different errors.

    Additionally uses a "remember rate" R(t) that decays over time:
    epoch 1: R=1.0 (use 100% of data)
    epoch Tk: R=τ   (use τ% of data, drop (1-τ)% with highest loss)

Integration:
    This module wraps the per-sample loss computation and selection.
    The trainer uses two model instances.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoTeachingSelector:
    """
    Co-Teaching sample selector.

    For each batch, selects the top R(t) fraction of samples with smallest
    loss, separately for each model, then exchanges the selections.

    Args:
        forget_rate: Final fraction of samples to forget (τ). Default: 0.3
        num_gradual: Epochs over which to decay remember rate. Default: 10
        num_classes: Number of classes
    """

    def __init__(
        self,
        forget_rate: float = 0.3,
        num_gradual: int = 10,
        num_classes: int = 500,
    ):
        self.forget_rate = forget_rate
        self.num_gradual = num_gradual
        self.num_classes = num_classes
        self.epoch = 0

    def get_remember_rate(self, epoch: int = None) -> float:
        """
        Compute remember rate for current epoch.

        R(t) = 1 - min(1, t/Tk) * τ
        where t=epoch, Tk=num_gradual, τ=forget_rate
        """
        if epoch is not None:
            self.epoch = epoch
        progress = min(1.0, self.epoch / max(1, self.num_gradual))
        return 1.0 - progress * self.forget_rate

    @torch.no_grad()
    def select_clean_samples(
        self,
        losses_a: torch.Tensor,
        losses_b: torch.Tensor,
        epoch: int,
    ) -> tuple:
        """
        Select samples with smallest loss for each network.

        Network A selects clean samples → feeds to Network B
        Network B selects clean samples → feeds to Network A

        Args:
            losses_a: (B,) per-sample losses from model A
            losses_b: (B,) per-sample losses from model B
            epoch: Current epoch

        Returns:
            (mask_a2b, mask_b2a): Boolean masks for samples each model
                                   selects for the other
        """
        batch_size = losses_a.size(0)
        remember_rate = self.get_remember_rate(epoch)
        num_remember = max(1, int(batch_size * remember_rate))

        # Model A selects its own small-loss samples → feeds to B
        _, indices_a = torch.topk(losses_a, num_remember, largest=False)
        mask_a2b = torch.zeros(batch_size, dtype=torch.bool, device=losses_a.device)
        mask_a2b[indices_a] = True

        # Model B selects its own small-loss samples → feeds to A
        _, indices_b = torch.topk(losses_b, num_remember, largest=False)
        mask_b2a = torch.zeros(batch_size, dtype=torch.bool, device=losses_b.device)
        mask_b2a[indices_b] = True

        return mask_a2b, mask_b2a


class CoTeachingLoss:
    """
    Wrapper that computes Co-Teaching losses for both models.

    Each model's loss is computed only on samples selected by the OTHER model.
    This prevents confirmation bias — each model can't reinforce its own mistakes.
    """

    def __init__(
        self,
        criterion_a: nn.Module,
        criterion_b: nn.Module,
        selector: CoTeachingSelector,
    ):
        self.criterion_a = criterion_a
        self.criterion_b = criterion_b
        self.selector = selector

    def compute_losses(
        self,
        indices: torch.Tensor,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
        labels: torch.Tensor,
        epoch: int,
    ) -> dict:
        """
        Compute Co-Teaching losses.

        Args:
            indices: (B,) sample indices
            logits_a: (B, C) model A predictions
            logits_b: (B, C) model B predictions
            labels: (B,) ground truth labels
            epoch: Current epoch

        Returns:
            dict with: loss_a, loss_b, mask_a2b, mask_b2a, remember_rate
        """
        # Per-sample losses for both models
        is_soft = (labels.dtype in (torch.float16, torch.float32, torch.float64)
                   and labels.dim() > 1)

        if is_soft:
            with torch.no_grad():
                probs_a = F.softmax(logits_a, dim=1)
                probs_b = F.softmax(logits_b, dim=1)
                losses_a = -(labels * torch.log(probs_a + 1e-8)).sum(dim=1)
                losses_b = -(labels * torch.log(probs_b + 1e-8)).sum(dim=1)
        else:
            losses_a = F.cross_entropy(logits_a, labels, reduction='none')
            losses_b = F.cross_entropy(logits_b, labels, reduction='none')

        # Co-Teaching selection
        mask_a2b, mask_b2a = self.selector.select_clean_samples(
            losses_a, losses_b, epoch
        )

        # Model A trained on samples Model B selected
        if mask_b2a.any():
            loss_a = self._compute_loss(
                self.criterion_a, indices[mask_b2a],
                logits_a[mask_b2a], labels[mask_b2a]
            )
        else:
            loss_a = torch.tensor(0.0, device=logits_a.device)

        # Model B trained on samples Model A selected
        if mask_a2b.any():
            loss_b = self._compute_loss(
                self.criterion_b, indices[mask_a2b],
                logits_b[mask_a2b], labels[mask_a2b]
            )
        else:
            loss_b = torch.tensor(0.0, device=logits_b.device)

        return {
            "loss_a": loss_a,
            "loss_b": loss_b,
            "mask_a2b": mask_a2b,
            "mask_b2a": mask_b2a,
            "remember_rate": self.selector.get_remember_rate(),
            "clean_ratio_a": mask_b2a.float().mean().item(),
            "clean_ratio_b": mask_a2b.float().mean().item(),
        }

    def _compute_loss(self, criterion, indices, logits, labels):
        """Try different calling conventions for the criterion."""
        try:
            return criterion(indices, logits, labels)
        except TypeError:
            try:
                return criterion(logits, labels)
            except TypeError:
                return F.cross_entropy(logits, labels)
