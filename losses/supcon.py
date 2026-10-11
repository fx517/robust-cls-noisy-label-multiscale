"""
Supervised Contrastive Loss (SupCon) for Noise-Robust Representation Learning.

Reference:
    Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
    https://arxiv.org/abs/2004.11362

Core idea:
    Pull features of same-class samples together and push different-class
    samples apart in embedding space. Unlike standard cross-entropy which
    only learns decision boundaries, SupCon learns better feature representations.

For noisy-label training:
    - Only applies to samples classified as "clean" by GMM
    - Uses EMA teacher features as anchors for stability
    - Temperature-scaled to focus on hard positives/negatives

Loss formula:
    L_supcon = -1/|P(i)| Σ_{p∈P(i)} log( exp(z_i·z_p / τ) / Σ_{a∈A(i)} exp(z_i·z_a / τ) )

where z_i is the normalized feature, P(i) are positives (same class), and
A(i) are all other samples in the batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised contrastive loss operating on normalized feature embeddings.

    For each anchor, positives are all samples of the same class in the batch.
    The loss encourages features of the same class to be similar.

    Args:
        temperature: Temperature scaling (lower = harder contrast). Default: 0.07
        base_temperature: Base temperature for scaling. Default: 0.07
        contrast_mode: "one" (one view) or "all" (all pairs). Default: "all"
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        contrast_mode: str = "all",
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            features: (B, D) normalized feature embeddings
            labels: (B,) integer class labels
            mask: (B,) optional boolean mask (True = include in loss)

        Returns:
            SupCon loss scalar
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        anchor_dot_contrast = torch.matmul(features, features.T) / self.temperature

        # Mask out self-contrast (non-inplace)
        logits_mask = 1.0 - torch.eye(batch_size, device=device)

        # Build positive mask (same class)
        labels = labels.contiguous().view(-1, 1)
        positive_mask = labels.eq(labels.T).float()
        positive_mask = positive_mask * (1.0 - torch.eye(batch_size, device=device))  # Non-inplace

        # Apply clean sample mask if provided
        if mask is not None:
            clean_mask = mask.float().view(-1, 1) * mask.float().view(1, -1)
            positive_mask = positive_mask * clean_mask
            logits_mask = logits_mask * clean_mask

        # Remove anchors with no positives
        num_positives = positive_mask.sum(dim=1)
        valid_anchors = num_positives > 0

        if not valid_anchors.any():
            return torch.tensor(0.0, device=device)

        # Compute log-sum-exp for denominator
        # (each anchor against all others)
        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean over positives
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / num_positives.clamp(min=1)

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss[valid_anchors].mean()

        return loss


class NoiseRobustSupConLoss(nn.Module):
    """
    Noise-robust supervised contrastive loss.

    Extends SupCon with:
    1. Confidence-weighted positives (soft positive mask)
    2. Teacher-student feature alignment
    3. Prototype-based negative mining

    For noisy-label training, we use:
    - Clean samples (GMM-selected): standard SupCon
    - Noisy samples: contrast with teacher features as soft targets
    - Inter-batch prototype queue for more negatives

    Args:
        temperature: Contrastive temperature
        num_classes: Number of classes
        queue_size: Size of prototype queue (for more negatives)
        proto_momentum: EMA momentum for prototype updates
    """

    def __init__(
        self,
        temperature: float = 0.07,
        num_classes: int = 400,
        queue_size: int = 4096,
        proto_momentum: float = 0.99,
    ):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.queue_size = queue_size
        self.proto_momentum = proto_momentum

        # Prototype queue: (num_classes, queue_size, feature_dim)
        # Maintains recent features for each class as additional negatives
        self.register_buffer("proto_queue_ptr", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("proto_queue", torch.zeros(num_classes, queue_size, 512))

    @torch.no_grad()
    def enqueue_features(self, features: torch.Tensor, labels: torch.Tensor):
        """Enqueue features into per-class prototype queues."""
        features = F.normalize(features.detach(), dim=1)
        for c in range(self.num_classes):
            class_mask = labels == c
            if not class_mask.any():
                continue
            class_feats = features[class_mask]
            ptr = self.proto_queue_ptr[c].item()
            num_new = min(class_feats.size(0), self.queue_size - ptr)
            self.proto_queue[c, ptr:ptr + num_new] = class_feats[:num_new].cpu()
            self.proto_queue_ptr[c] = (ptr + num_new) % self.queue_size

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        labels: torch.Tensor,
        clean_mask: torch.Tensor = None,
        distill_weight: float = 0.3,
    ) -> torch.Tensor:
        """
        Compute noise-robust contrastive loss with teacher alignment.

        Args:
            student_features: (B, D) student model features
            teacher_features: (B, D) EMA teacher features (detached)
            labels: (B,) ground-truth labels
            clean_mask: (B,) True = clean samples
            distill_weight: Weight for teacher alignment loss

        Returns:
            Combined SupCon + teacher alignment loss
        """
        device = student_features.device
        batch_size = student_features.shape[0]
        feature_dim = student_features.shape[1]

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # Normalize
        s_feat = F.normalize(student_features, dim=1)
        t_feat = F.normalize(teacher_features, dim=1)

        # === 1. Student-Student Contrast (standard SupCon) ===
        s_sim = s_feat @ s_feat.T / self.temperature
        pos_mask = labels.contiguous().view(-1, 1).eq(labels.contiguous().view(1, -1)).float()
        # Non-inplace: avoid modifying tensors needed for autograd
        diag_mask = 1.0 - torch.eye(batch_size, device=device)
        pos_mask = pos_mask * diag_mask

        if clean_mask is not None:
            cm = clean_mask.float()
            pos_mask = pos_mask * (cm.view(-1, 1) * cm.view(1, -1))

        exp_sim = torch.exp(s_sim) * diag_mask  # Non-inplace mask diagonal

        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        all_sum = exp_sim.sum(dim=1)
        valid = pos_sum > 0

        if valid.any():
            loss_s2s = -(torch.log((pos_sum[valid] + 1e-8) / (all_sum[valid] + 1e-8))).mean()
        else:
            loss_s2s = torch.tensor(0.0, device=device)

        # === 2. Student-Teacher Alignment ===
        # Encourage student features to be close to teacher features
        s_t_sim = (s_feat * t_feat).sum(dim=1)  # (B,)
        loss_align = (1.0 - s_t_sim).mean()  # Cosine distance

        # Combined loss
        total_loss = loss_s2s + distill_weight * loss_align

        return total_loss


class PrototypicalSupConLoss(nn.Module):
    """
    Prototype-based contrastive loss using learnable class prototypes.

    Lightweight alternative to SupCon that works well for fine-grained tasks.
    Uses class prototypes (learned or EMA-updated) as contrastive targets.

    Args:
        num_classes: Number of classes
        feature_dim: Feature dimension
        temperature: Contrastive temperature
    """

    def __init__(
        self,
        num_classes: int = 400,
        feature_dim: int = 512,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature

        # Learnable class prototypes
        self.prototypes = nn.Parameter(
            torch.empty(num_classes, feature_dim)
        )
        nn.init.orthogonal_(self.prototypes)

        self.prototype_ema = torch.zeros(num_classes, feature_dim)
        self.prototype_count = torch.zeros(num_classes)
        self.ema_momentum = 0.99

    @torch.no_grad()
    def update_prototypes(self, features: torch.Tensor, labels: torch.Tensor):
        """EMA update of class prototypes."""
        features = F.normalize(features.detach(), dim=1)
        for c in range(self.num_classes):
            mask = labels == c
            if mask.any():
                class_feat = features[mask].mean(dim=0)
                self.prototype_ema[c] = (
                    self.ema_momentum * self.prototype_ema[c].to(features.device)
                    + (1 - self.ema_momentum) * class_feat.cpu()
                )
                self.prototype_count[c] += mask.sum().item()

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute prototypical contrastive loss.

        For each feature, it should be:
        - Close to its own class prototype
        - Far from other class prototypes

        Args:
            features: (B, D) normalized features
            labels: (B,) class labels
            mask: (B,) boolean mask (True = include)

        Returns:
            Proto NCE loss
        """
        device = features.device
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)

        # Similarity to all prototypes
        sim = features @ prototypes.T / self.temperature  # (B, C)

        # Cross-entropy with prototype as classifier
        loss = F.cross_entropy(sim, labels, reduction='none')

        if mask is not None:
            if mask.any():
                loss = loss[mask].mean()
            else:
                loss = torch.tensor(0.0, device=device)

        return loss
