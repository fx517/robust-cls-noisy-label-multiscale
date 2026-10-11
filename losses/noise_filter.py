"""
GMM-based Clean/Noisy Sample Selection (DivideMix-style).

Reference:
    Li et al., "DivideMix: Learning with Noisy Labels as Semi-supervised Learning",
    ICLR 2020. https://arxiv.org/abs/2002.07394

Core idea:
    Fit a 2-component Gaussian Mixture Model (GMM) on per-sample losses.
    The lower-mean component corresponds to clean samples (small loss),
    while the higher-mean component corresponds to noisy samples (large loss).
    This dynamically separates clean and noisy subsets during training.

Improvements over vanilla DivideMix:
    - Uses both ELR target confidence AND per-sample loss for better separation
    - Momentum-based GMM fitting for stability
    - Confidence threshold as additional filter
"""

import torch
from sklearn.mixture import GaussianMixture


class GMMSampleSelector:
    """
    Dynamic sample selection using 2-component GMM on per-sample losses.

    Separates training samples into clean (likely correct label) and noisy
    (likely incorrect label) subsets after each epoch.

    Args:
        num_samples: Total number of training samples
        momentum: EMA momentum for loss history (smoothing)
        warmup_epochs: Epochs before starting GMM selection (train all as clean)
        p_threshold: Posterior threshold for clean sample classification
        use_confidence: Also use ELR target confidence for separation
    """

    def __init__(
        self,
        num_samples: int,
        momentum: float = 0.9,
        warmup_epochs: int = 10,
        p_threshold: float = 0.5,
        use_confidence: bool = True,
    ):
        self.num_samples = num_samples
        self.momentum = momentum
        self.warmup_epochs = warmup_epochs
        self.p_threshold = p_threshold
        self.use_confidence = use_confidence

        # Per-sample loss history (EMA)
        self.register_or_init_loss_history()

        # Current clean/noisy masks
        self.clean_mask = torch.ones(num_samples, dtype=torch.bool)
        self.noisy_mask = torch.zeros(num_samples, dtype=torch.bool)

        # Statistics
        self.gmm_means = [0.0, 0.0]
        self.clean_ratio = 1.0
        self.epoch = 0

    def register_or_init_loss_history(self):
        """Initialize per-sample loss EMA history."""
        self.loss_history = torch.zeros(self.num_samples)
        self.loss_count = torch.zeros(self.num_samples)

    @torch.no_grad()
    def update(
        self,
        indices: torch.Tensor,
        losses: torch.Tensor,
        elr_targets: torch.Tensor = None,
        labels: torch.Tensor = None,
        epoch: int = None,
    ) -> tuple:
        """
        Update sample selection using GMM on loss distribution.

        Args:
            indices: (B,) sample indices for this batch
            losses: (B,) per-sample losses
            elr_targets: (B, C) or None — ELR EMA targets for confidence filtering
            labels: (B,) or None — ground-truth labels
            epoch: Current epoch number

        Returns:
            (clean_mask, noisy_mask): Updated boolean masks for all samples
        """
        if epoch is not None:
            self.epoch = epoch

        # Update loss history with EMA
        idx = indices.cpu()
        loss_cpu = losses.detach().cpu()
        self.loss_history[idx] = (
            self.momentum * self.loss_history[idx]
            + (1 - self.momentum) * loss_cpu
        )
        self.loss_count[idx] += 1

        # NOTE: GMM fitting is expensive — only done in full_update() per epoch
        return self.clean_mask, self.noisy_mask

    def _fit_gmm(self):
        """Fit 2-component GMM on per-sample loss history."""
        # Use only samples that have been seen
        seen_mask = self.loss_count > 0
        if seen_mask.sum() < 100:
            return

        losses = self.loss_history[seen_mask].numpy().reshape(-1, 1)

        # Normalize for better GMM fitting
        loss_mean = losses.mean()
        loss_std = losses.std() + 1e-8
        losses_norm = (losses - loss_mean) / loss_std

        try:
            gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-3,
                                 covariance_type='full', random_state=42,
                                 n_init=3)  # Multiple inits for stability
            gmm.fit(losses_norm)
            probs = gmm.predict_proba(losses_norm)

            # Identify clean component (lower mean)
            means = gmm.means_.flatten()
            clean_comp = 0 if means[0] < means[1] else 1

            self.gmm_means = [float(means[0]), float(means[1])]

            # Classify each seen sample
            clean_seen = probs[:, clean_comp] >= self.p_threshold
            self.clean_mask[seen_mask.clone()] = torch.from_numpy(clean_seen)
            self.noisy_mask[seen_mask.clone()] = torch.from_numpy(~clean_seen)

            # Unseen samples default to clean
            self.clean_mask[~seen_mask] = True
            self.noisy_mask[~seen_mask] = False

            self.clean_ratio = self.clean_mask.float().mean().item()

        except Exception:
            # GMM fitting failed - keep previous masks
            pass

    def _confidence_filter(
        self, elr_targets: torch.Tensor, labels: torch.Tensor
    ):
        """
        Additional confidence-based filtering using ELR targets.

        Even if GMM considers a sample "clean" by loss, it might still be
        noisy if the ELR target's argmax disagrees with the given label.
        """
        batch_size = elr_targets.shape[0]
        if batch_size == 0:
            return

        # Check if ELR prediction agrees with given label
        elr_preds = elr_targets.argmax(dim=1)
        elr_confidence = elr_targets.max(dim=1).values
        agreement = (elr_preds == labels) | (elr_confidence < 0.3)

        # Only update for the batch indices
        # (full update requires all samples, done in full_update)

    @torch.no_grad()
    def full_update(
        self,
        elr_targets: torch.Tensor,
        labels: torch.Tensor,
        epoch: int,
    ):
        """
        Full epoch-level update with all available information.

        Args:
            elr_targets: (N, C) full ELR target matrix
            labels: (N,) ground-truth labels
            epoch: Current epoch
        """
        self.epoch = epoch

        if epoch < self.warmup_epochs:
            self.clean_mask[:] = True
            self.noisy_mask[:] = False
            return

        # Compute per-sample CE loss from ELR targets to given labels
        # (as proxy for "how much does the model disagree with the given label?")
        probs = torch.clamp(elr_targets, 1e-7, 1.0 - 1e-7)
        elr_preds = probs.argmax(dim=1)
        elr_confidence = probs.max(dim=1).values

        # Loss proxy: disagreement between ELR target and given label
        for i in range(len(labels)):
            if self.loss_count[i] == 0:
                # First time seeing this sample
                self.loss_history[i] = 1.0 - probs[i, labels[i]].item()
                self.loss_count[i] = 1

        # Fit GMM
        self._fit_gmm()

        # Cross-validate with prediction agreement
        agreement = (elr_preds == labels)
        high_confidence = elr_confidence > 0.5

        # A sample is clean if: GMM says clean AND (ELR agrees OR low confidence)
        clean_by_gmm = self.clean_mask.clone()
        clean_by_agreement = agreement | (~high_confidence)

        # Weighted combination
        self.clean_mask = clean_by_gmm & clean_by_agreement
        self.noisy_mask = ~self.clean_mask
        self.clean_ratio = self.clean_mask.float().mean().item()

    def get_probs(self) -> dict:
        """Get selection statistics."""
        return {
            "clean_ratio": self.clean_ratio,
            "num_clean": self.clean_mask.sum().item(),
            "num_noisy": self.noisy_mask.sum().item(),
            "gmm_mean_clean": self.gmm_means[0],
            "gmm_mean_noisy": self.gmm_means[1],
        }

    def to(self, device):
        """Move tensors to device (loss_history stays on CPU for sklearn)."""
        self.clean_mask = self.clean_mask.to(device)
        self.noisy_mask = self.noisy_mask.to(device)
        return self


class CoRefinementModule:
    """
    Label co-refinement: progressively correct noisy labels using model predictions.

    For samples classified as "noisy", blend the original label with the model's
    current prediction. The blending ratio increases over training as the model
    becomes more confident.

    Args:
        num_classes: Number of classes
        refine_alpha: EMA weight for label refinement (how much to trust model)
        warmup_epochs: Epochs before starting refinement
        max_alpha: Maximum refinement weight
    """

    def __init__(
        self,
        num_classes: int,
        refine_alpha: float = 0.5,
        warmup_epochs: int = 10,
        max_alpha: float = 0.9,
    ):
        self.num_classes = num_classes
        self.refine_alpha = refine_alpha
        self.warmup_epochs = warmup_epochs
        self.max_alpha = max_alpha

    def refine_labels(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        mask: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """
        Refine labels for masked (noisy) samples.

        Args:
            labels: (B,) original hard labels
            predictions: (B, C) model prediction probabilities
            mask: (B,) boolean mask of samples to refine
            epoch: Current epoch

        Returns:
            refined_labels: (B, C) soft labels (one-hot for clean, blended for noisy)
        """
        batch_size = labels.size(0)

        # Progress-based alpha
        progress = max(0.0, min(1.0, (epoch - self.warmup_epochs) / 50.0))
        alpha = self.refine_alpha + (self.max_alpha - self.refine_alpha) * progress

        # One-hot original labels
        labels_onehot = torch.zeros(
            batch_size, self.num_classes, device=labels.device
        )
        labels_onehot.scatter_(1, labels.unsqueeze(1).clamp(min=0), 1.0)

        # Blend for noisy samples
        refined = labels_onehot.clone()
        if mask.any():
            noisy_labels = labels_onehot[mask]
            noisy_preds = predictions[mask]
            refined[mask] = (1 - alpha) * noisy_labels + alpha * noisy_preds

        return refined
