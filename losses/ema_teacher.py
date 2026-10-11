"""
EMA (Exponential Moving Average) Teacher Model for Self-Distillation.

Reference:
    Tarvainen & Valpola, "Mean teachers are better role models", NeurIPS 2017.
    Grill et al., "Bootstrap Your Own Latent", NeurIPS 2020 (BYOL momentum encoder).

The EMA teacher maintains an exponential moving average of the student model's
parameters and provides stable pseudo-labels for distillation. This is particularly
effective for noisy-label training because:
1. EMA averaging acts as a denoising filter on model predictions
2. Teacher provides consistent targets even when student fluctuates
3. Prevents confirmation bias (teacher doesn't train on its own predictions)

Update rule: θ_teacher = α * θ_teacher + (1 - α) * θ_student
where α is the EMA momentum (typically 0.999-0.9999).
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class EMATeacher:
    """
    EMA teacher model wrapper.

    Maintains shadow copies of student model parameters, updated via EMA.
    Used for generating distillation targets and pseudo-labels.

    Args:
        model: Student model (nn.Module)
        alpha: EMA momentum (0.999 default). Higher = slower adaptation = more stable
        warmup_steps: Number of steps before starting EMA updates (allows student to warm up)
        use_dynamic_alpha: If True, increase alpha over time (more stable later)
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.999,
        warmup_steps: int = 100,
        use_dynamic_alpha: bool = True,
    ):
        self.model = model
        self.alpha = alpha
        self.base_alpha = alpha
        self.warmup_steps = warmup_steps
        self.use_dynamic_alpha = use_dynamic_alpha
        self.step_count = 0

        # Create shadow model (same architecture, EMA parameters)
        self.shadow = copy.deepcopy(model)
        self.shadow.requires_grad_(False)
        self.shadow.eval()

        # Freeze shadow parameters
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, step: int = None):
        """
        Update EMA teacher parameters.

        Args:
            step: Current training step (for dynamic alpha)
        """
        if step is not None:
            self.step_count = step

        # Dynamic alpha: increase over time for more stable teacher
        if self.use_dynamic_alpha:
            # Linearly increase from 0.99 to 0.9999
            progress = min(1.0, self.step_count / max(1, self.warmup_steps * 10))
            self.alpha = self.base_alpha + (0.9999 - self.base_alpha) * progress

        for shadow_param, student_param in zip(
            self.shadow.parameters(), self.model.parameters()
        ):
            shadow_param.data.mul_(self.alpha).add_(
                student_param.data, alpha=1.0 - self.alpha
            )

    def step(self, step_count: int):
        """Increment step counter and update."""
        self.step_count = step_count
        self.update(step_count)

    @torch.no_grad()
    def generate_logits(self, images: torch.Tensor) -> torch.Tensor:
        """
        Generate teacher logits for a batch of images.

        Args:
            images: (B, 3, H, W) input images

        Returns:
            teacher_logits: (B, num_classes) teacher predictions
        """
        was_training = self.shadow.training
        self.shadow.eval()
        logits = self.shadow(images)
        if was_training:
            self.shadow.train()
        return logits.detach()

    @torch.no_grad()
    def generate_soft_labels(
        self, images: torch.Tensor, temperature: float = 4.0
    ) -> torch.Tensor:
        """
        Generate soft labels via temperature-scaled softmax.

        Args:
            images: (B, 3, H, W) input images
            temperature: Softmax temperature (higher = softer)

        Returns:
            soft_labels: (B, num_classes) probability distribution
        """
        logits = self.generate_logits(images)
        return F.softmax(logits / temperature, dim=1)

    @torch.no_grad()
    def generate_features(self, images: torch.Tensor) -> torch.Tensor:
        """Generate feature representations for contrastive learning."""
        self.shadow.eval()
        features = self.shadow.get_visual_features(images)
        return features.detach()

    def state_dict(self) -> dict:
        """Get teacher state for checkpointing."""
        return {
            "shadow_state_dict": self.shadow.state_dict(),
            "alpha": self.alpha,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state_dict: dict):
        """Load teacher state from checkpoint."""
        self.shadow.load_state_dict(state_dict["shadow_state_dict"])
        self.alpha = state_dict.get("alpha", self.base_alpha)
        self.step_count = state_dict.get("step_count", 0)

    def to(self, device: torch.device):
        """Move teacher to device."""
        self.shadow = self.shadow.to(device)
        return self

    def train(self):
        """Put shadow in train mode (but gradients still disabled)."""
        self.shadow.train()

    def eval(self):
        """Put shadow in eval mode."""
        self.shadow.eval()
