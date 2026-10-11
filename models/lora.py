"""
LoRA (Low-Rank Adaptation) module for efficient fine-tuning.

Applies low-rank decomposition to linear layers: W' = W + (alpha/r) * B @ A
where A ∈ R^{r×in}, B ∈ R^{out×r}, and r << min(in, out).

This drastically reduces trainable parameters and memory usage while
preserving the pre-trained knowledge in the original weights.
"""

import math
from typing import Optional, List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation wrapper for nn.Linear layers.

    The forward pass computes: y = Wx + (alpha / r) * (B @ A) @ x
    where W is frozen and A, B are trainable low-rank matrices.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        r: LoRA rank (typically 4-64)
        alpha: Scaling factor (typically 1-32)
        dropout: Dropout rate applied to the LoRA branch
        bias: Whether the original linear layer has bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0

        # Original (frozen) linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA low-rank matrices
        if r > 0:
            # A: r × in_features, initialized with Kaiming uniform
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            # B: out_features × r, initialized to zero
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))

            # Initialize A with Kaiming, B with zeros (so LoRA is identity at start)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze original weights
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    @property
    def weight(self):
        """Expose weight for direct access (compatibility with attention modules)."""
        return self.linear.weight

    @weight.setter
    def weight(self, value):
        self.linear.weight = value

    @property
    def bias(self):
        """Expose bias for direct access."""
        return self.linear.bias

    @bias.setter
    def bias(self, value):
        self.linear.bias = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen path
        result = self.linear(x)

        # LoRA path
        if self.lora_A is not None and self.lora_B is not None:
            # (x @ A^T) @ B^T = x @ (B@A)^T
            lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
            result = result + self.scaling * lora_out

        return result

    def merge_weights(self) -> None:
        """Merge LoRA weights into the original linear layer (for inference)."""
        if self.lora_A is not None and self.lora_B is not None:
            delta_w = self.scaling * (self.lora_B @ self.lora_A)
            self.linear.weight.data += delta_w
            # Zero out LoRA params after merging
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()

    def unmerge_weights(self) -> None:
        """Reverse the merge operation."""
        if self.lora_A is not None and self.lora_B is not None:
            delta_w = self.scaling * (self.lora_B @ self.lora_A)
            self.linear.weight.data -= delta_w


def apply_lora_to_clip(
    model: nn.Module,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Apply LoRA to target linear layers in a CLIP ViT model.

    This searches through the model's named modules and replaces
    specified linear layers with LoRALinear wrappers.

    Args:
        model: CLIP model (visual encoder)
        r: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout rate
        target_modules: List of module name substrings to target
                       (e.g., ["q_proj", "v_proj", "k_proj", "out_proj"])

    Returns:
        Model with LoRA applied (same model, modified in-place)
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "c_fc", "c_proj", "fc1", "fc2"]

    lora_count = 0

    def _apply(parent_module: nn.Module, prefix: str = ""):
        nonlocal lora_count
        for name, child in parent_module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear):
                # Check if this linear layer should get LoRA
                should_apply = any(t in name.lower() for t in target_modules)
                if should_apply and child.in_features > 0 and child.out_features > 0:
                    lora_layer = LoRALinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        r=r,
                        alpha=alpha,
                        dropout=dropout,
                        bias=child.bias is not None,
                    )
                    # Copy frozen weights
                    lora_layer.linear.weight.data.copy_(child.weight.data)
                    if child.bias is not None:
                        lora_layer.linear.bias.data.copy_(child.bias.data)

                    setattr(parent_module, name, lora_layer)
                    lora_count += 1
            else:
                _apply(child, full_name)

    _apply(model)
    print(f"[LoRA] Applied LoRA to {lora_count} layers (r={r}, alpha={alpha})")
    return model


def count_lora_params(model: nn.Module) -> tuple:
    """Count trainable and total parameters in the model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
