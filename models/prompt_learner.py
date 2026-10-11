"""
Learnable Text Prompts for CLIP (CoOp-style).

Reference:
    Zhou et al., "Learning to Prompt for Vision-Language Models", IJCV 2022.
    https://arxiv.org/abs/2109.01134

Core idea: instead of hand-crafted prompts like "a photo of a {class}",
learn a set of continuous prompt vectors that are prepended to the class token.
This transforms CLIP's text encoder into a trainable classifier that provides
complementary signals to the visual branch.

Architecture:
    [V1][V2]...[VM][CLASS]  →  Text Encoder  →  Text Features (D,)
    └── learnable ──┘

where V1...VM are M learnable context vectors, CLASS is the class name token.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptLearner(nn.Module):
    """
    CoOp-style learnable text prompt learner for CLIP.

    Learns continuous prompt vectors that, combined with class name tokens,
    produce class-specific text embeddings for classification.

    Args:
        clip_model: OpenCLIP model instance (for tokenizer and text encoder)
        class_names: List of class name strings (e.g., ["airplane_001", ...])
        n_ctx: Number of learnable context tokens (default: 16)
        ctx_init: Initial context words (e.g., "a photo of a") or None for random
        class_token_position: Where to place class token ("end" or "middle")
        device: Device for prompt vectors
    """

    def __init__(
        self,
        clip_model,
        class_names: list,
        n_ctx: int = 16,
        ctx_init: str = None,
        class_token_position: str = "end",
        device: torch.device = None,
    ):
        super().__init__()

        self.n_ctx = n_ctx
        self.class_token_position = class_token_position
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get CLIP's token embedding dimension
        if hasattr(clip_model, 'token_embedding'):
            self.embed_dim = clip_model.token_embedding.embedding_dim
        elif hasattr(clip_model, 'positional_embedding'):
            self.embed_dim = clip_model.positional_embedding.shape[-1]
        else:
            # Default for ViT-B/32
            self.embed_dim = 512

        # Get tokenizer and convert class names to token IDs
        self.tokenizer = self._get_tokenizer(clip_model)
        self.class_names = class_names
        self.n_cls = len(class_names)

        # Tokenize class names (just the class name token without context)
        classname_tokens = self._tokenize_classnames(class_names)
        self.register_buffer("classname_tokens", classname_tokens)

        # Initialize learnable context vectors
        if ctx_init is not None:
            # Initialize from given prompt words
            ctx_init_tokens = self._tokenize_prompt(ctx_init)
            with torch.no_grad():
                init_vectors = clip_model.token_embedding(ctx_init_tokens.to(self.device))
            init_vectors = init_vectors.mean(dim=0, keepdim=True)  # Average over context words
            init_vectors = init_vectors.repeat(n_ctx, 1)  # (n_ctx, embed_dim)
            self.ctx = nn.Parameter(init_vectors)
        else:
            # Random initialization
            self.ctx = nn.Parameter(torch.empty(n_ctx, self.embed_dim))
            nn.init.normal_(self.ctx, std=0.02)

        # Store the token embedding for reference (frozen)
        self._token_embedding = clip_model.token_embedding

    def _get_tokenizer(self, clip_model):
        """Extract tokenizer from CLIP model."""
        import open_clip
        if hasattr(clip_model, 'tokenize'):
            return clip_model.tokenize
        return open_clip.get_tokenizer("ViT-B-32")

    def _tokenize_classnames(self, class_names: list) -> torch.Tensor:
        """Tokenize class names to get their token IDs."""
        # Use the tokenizer's internal method or direct call
        import open_clip
        if hasattr(open_clip, 'tokenize'):
            tokens = open_clip.tokenize(class_names)
        else:
            # Simple fallback: use clip_model's tokenizer
            tokens = self.tokenizer(class_names)
        return tokens  # (n_cls, max_len)

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        """Tokenize a prompt string."""
        import open_clip
        if hasattr(open_clip, 'tokenize'):
            tokens = open_clip.tokenize([prompt])
        else:
            tokens = self.tokenizer([prompt])
        return tokens

    def forward(self) -> torch.Tensor:
        """
        Construct text embeddings for all classes using learned prompts.

        Returns:
            text_features: (n_cls, embed_dim) normalized text features
        """
        # Get class name token embeddings
        class_embeddings = self._token_embedding(
            self.classname_tokens.to(self.ctx.device)
        )  # (n_cls, max_len, embed_dim)

        # Get learned context
        ctx = self.ctx.unsqueeze(0)  # (1, n_ctx, embed_dim)

        # Build prompt sequence for each class
        if self.class_token_position == "end":
            # [CTX][CTX]...[CTX][CLASS]
            prefix = ctx.expand(self.n_cls, -1, -1)  # (n_cls, n_ctx, embed_dim)
            prompts = torch.cat([prefix, class_embeddings], dim=1)

        elif self.class_token_position == "middle":
            # [CTX]...[CTX][CLASS][CTX]...[CTX]
            half = self.n_ctx // 2
            prefix = ctx[:, :half, :].expand(self.n_cls, -1, -1)
            suffix = ctx[:, half:, :].expand(self.n_cls, -1, -1)
            prompts = torch.cat([prefix, class_embeddings, suffix], dim=1)
        else:
            raise ValueError(f"Unknown class_token_position: {self.class_token_position}")

        # prompts shape: (n_cls, n_ctx + max_token_len, embed_dim)
        # For text encoder, we pass this through the transformer
        # Since we have raw embeddings, we need to pass them through the text transformer
        # directly (skipping token embedding since we already have embeddings)

        return prompts  # Return raw prompt embeddings

    def encode_text(self, text_encoder, prompts: torch.Tensor = None) -> torch.Tensor:
        """
        Encode prompt embeddings using CLIP's text transformer.

        Args:
            text_encoder: CLIP's text transformer (nn.Module)
            prompts: Optional pre-computed prompts. If None, calls forward().

        Returns:
            text_features: (n_cls, embed_dim) normalized text features
        """
        if prompts is None:
            prompts = self.forward()

        n_cls, seq_len, embed_dim = prompts.shape

        # Use CLIP's text transformer to encode the prompt embeddings
        # The text encoder expects token embeddings; we directly provide them

        # We need to handle this carefully - open_clip's text encoder
        # typically expects token IDs. We need to use the transformer directly.

        # Most CLIP implementations have:
        # - token_embedding
        # - positional_embedding
        # - transformer
        # - ln_final

        device = prompts.device

        # Handle different CLIP model structures
        if hasattr(text_encoder, 'positional_embedding'):
            x = prompts + text_encoder.positional_embedding[:seq_len].to(device)
        else:
            x = prompts

        # Apply causal mask for autoregressive transformer
        if hasattr(text_encoder, 'transformer'):
            # Standard CLIP text transformer
            attn_mask = self._build_causal_mask(seq_len, device)
            x = x.permute(1, 0, 2)  # (seq_len, n_cls, embed_dim)

            for block in text_encoder.transformer.resblocks:
                x = block(x, attn_mask=attn_mask)

            x = x.permute(1, 0, 2)  # (n_cls, seq_len, embed_dim)

            # Take the feature at the EOT (end-of-text) position
            x = text_encoder.ln_final(x)
            eot_indices = self._get_eot_indices(n_cls).to(device)
            text_features = x[torch.arange(n_cls, device=device), eot_indices]

            return F.normalize(text_features, dim=-1)

        # Fallback: try using the full encode_text pipeline
        return self._encode_via_tokenizer(text_encoder)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build causal attention mask for autoregressive transformer."""
        mask = torch.empty(seq_len, seq_len, device=device)
        mask.fill_(float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        return mask

    def _get_eot_indices(self, n_cls: int) -> torch.Tensor:
        """Get indices of the last non-padding token (EOT token position)."""
        return torch.full((n_cls,), self.classname_tokens.argmax(dim=-1).mode().values.item(),
                          dtype=torch.long)

    def _encode_via_tokenizer(self, text_encoder) -> torch.Tensor:
        """Fallback: use token-based encoding."""
        # This bypasses the learned prompts — used as fallback
        raise NotImplementedError(
            "Cannot encode via tokenizer with learned prompts. "
            "Use encode_text() with proper text transformer."
        )


class UnifiedPromptLearner(nn.Module):
    """
    Lightweight prompt learner that combines learnable prompts with a
    projection head for direct classification.

    Simpler than full PromptLearner — generates one text prototype per class
    that can be compared with visual features via cosine similarity.
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 512,
        n_ctx: int = 8,
        use_prompts: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.use_prompts = use_prompts

        if use_prompts:
            # Learnable class prototypes (initialized randomly)
            self.prompt_vectors = nn.Parameter(
                torch.empty(num_classes, feature_dim)
            )
            nn.init.normal_(self.prompt_vectors, std=0.02)

            # Learnable context for rich prompt representation
            self.context = nn.Parameter(
                torch.empty(n_ctx, feature_dim)
            )
            nn.init.normal_(self.context, std=0.02)

            # Lightweight fusion: context-aware prototype computation
            self.prototype_proj = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
            )
        else:
            # Simple learnable prototypes
            self.prototypes = nn.Parameter(
                torch.empty(num_classes, feature_dim)
            )
            nn.init.normal_(self.prototypes, std=0.02)

    def forward(self) -> torch.Tensor:
        """
        Generate class prototype features.

        Returns:
            prototypes: (num_classes, feature_dim) normalized class prototypes
                         for cosine similarity classification
        """
        if self.use_prompts:
            # Fuse learned prompt vectors with context
            ctx_mean = self.context.mean(dim=0, keepdim=True)  # (1, D)
            ctx_expanded = ctx_mean.expand(self.num_classes, -1)  # (C, D)
            fused = self.prototype_proj(
                torch.cat([self.prompt_vectors, ctx_expanded], dim=-1)
            )
            return F.normalize(fused, dim=-1)
        else:
            return F.normalize(self.prototypes, dim=-1)

    def classify(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Classify visual features using cosine similarity with prototypes.

        Args:
            visual_features: (B, D) normalized visual features

        Returns:
            logits: (B, num_classes) cosine similarity scores
        """
        prototypes = self.forward()  # (C, D)
        visual_features = F.normalize(visual_features, dim=-1)  # (B, D)

        # Cosine similarity with temperature scaling
        logits = visual_features @ prototypes.T  # (B, C)
        return logits * 10.0  # Temperature scaling (learnable in CLIP)
