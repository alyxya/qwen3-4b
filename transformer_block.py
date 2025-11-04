"""
Transformer Block for Qwen3 4B

A single transformer layer consists of:
1. Input RMSNorm
2. Self-attention with Q/K norm and residual connection
3. Post-attention RMSNorm
4. MLP with residual connection
"""

import torch
import torch.nn as nn
from attention import Attention
from mlp import MLP


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    RMSNorm is simpler than LayerNorm - it only normalizes by RMS (no mean centering),
    and only has a scale parameter (no bias).
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        """
        Initialize RMSNorm

        Args:
            d_model: Model dimension (2560 for Qwen3 4B)
            eps: Small constant for numerical stability (1e-6 for Qwen3 4B)
        """
        super().__init__()
        self.eps = eps
        # Scale parameter (learnable)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm

        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)

        Returns:
            Normalized tensor, same shape as input
        """
        # Compute RMS: sqrt(mean(x^2))
        # x: (batch, seq_len, d_model)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        # x / rms: (batch, seq_len, d_model)
        # self.weight: (d_model,) broadcasts to (batch, seq_len, d_model)
        return self.weight * (x / rms)


class TransformerBlock(nn.Module):
    """
    Single transformer block with self-attention and MLP

    Architecture:
    x → RMSNorm → Attention (with Q/K norm) → + (residual) →
    → RMSNorm → MLP → + (residual) → output
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rope_theta: float = 5000000.0,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        """
        Initialize transformer block

        Args:
            d_model: Model dimension (2560 for Qwen3 4B)
            num_heads: Number of query heads (32 for Qwen3 4B)
            num_kv_heads: Number of key/value heads (8 for Qwen3 4B)
            head_dim: Dimension per head (128 for Qwen3 4B)
            intermediate_size: MLP hidden dimension (9728 for Qwen3 4B)
            rope_theta: RoPE base frequency (5000000 for Qwen3 4B)
            rms_norm_eps: RMSNorm epsilon (1e-6 for Qwen3 4B)
        """
        super().__init__()

        # Pre-attention norm
        self.input_layernorm = RMSNorm(d_model, eps=rms_norm_eps)

        # Self-attention
        self.self_attn = Attention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
        )

        # Q and K norms (applied after projection, per head)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        # Post-attention norm
        self.post_attention_layernorm = RMSNorm(d_model, eps=rms_norm_eps)

        # MLP
        self.mlp = MLP(d_model=d_model, intermediate_size=intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        cache_k: torch.Tensor | None = None,
        cache_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer block

        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)
            cache_k: Cached key tensor or None
            cache_v: Cached value tensor or None

        Returns:
            Tuple of (output, new_cache_k, new_cache_v)
            - output: shape (batch_size, seq_len, d_model)
            - new_cache_k: Updated key cache
            - new_cache_v: Updated value cache
        """
        # Self-attention with residual connection
        residual = x
        x = self.input_layernorm(x)

        # TODO: Apply q_norm and k_norm inside attention
        # For now, attention doesn't use them - we'll add this when loading weights
        attn_output, new_cache_k, new_cache_v = self.self_attn(x, cache_k=cache_k, cache_v=cache_v)

        x = residual + attn_output

        # MLP with residual connection
        residual = x
        x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(x)
        x = residual + mlp_output

        return x, new_cache_k, new_cache_v
