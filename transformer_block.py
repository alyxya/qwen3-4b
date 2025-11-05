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
from rmsnorm import RMSNorm


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
        max_position_embeddings: int,
        rope_theta: float,
        rms_norm_eps: float,
    ) -> None:
        """
        Initialize transformer block

        Args:
            d_model: Model dimension (2560 for Qwen3 4B)
            num_heads: Number of query heads (32 for Qwen3 4B)
            num_kv_heads: Number of key/value heads (8 for Qwen3 4B)
            head_dim: Dimension per head (128 for Qwen3 4B)
            intermediate_size: MLP hidden dimension (9728 for Qwen3 4B)
            max_position_embeddings: Maximum sequence length (262144 for Qwen3 4B)
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
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
        )

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

        attn_output, new_cache_k, new_cache_v = self.self_attn(
            x,
            cache_k=cache_k,
            cache_v=cache_v,
        )

        x = residual + attn_output

        # MLP with residual connection
        residual = x
        x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(x)
        x = residual + mlp_output

        return x, new_cache_k, new_cache_v
