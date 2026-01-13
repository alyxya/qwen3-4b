"""Transformer block: RMSNorm → Attention → RMSNorm → MLP (with residuals)"""

import torch
import torch.nn as nn
from .attention import Attention
from .mlp import MLP
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """Single transformer layer with pre-norm architecture"""

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
        super().__init__()
        self.input_layernorm = RMSNorm(d_model, eps=rms_norm_eps)
        self.self_attn = Attention(
            d_model, num_heads, num_kv_heads, head_dim,
            max_position_embeddings, rope_theta, rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(d_model, eps=rms_norm_eps)
        self.mlp = MLP(d_model, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        cache_k: torch.Tensor | None = None,
        cache_v: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through transformer block

        Args:
            x: (batch, seq, d_model) - input hidden states
            cache_k: (batch, num_kv_heads, cache_len, head_dim) - cached keys
            cache_v: (batch, num_kv_heads, cache_len, head_dim) - cached values
            attention_mask: (batch, kv_seq_len) - 1 for keep, 0 for mask
            position_ids: (batch, seq_len) or (seq_len,) - absolute positions for RoPE

        Returns:
            x: (batch, seq, d_model) - output hidden states
            new_cache_k: (batch, num_kv_heads, cache_len + seq, head_dim) - updated keys
            new_cache_v: (batch, num_kv_heads, cache_len + seq, head_dim) - updated values
        """
        # Attention block with residual connection
        attn_output, new_cache_k, new_cache_v = self.self_attn(
            self.input_layernorm(x),  # (batch, seq, d_model)
            cache_k=cache_k,
            cache_v=cache_v,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        x = x + attn_output  # (batch, seq, d_model)

        # MLP block with residual connection
        x = x + self.mlp(self.post_attention_layernorm(x))  # (batch, seq, d_model)

        return x, new_cache_k, new_cache_v
