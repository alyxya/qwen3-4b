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
        self, x: torch.Tensor, cache_k: torch.Tensor | None = None, cache_v: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Attention block
        attn_output, new_cache_k, new_cache_v = self.self_attn(
            self.input_layernorm(x), cache_k, cache_v
        )
        x = x + attn_output

        # MLP block
        x = x + self.mlp(self.post_attention_layernorm(x))

        return x, new_cache_k, new_cache_v
