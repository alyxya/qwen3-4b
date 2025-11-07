"""Multi-head Attention with Grouped Query Attention (GQA) and KV caching"""

import torch
import torch.nn as nn
from .rmsnorm import RMSNorm
from .rope import RoPE


class Attention(nn.Module):
    """GQA attention: K/V have fewer heads than Q (8 KV heads, 32 Q heads)"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_queries_per_kv = num_heads // num_kv_heads

        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)
        self.rope = RoPE(head_dim, max_position_embeddings, rope_theta)

    def forward(
        self, x: torch.Tensor, cache_k: torch.Tensor | None = None, cache_v: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # Compute position IDs from cache length
        start_pos = cache_k.shape[2] if cache_k is not None else 0
        position_ids = torch.arange(start_pos, start_pos + seq_len, device=x.device)

        # Project and reshape to heads
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RMSNorm and RoPE
        q = self.rope(self.q_norm(q), position_ids)
        k = self.rope(self.k_norm(k), position_ids)

        # Concatenate with cache
        if cache_k is not None:
            k = torch.cat([cache_k, k], dim=2)
            v = torch.cat([cache_v, v], dim=2)

        # Group Q heads for GQA: (batch, num_kv_heads, queries_per_kv, seq, head_dim)
        q = q.view(batch_size, self.num_kv_heads, self.num_queries_per_kv, seq_len, self.head_dim)

        # Attention scores
        scores = torch.einsum("bghsd,bgkd->bghsk", q, k) / (self.head_dim**0.5)

        # Causal mask (only for prefill)
        if seq_len > 1:
            kv_seq_len = k.size(2)
            mask = torch.triu(
                torch.full((seq_len, kv_seq_len), float("-inf"), device=x.device, dtype=scores.dtype),
                diagonal=kv_seq_len - seq_len + 1
            )
            scores = scores + mask

        # Apply attention and project output
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.einsum("bghsk,bgkd->bghsd", attn_weights, v)
        output = output.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)

        return output, k, v
