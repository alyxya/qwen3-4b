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
        """Forward pass through attention with optional KV caching

        Args:
            x: (batch, seq, d_model) - input hidden states
            cache_k: (batch, num_kv_heads, cache_len, head_dim) - cached keys
            cache_v: (batch, num_kv_heads, cache_len, head_dim) - cached values

        Returns:
            output: (batch, seq, d_model) - attention output
            k: (batch, num_kv_heads, cache_len + seq, head_dim) - updated keys
            v: (batch, num_kv_heads, cache_len + seq, head_dim) - updated values
        """
        batch_size, seq_len, _ = x.shape  # (batch, seq, d_model)

        # Compute position IDs from cache length
        start_pos = cache_k.shape[2] if cache_k is not None else 0
        position_ids = torch.arange(start_pos, start_pos + seq_len, device=x.device)  # (seq,)

        # Project, reshape, normalize, then transpose (matches HuggingFace order)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        # q: (batch, seq, num_heads * head_dim) -> (batch, seq, num_heads, head_dim)
        q = self.q_norm(q).transpose(1, 2)
        # q: (batch, seq, num_heads, head_dim) -> (batch, num_heads, seq, head_dim)

        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        # k: (batch, seq, num_kv_heads * head_dim) -> (batch, seq, num_kv_heads, head_dim)
        k = self.k_norm(k).transpose(1, 2)
        # k: (batch, seq, num_kv_heads, head_dim) -> (batch, num_kv_heads, seq, head_dim)

        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # v: (batch, seq, num_kv_heads * head_dim) -> (batch, seq, num_kv_heads, head_dim) -> (batch, num_kv_heads, seq, head_dim)

        # Apply RoPE
        q = self.rope(q, position_ids)  # (batch, num_heads, seq, head_dim)
        k = self.rope(k, position_ids)  # (batch, num_kv_heads, seq, head_dim)

        # Concatenate with cache
        if cache_k is not None:
            k = torch.cat([cache_k, k], dim=2)  # (batch, num_kv_heads, cache_len + seq, head_dim)
            v = torch.cat([cache_v, v], dim=2)  # (batch, num_kv_heads, cache_len + seq, head_dim)

        # Group Q heads for GQA
        q = q.view(batch_size, self.num_kv_heads, self.num_queries_per_kv, seq_len, self.head_dim)
        # (batch, num_heads, seq, head_dim) -> (batch, num_kv_heads, queries_per_kv, seq, head_dim)

        # Attention scores using einsum
        scores = torch.einsum("bghsd,bgkd->bghsk", q, k) / (self.head_dim**0.5)
        # (batch, num_kv_heads, queries_per_kv, seq, head_dim) x (batch, num_kv_heads, kv_seq, head_dim)
        # -> (batch, num_kv_heads, queries_per_kv, seq, kv_seq)

        # Causal mask (only for prefill when seq > 1)
        if seq_len > 1:
            kv_seq_len = k.size(2)  # cache_len + seq
            mask = torch.triu(
                torch.full((seq_len, kv_seq_len), float("-inf"), device=x.device, dtype=scores.dtype),
                diagonal=kv_seq_len - seq_len + 1
            )  # (seq, kv_seq)
            scores = scores + mask  # (batch, num_kv_heads, queries_per_kv, seq, kv_seq)

        # Softmax in float32 for numerical stability (prevents overflow/underflow in exp)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        # (batch, num_kv_heads, queries_per_kv, seq, kv_seq)

        # Apply attention using einsum
        output = torch.einsum("bghsk,bgkd->bghsd", attn_weights, v)
        # (batch, num_kv_heads, queries_per_kv, seq, kv_seq) x (batch, num_kv_heads, kv_seq, head_dim)
        # -> (batch, num_kv_heads, queries_per_kv, seq, head_dim)
        output = output.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        # (batch, num_kv_heads, queries_per_kv, seq, head_dim) -> (batch, num_heads, seq, head_dim)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        # (batch, num_heads, seq, head_dim) -> (batch, seq, num_heads, head_dim) -> (batch, seq, num_heads * head_dim)
        output = self.o_proj(output)  # (batch, seq, d_model)

        return output, k, v
