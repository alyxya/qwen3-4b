"""
Multi-head Attention with Grouped Query Attention (GQA) for Qwen3 4B

GQA uses fewer Key/Value heads than Query heads to save memory:
- 32 Query heads
- 8 Key/Value heads (each shared by 4 Query heads)
"""

import torch
import torch.nn as nn
from rmsnorm import RMSNorm
from rope import RoPE


class Attention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA) and KV cache

    Standard attention: Q, K, V all have same number of heads
    GQA: K and V have fewer heads, shared across multiple Q heads
    """

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
        """
        Initialize attention layer

        Args:
            d_model: Model dimension (2560 for Qwen3 4B)
            num_heads: Number of query heads (32 for Qwen3 4B)
            num_kv_heads: Number of key/value heads (8 for Qwen3 4B)
            head_dim: Dimension per head (128 for Qwen3 4B)
            max_position_embeddings: Maximum sequence length (262144 for Qwen3 4B)
            rope_theta: RoPE base frequency (5000000 for Qwen3 4B)
            rms_norm_eps: RMSNorm epsilon applied to projected Q and K heads
        """
        super().__init__()
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads
        self.head_dim: int = head_dim
        self.num_queries_per_kv: int = num_heads // num_kv_heads  # 32 // 8 = 4
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        # Projection layers to match HuggingFace naming
        # Using nn.Linear without bias to match pretrained weights
        # Query: projects from d_model to (num_heads * head_dim)
        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)  # (2560 -> 4096)

        # Key: projects from d_model to (num_kv_heads * head_dim)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)  # (2560 -> 1024)

        # Value: projects from d_model to (num_kv_heads * head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)  # (2560 -> 1024)

        # Output: projects from (num_heads * head_dim) back to d_model
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)  # (4096 -> 2560)

        # RoPE for positional encoding
        self.rope = RoPE(head_dim=head_dim, max_seq_len=max_position_embeddings, theta=rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        cache_k: torch.Tensor | None = None,
        cache_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional KV caching

        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)
            cache_k: Cached key tensor or None
            cache_v: Cached value tensor or None

        Returns:
            Tuple of (output, new_cache_k, new_cache_v)
        """
        batch_size, seq_len, _ = x.shape  # x: (batch, seq, dim)

        # Derive position_ids automatically from cache
        if cache_k is not None:
            start_pos = cache_k.shape[2]
            position_ids = torch.arange(start_pos, start_pos + seq_len, device=x.device)  # (seq,)
        else:
            position_ids = torch.arange(seq_len, device=x.device)  # (seq,)

        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq, num_heads * head_dim) = (batch, seq, 4096)
        k = self.k_proj(x)  # (batch, seq, num_kv_heads * head_dim) = (batch, seq, 1024)
        v = self.v_proj(x)  # (batch, seq, num_kv_heads * head_dim) = (batch, seq, 1024)

        # Reshape to separate heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)      # (batch, seq, 32, 128)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)   # (batch, seq, 8, 128)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)   # (batch, seq, 8, 128)

        q = q.transpose(1, 2)  # (batch, 32, seq, 128)
        k = k.transpose(1, 2)  # (batch, 8, seq, 128)
        v = v.transpose(1, 2)  # (batch, 8, seq, 128)

        # Apply per-head RMSNorm and RoPE
        q = self.q_norm(q)  # (batch, 32, seq, 128)
        k = self.k_norm(k)  # (batch, 8, seq, 128)
        q = self.rope(q, position_ids)  # (batch, 32, seq, 128)
        k = self.rope(k, position_ids)  # (batch, 8, seq, 128)

        # Concatenate with KV cache if provided
        if cache_k is not None and cache_v is not None:
            k = torch.cat([cache_k, k], dim=2)  # (batch, 8, seq_total, 128)
            v = torch.cat([cache_v, v], dim=2)  # (batch, 8, seq_total, 128)

        new_cache_k = k  # (batch, 8, seq_total, 128)
        new_cache_v = v  # (batch, 8, seq_total, 128)

        # Reshape Q for Grouped Query Attention
        # Group Q heads per KV head instead of expanding K,V to match Q heads
        q = q.view(batch_size, self.num_kv_heads, self.num_queries_per_kv, seq_len, self.head_dim)  # (batch, 8, 4, seq, 128)

        # Compute attention scores
        kv_seq_len = k.size(2)
        scores = torch.einsum("bghsd,bgkd->bghsk", q, k)  # (batch, 8, 4, seq, seq_total)
        scores = scores / (self.head_dim ** 0.5)  # (batch, 8, 4, seq, seq_total)

        # Apply causal mask for prefill (seq_len > 1)
        if seq_len > 1:
            mask = torch.full((seq_len, kv_seq_len), float("-inf"), device=scores.device)  # (seq, seq_total)
            mask = torch.triu(mask, diagonal=kv_seq_len - seq_len + 1)  # (seq, seq_total)
            scores = scores + mask  # (batch, 8, 4, seq, seq_total)

        attn_weights = torch.softmax(scores, dim=-1)  # (batch, 8, 4, seq, seq_total)

        # Apply attention to values
        output = torch.einsum("bghsk,bgkd->bghsd", attn_weights, v)  # (batch, 8, 4, seq, 128)

        # Reshape and project output
        output = output.reshape(batch_size, self.num_heads, seq_len, self.head_dim)  # (batch, 32, seq, 128)
        output = output.transpose(1, 2)  # (batch, seq, 32, 128)
        output = output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)  # (batch, seq, 4096)
        output = self.o_proj(output)  # (batch, seq, dim) = (batch, seq, 2560)

        return output, new_cache_k, new_cache_v
