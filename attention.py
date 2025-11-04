"""
Multi-head Attention with Grouped Query Attention (GQA) for Qwen3 4B

GQA uses fewer Key/Value heads than Query heads to save memory:
- 32 Query heads
- 8 Key/Value heads (each shared by 4 Query heads)
"""

import torch
import torch.nn as nn
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
        rope_theta: float = 5000000.0,
    ) -> None:
        """
        Initialize attention layer

        Args:
            d_model: Model dimension (2560 for Qwen3 4B)
            num_heads: Number of query heads (32 for Qwen3 4B)
            num_kv_heads: Number of key/value heads (8 for Qwen3 4B)
            head_dim: Dimension per head (128 for Qwen3 4B)
            rope_theta: RoPE base frequency (5000000 for Qwen3 4B)
        """
        super().__init__()
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads
        self.head_dim: int = head_dim
        self.num_queries_per_kv: int = num_heads // num_kv_heads  # 32 // 8 = 4

        # Projection weight matrices (no bias)
        # Query: projects from d_model to (num_heads * head_dim)
        self.w_q = nn.Parameter(torch.randn(num_heads * head_dim, d_model))  # (4096, 2560)

        # Key: projects from d_model to (num_kv_heads * head_dim)
        self.w_k = nn.Parameter(torch.randn(num_kv_heads * head_dim, d_model))  # (1024, 2560)

        # Value: projects from d_model to (num_kv_heads * head_dim)
        self.w_v = nn.Parameter(torch.randn(num_kv_heads * head_dim, d_model))  # (1024, 2560)

        # Output: projects from (num_heads * head_dim) back to d_model
        self.w_o = nn.Parameter(torch.randn(d_model, num_heads * head_dim))  # (2560, 4096)

        # RoPE for positional encoding
        self.rope = RoPE(head_dim=head_dim, theta=rope_theta)

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
            cache_k: Cached key tensor, shape (batch_size, num_kv_heads, past_seq_len, head_dim) or None
            cache_v: Cached value tensor, shape (batch_size, num_kv_heads, past_seq_len, head_dim) or None

        Returns:
            Tuple of (output, new_cache_k, new_cache_v)
            - output: shape (batch_size, seq_len, d_model)
            - new_cache_k: Updated key cache, shape (batch_size, num_kv_heads, total_seq_len, head_dim)
            - new_cache_v: Updated value cache, shape (batch_size, num_kv_heads, total_seq_len, head_dim)

            Note: Cache is always returned. To use caching, pass the returned cache back in the next call.
                  To disable caching, simply don't pass the cache back (pass None).
        """
        batch_size, seq_len, _ = x.shape

        # Derive position_ids automatically from cache
        if cache_k is not None:
            # Continue from where cache left off
            start_pos = cache_k.shape[2]
            position_ids = torch.arange(start_pos, start_pos + seq_len, device=x.device)
        else:
            # Start from position 0
            position_ids = torch.arange(seq_len, device=x.device)

        # Project to Q, K, V using einsum
        # x: (batch, seq_len, d_model) - "bsd"
        # w_q: (num_heads * head_dim, d_model) - "hd"
        # q: (batch, seq_len, num_heads * head_dim) - "bsh"
        q = torch.einsum("bsd,hd->bsh", x, self.w_q)  # (batch, seq_len, 4096)
        k = torch.einsum("bsd,kd->bsk", x, self.w_k)  # (batch, seq_len, 1024)
        v = torch.einsum("bsd,vd->bsv", x, self.w_v)  # (batch, seq_len, 1024)

        # Reshape to separate heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_kv_heads, seq_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_kv_heads, seq_len, head_dim)

        # Apply RoPE to Q and K
        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        # Handle KV cache for generation
        if cache_k is not None and cache_v is not None:
            # Concatenate new K, V with cached K, V
            k = torch.cat([cache_k, k], dim=2)  # Concat along seq_len
            v = torch.cat([cache_v, v], dim=2)

        # Always return the current K, V as cache (caller decides whether to use it)
        new_cache_k = k
        new_cache_v = v

        # Expand K, V to match number of Q heads (GQA)
        # Each KV head is repeated num_queries_per_kv times
        # k: (batch, num_kv_heads, seq_len, head_dim) -> (batch, num_heads, seq_len, head_dim)
        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Compute attention scores using einsum
        # q: (batch, num_heads, seq_len, head_dim) - "bhsd"
        # k: (batch, num_heads, kv_seq_len, head_dim) - "bhkd"
        # scores: (batch, num_heads, seq_len, kv_seq_len) - "bhsk"
        scores = torch.einsum("bhsd,bhkd->bhsk", q, k)  # (batch, num_heads, seq_len, kv_seq_len)
        scores = scores / (self.head_dim ** 0.5)

        # Apply causal mask (prevent attending to future tokens)
        # When generating single tokens with cache (seq_len=1), we attend to all cached tokens
        # When prefill (seq_len > 1), we need causal masking
        kv_seq_len = k.size(2)
        if seq_len > 1:
            # Create causal mask: upper triangular matrix of -inf
            # Shape: (seq_len, kv_seq_len)
            mask = torch.full((seq_len, kv_seq_len), float("-inf"), device=scores.device)
            mask = torch.triu(mask, diagonal=kv_seq_len - seq_len + 1)
            scores = scores + mask

        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values using einsum
        # attn_weights: (batch, num_heads, seq_len, kv_seq_len) - "bhsk"
        # v: (batch, num_heads, kv_seq_len, head_dim) - "bhkd"
        # output: (batch, num_heads, seq_len, head_dim) - "bhsd"
        output = torch.einsum("bhsk,bhkd->bhsd", attn_weights, v)  # (batch, num_heads, seq_len, head_dim)

        # Transpose back and concatenate heads
        output = output.transpose(1, 2)  # (batch, seq_len, num_heads, head_dim)
        output = output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        # Final projection using einsum
        # output: (batch, seq_len, num_heads * head_dim) - "bsh"
        # w_o: (d_model, num_heads * head_dim) - "dh"
        # result: (batch, seq_len, d_model) - "bsd"
        output = torch.einsum("bsh,dh->bsd", output, self.w_o)  # (batch, seq_len, 2560)

        return output, new_cache_k, new_cache_v
