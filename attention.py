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
               NOTE: All sequences in batch must have same length. For variable-length
               sequences, pad to same length and use attention_mask (not yet implemented).
            cache_k: Cached key tensor, shape (batch_size, num_kv_heads, past_seq_len, head_dim) or None
            cache_v: Cached value tensor, shape (batch_size, num_kv_heads, past_seq_len, head_dim) or None

        Returns:
            Tuple of (output, new_cache_k, new_cache_v)
            - output: shape (batch_size, seq_len, d_model)
            - new_cache_k: Updated key cache, shape (batch_size, num_kv_heads, total_seq_len, head_dim)
            - new_cache_v: Updated value cache, shape (batch_size, num_kv_heads, total_seq_len, head_dim)

            Note: Cache is always returned. To use caching, pass the returned cache back in the next call.
                  To disable caching, simply don't pass the cache back (pass None).

        TODO: Add attention_mask parameter to support variable-length sequences with padding.
              This would mask out padded positions in the attention computation.
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

        # Project to Q, K, V using Linear layers
        q = self.q_proj(x)  # (batch, seq_len, num_heads * head_dim)
        k = self.k_proj(x)  # (batch, seq_len, num_kv_heads * head_dim)
        v = self.v_proj(x)  # (batch, seq_len, num_kv_heads * head_dim)

        # Reshape to separate heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_kv_heads, seq_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_kv_heads, seq_len, head_dim)

        # Apply per-head RMSNorm prior to RoPE if provided
        q = self.q_norm(q)
        k = self.k_norm(k)

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

        # Reshape Q to group queries per KV head for GQA (Grouped Query Attention)
        # Instead of expanding K,V to match Q heads, we group Q heads per KV head
        # q: (batch, num_heads, seq_len, head_dim) -> (batch, num_kv_heads, num_queries_per_kv, seq_len, head_dim)
        q = q.view(batch_size, self.num_kv_heads, self.num_queries_per_kv, seq_len, self.head_dim)
        # k, v stay as: (batch, num_kv_heads, kv_seq_len, head_dim)

        # Compute attention scores using einsum with broadcasting
        # q: (batch, kv_heads, queries_per_kv, seq_len, head_dim) - "bghsd"
        # k: (batch, kv_heads, kv_seq_len, head_dim) - "bgkd"
        # scores: (batch, kv_heads, queries_per_kv, seq_len, kv_seq_len) - "bghsk"
        kv_seq_len = k.size(2)
        scores = torch.einsum("bghsd,bgkd->bghsk", q, k)  # Broadcasting over kv_heads
        scores = scores / (self.head_dim ** 0.5)

        # Apply causal mask (prevent attending to future tokens)
        # When generating single tokens with cache (seq_len=1), we attend to all cached tokens
        # When prefill (seq_len > 1), we need causal masking
        if seq_len > 1:
            # Create causal mask: upper triangular matrix of -inf
            # Shape: (seq_len, kv_seq_len)
            mask = torch.full((seq_len, kv_seq_len), float("-inf"), device=scores.device)
            mask = torch.triu(mask, diagonal=kv_seq_len - seq_len + 1)
            scores = scores + mask

        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values using einsum with broadcasting
        # attn_weights: (batch, kv_heads, queries_per_kv, seq_len, kv_seq_len) - "bghsk"
        # v: (batch, kv_heads, kv_seq_len, head_dim) - "bgkd"
        # output: (batch, kv_heads, queries_per_kv, seq_len, head_dim) - "bghsd"
        output = torch.einsum("bghsk,bgkd->bghsd", attn_weights, v)

        # Reshape back to (batch, num_heads, seq_len, head_dim) then transpose
        output = output.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        output = output.transpose(1, 2)  # (batch, seq_len, num_heads, head_dim)
        output = output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        # Final projection using Linear layer
        output = self.o_proj(output)  # (batch, seq_len, d_model)

        return output, new_cache_k, new_cache_v
