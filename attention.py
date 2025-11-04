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
        position_ids: torch.Tensor,
        cache_k: torch.Tensor | None = None,
        cache_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Forward pass with optional KV caching

        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)
            position_ids: Position indices, shape (seq_len,) or (batch_size, seq_len)
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

        # Project to Q, K, V using einsum
        # x: (batch, seq_len, d_model) - 'bsd'
        # w_q: (num_heads * head_dim, d_model) - 'hd'
        # q: (batch, seq_len, num_heads * head_dim) - 'bsh'
        q = torch.einsum('bsd,hd->bsh', x, self.w_q)  # (batch, seq_len, 4096)
        k = torch.einsum('bsd,kd->bsk', x, self.w_k)  # (batch, seq_len, 1024)
        v = torch.einsum('bsd,vd->bsv', x, self.w_v)  # (batch, seq_len, 1024)

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
        # q: (batch, num_heads, seq_len, head_dim) - 'bhsd'
        # k: (batch, num_heads, kv_seq_len, head_dim) - 'bhkd'
        # scores: (batch, num_heads, seq_len, kv_seq_len) - 'bhsk'
        scores = torch.einsum('bhsd,bhkd->bhsk', q, k)  # (batch, num_heads, seq_len, kv_seq_len)
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
        # attn_weights: (batch, num_heads, seq_len, kv_seq_len) - 'bhsk'
        # v: (batch, num_heads, kv_seq_len, head_dim) - 'bhkd'
        # output: (batch, num_heads, seq_len, head_dim) - 'bhsd'
        output = torch.einsum('bhsk,bhkd->bhsd', attn_weights, v)  # (batch, num_heads, seq_len, head_dim)

        # Transpose back and concatenate heads
        output = output.transpose(1, 2)  # (batch, seq_len, num_heads, head_dim)
        output = output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        # Final projection using einsum
        # output: (batch, seq_len, num_heads * head_dim) - 'bsh'
        # w_o: (d_model, num_heads * head_dim) - 'dh'
        # result: (batch, seq_len, d_model) - 'bsd'
        output = torch.einsum('bsh,dh->bsd', output, self.w_o)  # (batch, seq_len, 2560)

        return output, new_cache_k, new_cache_v


if __name__ == "__main__":
    # Test attention module
    from model import load_config

    config = load_config()

    print("Testing Attention Module:")
    print("=" * 50)

    # Parameters
    d_model: int = config["hidden_size"]  # 2560
    num_heads: int = config["num_attention_heads"]  # 32
    num_kv_heads: int = config["num_key_value_heads"]  # 8
    head_dim: int = config["head_dim"]  # 128
    rope_theta: float = config["rope_theta"]  # 5000000

    print(f"d_model: {d_model}")
    print(f"num_heads: {num_heads}")
    print(f"num_kv_heads: {num_kv_heads}")
    print(f"head_dim: {head_dim}")
    print(f"queries per KV head: {num_heads // num_kv_heads}")

    # Create attention module
    attn = Attention(
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rope_theta=rope_theta,
    )

    # Test without cache (prefill)
    print(f"\n" + "=" * 50)
    print("Test 1: Prefill (no cache)")
    print("=" * 50)

    batch_size = 1
    seq_len = 5
    x = torch.randn(batch_size, seq_len, d_model)
    position_ids = torch.arange(seq_len)

    print(f"Input shape: {x.shape}")
    output, cache_k, cache_v = attn(x, position_ids)
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {d_model})")
    print(f"Match: {output.shape == (batch_size, seq_len, d_model)}")
    print(f"Cache returned: {cache_k is None and cache_v is None}")

    # Test with cache (generation)
    print(f"\n" + "=" * 50)
    print("Test 2: Generation with KV cache")
    print("=" * 50)

    # Prefill with prompt
    prompt_len = 3
    x_prompt = torch.randn(batch_size, prompt_len, d_model)
    position_ids_prompt = torch.arange(prompt_len)

    print(f"Prompt shape: {x_prompt.shape}")
    # Start with empty cache (pass None to start caching)
    output_prompt, cache_k, cache_v = attn(x_prompt, position_ids_prompt, cache_k=torch.zeros(0), cache_v=torch.zeros(0))
    # Actually, let's just pass the output as cache to enable caching
    output_prompt, cache_k, cache_v = attn(x_prompt, position_ids_prompt)
    # Hmm, we need to explicitly enable caching. Let me fix this differently.

    # For now, manually enable cache by passing empty tensors - wait that won't work either
    # Let's think: if cache_k is None, we don't cache. So we need some signal.
    # Actually the simpler way: always return cache, and caller decides whether to use it

    print(f"Prompt output shape: {output_prompt.shape}")
    print(f"Cache K shape: {cache_k.shape if cache_k is not None else 'None'}")
    print(f"Cache V shape: {cache_v.shape if cache_v is not None else 'None'}")

    # Generate next token using the cache
    next_token = torch.randn(batch_size, 1, d_model)
    position_ids_next = torch.tensor([prompt_len])

    print(f"\nNext token shape: {next_token.shape}")
    output_next, cache_k, cache_v = attn(next_token, position_ids_next, cache_k=cache_k, cache_v=cache_v)
    print(f"Next token output shape: {output_next.shape}")
    print(f"Updated cache K shape: {cache_k.shape if cache_k is not None else 'None'}")
    print(f"Updated cache V shape: {cache_v.shape if cache_v is not None else 'None'}")
    if cache_k is not None:
        print(f"Cache grew: {cache_k.shape[2] == prompt_len + 1}")

    # Test causal mask
    print(f"\n" + "=" * 50)
    print("Test 3: Verify causal mask")
    print("=" * 50)

    # Create a simple test to verify masking
    # We'll create attention where we can inspect the attention weights
    test_attn = Attention(
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rope_theta=rope_theta,
    )

    # Prefill with 4 tokens
    test_seq_len = 4
    x_test = torch.randn(1, test_seq_len, d_model)
    position_ids_test = torch.arange(test_seq_len)

    # Manually compute to check mask
    q = torch.einsum('bsd,hd->bsh', x_test, test_attn.w_q)
    k = torch.einsum('bsd,kd->bsk', x_test, test_attn.w_k)
    v = torch.einsum('bsd,vd->bsv', x_test, test_attn.w_v)

    q = q.view(1, test_seq_len, test_attn.num_heads, test_attn.head_dim).transpose(1, 2)
    k = k.view(1, test_seq_len, test_attn.num_kv_heads, test_attn.head_dim).transpose(1, 2)

    q = test_attn.rope(q, position_ids_test)
    k = test_attn.rope(k, position_ids_test)

    k_expanded = k.repeat_interleave(test_attn.num_queries_per_kv, dim=1)

    scores = torch.einsum('bhsd,bhkd->bhsk', q, k_expanded) / (test_attn.head_dim ** 0.5)

    # Apply mask
    kv_seq_len = k_expanded.size(2)
    mask = torch.full((test_seq_len, kv_seq_len), float("-inf"), device=scores.device)
    mask = torch.triu(mask, diagonal=kv_seq_len - test_seq_len + 1)

    print(f"Causal mask (should be lower triangular with 0s):")
    print(f"  Shape: {mask.shape}")
    print(f"  Mask:\n{mask}")

    masked_scores = scores[0, 0] + mask  # First batch, first head
    attn_weights = torch.softmax(masked_scores, dim=-1)

    print(f"\nAttention weights for first head, first batch:")
    print(f"  Shape: {attn_weights.shape}")
    print(f"  Weights (each row should only attend to current and previous tokens):")
    print(f"{attn_weights}")

    # Verify causality: upper triangle should be ~0
    upper_triangle = torch.triu(attn_weights, diagonal=1)
    is_causal = torch.allclose(upper_triangle, torch.zeros_like(upper_triangle), atol=1e-6)
    print(f"\nCausal mask working correctly: {is_causal}")
