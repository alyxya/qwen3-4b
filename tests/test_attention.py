"""Tests for multi-head attention with GQA"""

import torch
import pytest
from attention import Attention


@pytest.fixture
def attention_layer(config):
    """Create attention layer"""
    return Attention(
        d_model=config["hidden_size"],
        num_heads=config["num_attention_heads"],
        num_kv_heads=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        rope_theta=config["rope_theta"]
    )


def test_attention_creation(attention_layer, config):
    """Test that attention layer is created with correct dimensions"""
    assert attention_layer.d_model == config["hidden_size"]
    assert attention_layer.num_heads == config["num_attention_heads"]
    assert attention_layer.num_kv_heads == config["num_key_value_heads"]
    assert attention_layer.head_dim == config["head_dim"]


def test_attention_prefill(attention_layer, config):
    """Test attention forward pass without cache (prefill)"""
    batch_size = 1
    seq_len = 5
    d_model = config["hidden_size"]

    x = torch.randn(batch_size, seq_len, d_model)

    output, cache_k, cache_v = attention_layer(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model)

    # Check cache is returned
    assert cache_k is not None
    assert cache_v is not None
    assert cache_k.shape == (batch_size, config["num_key_value_heads"], seq_len, config["head_dim"])
    assert cache_v.shape == (batch_size, config["num_key_value_heads"], seq_len, config["head_dim"])


def test_attention_with_cache(attention_layer, config):
    """Test attention generation with KV cache"""
    batch_size = 1
    d_model = config["hidden_size"]

    # Prefill with prompt
    prompt_len = 3
    x_prompt = torch.randn(batch_size, prompt_len, d_model)

    output_prompt, cache_k, cache_v = attention_layer(x_prompt)

    assert output_prompt.shape == (batch_size, prompt_len, d_model)
    assert cache_k.shape[2] == prompt_len  # Cache seq_len matches prompt

    # Generate next token using cache
    next_token = torch.randn(batch_size, 1, d_model)

    output_next, cache_k_new, cache_v_new = attention_layer(
        next_token, cache_k=cache_k, cache_v=cache_v
    )

    assert output_next.shape == (batch_size, 1, d_model)
    assert cache_k_new.shape[2] == prompt_len + 1  # Cache grew by 1
    assert cache_v_new.shape[2] == prompt_len + 1


def test_attention_causal_mask(attention_layer, config):
    """Test that causal mask prevents attending to future tokens"""
    batch_size = 1
    seq_len = 4
    d_model = config["hidden_size"]

    x = torch.randn(batch_size, seq_len, d_model)
    position_ids = torch.arange(seq_len, device=x.device)

    # Manually compute attention to inspect weights
    q = torch.einsum("bsd,hd->bsh", x, attention_layer.w_q)
    k = torch.einsum("bsd,kd->bsk", x, attention_layer.w_k)

    q = q.view(batch_size, seq_len, attention_layer.num_heads, attention_layer.head_dim)
    k = k.view(batch_size, seq_len, attention_layer.num_kv_heads, attention_layer.head_dim)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    q = attention_layer.rope(q, position_ids)
    k = attention_layer.rope(k, position_ids)

    # Reshape q to group queries per KV head (same as in attention.py)
    q = q.view(batch_size, attention_layer.num_kv_heads, attention_layer.num_queries_per_kv, seq_len, attention_layer.head_dim)

    # Compute scores with broadcasting
    scores = torch.einsum("bghsd,bgkd->bghsk", q, k) / (attention_layer.head_dim ** 0.5)

    # Apply causal mask
    mask = torch.full((seq_len, seq_len), float("-inf"), device=scores.device)
    mask = torch.triu(mask, diagonal=1)
    # Extract first batch, first kv_head, first query within that group
    masked_scores = scores[0, 0, 0] + mask

    attn_weights = torch.softmax(masked_scores, dim=-1)

    # Verify upper triangle is zero (causal)
    upper_triangle = torch.triu(attn_weights, diagonal=1)
    assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle), atol=1e-6)


def test_attention_gqa_expansion(attention_layer, config):
    """Test that GQA correctly expands KV heads"""
    num_queries_per_kv = config["num_attention_heads"] // config["num_key_value_heads"]
    assert attention_layer.num_queries_per_kv == num_queries_per_kv

    # Each KV head should be shared by num_queries_per_kv query heads
    assert config["num_attention_heads"] == config["num_key_value_heads"] * num_queries_per_kv
