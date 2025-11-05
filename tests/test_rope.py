"""Tests for RoPE (Rotary Position Embeddings)"""

import torch
import pytest
from src.rope import RoPE


@pytest.fixture
def rope(config):
    """Create RoPE module"""
    return RoPE(
        head_dim=config["head_dim"],
        max_seq_len=config["max_position_embeddings"],
        theta=config["rope_theta"],
    )


def test_rope_creation(rope, config):
    """Test that RoPE is created with correct parameters"""
    assert rope.head_dim == config["head_dim"]
    assert rope.theta == config["rope_theta"]
    assert rope.inv_freq.shape == (config["head_dim"] // 2,)


def test_rope_forward_shape(rope, config):
    """Test that RoPE maintains input shape"""
    batch_size = 2
    num_heads = config["num_attention_heads"]
    seq_len = 10
    head_dim = config["head_dim"]

    x = torch.randn(batch_size, num_heads, seq_len, head_dim)
    position_ids = torch.arange(seq_len)

    output = rope(x, position_ids)

    assert output.shape == x.shape


def test_rope_with_1d_position_ids(rope, config):
    """Test RoPE with 1D position_ids"""
    batch_size = 2
    num_heads = config["num_attention_heads"]
    seq_len = 5
    head_dim = config["head_dim"]

    x = torch.randn(batch_size, num_heads, seq_len, head_dim)
    position_ids = torch.arange(seq_len)  # 1D

    output = rope(x, position_ids)

    assert output.shape == (batch_size, num_heads, seq_len, head_dim)


def test_rope_with_2d_position_ids(rope, config):
    """Test RoPE with 2D batched position_ids"""
    batch_size = 2
    num_heads = config["num_attention_heads"]
    seq_len = 5
    head_dim = config["head_dim"]

    x = torch.randn(batch_size, num_heads, seq_len, head_dim)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)  # 2D

    output = rope(x, position_ids)

    assert output.shape == (batch_size, num_heads, seq_len, head_dim)


def test_rope_consistency_1d_vs_2d(rope, config):
    """Test that 1D and 2D position_ids give same result"""
    batch_size = 2
    num_heads = config["num_attention_heads"]
    seq_len = 5
    head_dim = config["head_dim"]

    x = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # 1D position_ids
    position_ids_1d = torch.arange(seq_len)
    output_1d = rope(x, position_ids_1d)

    # 2D position_ids
    position_ids_2d = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    output_2d = rope(x, position_ids_2d)

    assert torch.allclose(output_1d, output_2d)


def test_rope_changes_input(rope, config):
    """Test that RoPE actually transforms the input"""
    batch_size = 1
    num_heads = config["num_attention_heads"]
    seq_len = 3
    head_dim = config["head_dim"]

    x = torch.randn(batch_size, num_heads, seq_len, head_dim)
    position_ids = torch.arange(seq_len)

    output = rope(x, position_ids)

    # Output should be different from input (RoPE applies rotation)
    assert not torch.allclose(x, output)
