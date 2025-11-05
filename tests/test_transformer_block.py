"""Tests for transformer block"""

import torch
import torch.nn as nn
import pytest
from model import load_config
from rmsnorm import RMSNorm
from transformer_block import TransformerBlock


@pytest.fixture
def config():
    """Load model config"""
    return load_config()


@pytest.fixture
def rms_norm(config):
    """Create RMSNorm layer"""
    return RMSNorm(d_model=config["hidden_size"], eps=config["rms_norm_eps"])


@pytest.fixture
def transformer_block(config):
    """Create transformer block"""
    return TransformerBlock(
        d_model=config["hidden_size"],
        num_heads=config["num_attention_heads"],
        num_kv_heads=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        intermediate_size=config["intermediate_size"],
        rope_theta=config["rope_theta"],
        rms_norm_eps=config["rms_norm_eps"],
    )


def test_rmsnorm_creation(rms_norm, config):
    """Test that RMSNorm is created with correct parameters"""
    assert rms_norm.eps == config["rms_norm_eps"]
    assert rms_norm.weight.shape == (config["hidden_size"],)


def test_rmsnorm_forward(rms_norm, config):
    """Test RMSNorm forward pass"""
    batch_size = 2
    seq_len = 5
    d_model = config["hidden_size"]

    x = torch.randn(batch_size, seq_len, d_model)
    output = rms_norm(x)

    assert output.shape == x.shape


def test_rmsnorm_normalizes(rms_norm):
    """Test that RMSNorm actually normalizes"""
    x = torch.randn(2, 3, rms_norm.weight.shape[0]) * 10  # Scale up

    output = rms_norm(x)

    # RMS of output should be close to 1 (before scaling by weight)
    rms_output = torch.sqrt(torch.mean(output ** 2, dim=-1))

    # The RMS won't be exactly 1 because of the weight parameter
    # but it should be controlled
    assert rms_output.mean() > 0.1
    assert rms_output.mean() < 10.0


def test_transformer_block_creation(transformer_block, config):
    """Test that transformer block is created with correct components"""
    assert transformer_block.input_layernorm is not None
    assert transformer_block.self_attn is not None
    assert transformer_block.self_attn.q_norm is not None
    assert transformer_block.self_attn.k_norm is not None
    assert transformer_block.post_attention_layernorm is not None
    assert transformer_block.mlp is not None


def test_transformer_block_forward(transformer_block, config):
    """Test transformer block forward pass"""
    batch_size = 1
    seq_len = 5
    d_model = config["hidden_size"]

    x = torch.randn(batch_size, seq_len, d_model)
    output, cache_k, cache_v = transformer_block(x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert cache_k is not None
    assert cache_v is not None


def test_transformer_block_with_cache(transformer_block, config):
    """Test transformer block with KV cache"""
    batch_size = 1
    d_model = config["hidden_size"]

    # Prefill
    prompt_len = 3
    x_prompt = torch.randn(batch_size, prompt_len, d_model)
    output_prompt, cache_k, cache_v = transformer_block(x_prompt)

    assert output_prompt.shape == (batch_size, prompt_len, d_model)

    # Generation with cache
    next_token = torch.randn(batch_size, 1, d_model)
    output_next, cache_k_new, cache_v_new = transformer_block(
        next_token, cache_k=cache_k, cache_v=cache_v
    )

    assert output_next.shape == (batch_size, 1, d_model)
    assert cache_k_new.shape[2] == prompt_len + 1  # Cache grew


def test_transformer_block_residual_connections(transformer_block, config):
    """Test that residual connections work"""
    batch_size = 1
    seq_len = 3
    d_model = config["hidden_size"]

    # Use very small input to see if residual helps
    x = torch.randn(batch_size, seq_len, d_model) * 0.01
    output, _, _ = transformer_block(x)

    # Output should not be all zeros (residual + transformations)
    assert not torch.allclose(output, torch.zeros_like(output))
    assert output.abs().mean() > 0.0


def test_transformer_block_applies_qk_norm(transformer_block, config):
    """Ensure q_norm and k_norm are invoked during attention"""

    class TrackingNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.called = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.called = True
            return x

    transformer_block.self_attn.q_norm = TrackingNorm()
    transformer_block.self_attn.k_norm = TrackingNorm()

    x = torch.randn(1, 2, config["hidden_size"])
    transformer_block(x)

    assert transformer_block.self_attn.q_norm.called
    assert transformer_block.self_attn.k_norm.called
