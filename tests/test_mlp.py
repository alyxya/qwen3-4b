"""Tests for MLP (feed-forward network)"""

import torch
import pytest
from mlp import MLP


@pytest.fixture
def mlp_layer(config):
    """Create MLP layer"""
    return MLP(
        d_model=config["hidden_size"],
        intermediate_size=config["intermediate_size"]
    )


def test_mlp_creation(mlp_layer, config):
    """Test that MLP layer is created with correct dimensions"""
    assert mlp_layer.d_model == config["hidden_size"]
    assert mlp_layer.intermediate_size == config["intermediate_size"]
    assert mlp_layer.gate_proj.weight.shape == (config["intermediate_size"], config["hidden_size"])
    assert mlp_layer.up_proj.weight.shape == (config["intermediate_size"], config["hidden_size"])
    assert mlp_layer.down_proj.weight.shape == (config["hidden_size"], config["intermediate_size"])


def test_mlp_forward(mlp_layer, config):
    """Test MLP forward pass produces correct output shape"""
    batch_size = 2
    seq_len = 5
    d_model = config["hidden_size"]

    x = torch.randn(batch_size, seq_len, d_model)
    output = mlp_layer(x)

    assert output.shape == (batch_size, seq_len, d_model)


def test_mlp_intermediate_shapes(mlp_layer, config):
    """Test that intermediate activations have correct shapes"""
    batch_size = 2
    seq_len = 5
    d_model = config["hidden_size"]
    intermediate_size = config["intermediate_size"]

    x = torch.randn(batch_size, seq_len, d_model)

    # Gate pathway
    gate = mlp_layer.gate_proj(x)
    assert gate.shape == (batch_size, seq_len, intermediate_size)

    # Up pathway
    up = mlp_layer.up_proj(x)
    assert up.shape == (batch_size, seq_len, intermediate_size)

    # Hidden (after gating)
    hidden = mlp_layer.activation(gate) * up
    assert hidden.shape == (batch_size, seq_len, intermediate_size)

    # Output
    output = mlp_layer.down_proj(hidden)
    assert output.shape == (batch_size, seq_len, d_model)


def test_mlp_silu_activation(mlp_layer):
    """Test that SiLU activation works correctly"""
    # SiLU(x) = x * sigmoid(x)
    test_input = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    test_output = mlp_layer.activation(test_input)

    # SiLU(0) should be 0
    assert torch.allclose(test_output[2], torch.tensor(0.0), atol=1e-6)

    # SiLU(1) should be approximately 0.731
    assert torch.allclose(test_output[3], torch.tensor(0.7310585786), atol=1e-4)

    # SiLU should be smooth and continuous
    assert test_output.shape == test_input.shape


def test_mlp_expansion_ratio(mlp_layer, config):
    """Test that MLP expansion ratio is correct"""
    expansion_ratio = config["intermediate_size"] / config["hidden_size"]
    # Qwen3 4B uses expansion ratio of ~3.8x
    assert expansion_ratio > 3.5
    assert expansion_ratio < 4.0
