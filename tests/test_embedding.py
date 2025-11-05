"""Tests for embedding layer"""

import torch
import pytest
from src.embedding import Embedding


@pytest.fixture
def embedding_layer(config):
    """Create embedding layer"""
    return Embedding(vocab_size=config["vocab_size"], d_model=config["hidden_size"])


def test_embedding_creation(embedding_layer, config):
    """Test that embedding layer is created with correct dimensions"""
    assert embedding_layer.vocab_size == config["vocab_size"]
    assert embedding_layer.d_model == config["hidden_size"]
    assert embedding_layer.embedding.weight.shape == (
        config["vocab_size"],
        config["hidden_size"],
    )


def test_embedding_forward(embedding_layer, config):
    """Test forward pass through embedding layer"""
    batch_size = 2
    seq_len = 5

    # Create random token IDs
    token_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))

    # Forward pass
    output = embedding_layer(token_ids)

    # Check output shape
    assert output.shape == (batch_size, seq_len, config["hidden_size"])


def test_embedding_lookup(embedding_layer):
    """Test that embedding performs correct lookup"""
    # Single token
    token_id = torch.tensor([42])
    output = embedding_layer(token_id)

    # Should match the 42nd row of the weight matrix
    expected = embedding_layer.embedding.weight[42]
    assert torch.allclose(output[0], expected)


def test_embedding_batch(embedding_layer):
    """Test embedding with batched input"""
    batch_size = 3
    seq_len = 4
    token_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    output = embedding_layer(token_ids)

    # Check shape
    assert output.shape == (batch_size, seq_len, embedding_layer.d_model)

    # Check first token of first batch matches
    assert torch.allclose(output[0, 0], embedding_layer.embedding.weight[1])
