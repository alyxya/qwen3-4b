"""Tests for model configuration and Qwen3Model"""

import pytest
import torch
from src.model import Qwen3Model


def test_config_loaded(config):
    """Test that config is loaded successfully"""
    assert config is not None
    assert isinstance(config, dict)


def test_config_has_required_fields(config):
    """Test that config contains all required fields"""
    required_fields = [
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "num_hidden_layers",
        "intermediate_size",
        "vocab_size",
        "rope_theta",
        "head_dim",
    ]

    for field in required_fields:
        assert field in config, f"Missing required field: {field}"


def test_config_qwen3_4b_values(config):
    """Test that config matches expected Qwen3 4B values"""
    assert config["hidden_size"] == 2560
    assert config["num_attention_heads"] == 32
    assert config["num_key_value_heads"] == 8
    assert config["num_hidden_layers"] == 36
    assert config["intermediate_size"] == 9728
    assert config["vocab_size"] == 151936
    assert config["rope_theta"] == 5000000.0
    assert config["head_dim"] == 128


def test_config_gqa_ratio(config):
    """Test that GQA ratio is correct"""
    gqa_ratio = config["num_attention_heads"] // config["num_key_value_heads"]
    assert gqa_ratio == 4  # 32 // 8 = 4


def test_config_head_dim_calculation(config):
    """Test that head_dim matches expected calculation"""
    # Note: head_dim is explicitly 128 in config, not derived from hidden_size
    assert config["head_dim"] == 128
    # Total query dimension: num_heads * head_dim
    assert config["num_attention_heads"] * config["head_dim"] == 4096


@pytest.mark.slow
def test_model_initialization(model):
    """Test that model initializes successfully"""
    assert model is not None
    assert isinstance(model, Qwen3Model)
    assert model.vocab_size == 151936
    assert model.d_model == 2560
    assert model.num_layers == 36
    assert model.num_heads == 32
    assert model.num_kv_heads == 8


@pytest.mark.slow
def test_model_parameter_count(model):
    """Test that model has correct number of parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    # Should be approximately 4B parameters
    assert 4.0e9 < total_params < 4.1e9
    assert total_params == 4022468096


@pytest.mark.slow
def test_model_has_layers(model):
    """Test that model has correct number of layers"""
    assert len(model.layers) == 36
    assert model.embed_tokens is not None
    assert model.norm is not None
    assert model.lm_head is not None


@pytest.mark.slow
def test_model_forward_pass(model, config):
    """Test that model forward pass works correctly"""
    batch_size = 2
    seq_len = 10
    dummy_input = torch.randint(0, config["vocab_size"], (batch_size, seq_len))

    with torch.no_grad():
        logits, cache_k, cache_v = model(dummy_input)

    # Check output shapes
    assert logits.shape == (batch_size, seq_len, config["vocab_size"])
    assert len(cache_k) == 36
    assert len(cache_v) == 36

    # Check cache shapes
    # cache shape: (batch_size, num_kv_heads, seq_len, head_dim)
    assert cache_k[0].shape == (batch_size, 8, seq_len, 128)
    assert cache_v[0].shape == (batch_size, 8, seq_len, 128)


@pytest.mark.slow
def test_model_forward_with_cache(model, config):
    """Test that model forward pass works with KV cache"""
    batch_size = 1
    seq_len = 5

    # First forward pass
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    with torch.no_grad():
        logits1, cache_k, cache_v = model(input_ids)

    # Second forward pass with cache (single token)
    next_token = torch.randint(0, config["vocab_size"], (batch_size, 1))
    with torch.no_grad():
        logits2, cache_k2, cache_v2 = model(
            next_token, cache_k=cache_k, cache_v=cache_v
        )

    # Check that cache grew
    assert cache_k2[0].shape[2] == seq_len + 1  # seq_len dimension should grow
    assert cache_v2[0].shape[2] == seq_len + 1
    assert logits2.shape == (batch_size, 1, config["vocab_size"])


def test_generate_per_sequence_stop():
    """Ensure generation stops per sequence, not per batch."""

    class DummyModel(Qwen3Model):
        def __init__(
            self,
            device: str | torch.device,
            step_tokens: list[list[int]],
        ) -> None:
            torch.nn.Module.__init__(self)
            self.device = device if isinstance(device, torch.device) else torch.device(device)
            self._step_tokens = step_tokens
            self._step = 0
            self.vocab_size = max(max(row) for row in step_tokens) + 1

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            cache_k: list[torch.Tensor] | None = None,
            cache_v: list[torch.Tensor] | None = None,
        ) -> tuple[torch.Tensor, list[torch.Tensor] | None, list[torch.Tensor] | None]:
            batch_size, seq_len = input_ids.shape
            logits = torch.full(
                (batch_size, seq_len, self.vocab_size),
                -1e9,
                device=input_ids.device,
            )
            step_ids = self._step_tokens[min(self._step, len(self._step_tokens) - 1)]
            for batch_idx, token_id in enumerate(step_ids):
                logits[batch_idx, -1, token_id] = 0.0
            self._step += 1
            return logits, cache_k, cache_v

    step_tokens = [
        [2, 5],  # seq0 stops immediately, seq1 continues
        [2, 2],  # seq1 stops on the next step
    ]
    model = DummyModel(
        device="cpu",
        step_tokens=step_tokens,
    )
    input_ids = torch.tensor([[1], [1]])
    new_tokens, _, _ = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        temperature=0.0,
        stop_token_ids=[2],
    )

    assert new_tokens.shape == (2, 2)
    assert new_tokens[0].tolist() == [2, 2]
    assert new_tokens[1].tolist() == [5, 2]
