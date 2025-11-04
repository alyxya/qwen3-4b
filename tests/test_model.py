"""Tests for model configuration"""

import pytest
from model import load_config


@pytest.fixture
def config():
    """Load model config"""
    return load_config()


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
