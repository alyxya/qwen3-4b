"""Tests for tokenizer"""

import pytest
from src.tokenizer import Tokenizer


@pytest.fixture
def tokenizer():
    """Create tokenizer instance"""
    return Tokenizer()


def test_tokenizer_creation(tokenizer):
    """Test that tokenizer is created successfully"""
    assert len(tokenizer.vocab) > 0
    assert len(tokenizer.merges) > 0


def test_encode_simple(tokenizer):
    """Test encoding simple text"""
    text = "Hello, world!"
    token_ids = tokenizer.encode(text)

    assert isinstance(token_ids, list)
    assert len(token_ids) > 0
    assert all(isinstance(tid, int) for tid in token_ids)
    assert all(0 <= tid < len(tokenizer.vocab) for tid in token_ids)


def test_decode_simple(tokenizer):
    """Test decoding token IDs"""
    text = "Hello, world!"
    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)

    assert isinstance(decoded, str)
    assert decoded == text


def test_encode_decode_roundtrip(tokenizer):
    """Test that encode->decode is a perfect roundtrip"""
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Python programming is fun!",
        "123456789",
        "Special chars: @#$%^&*()",
    ]

    for text in test_texts:
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)
        assert decoded == text, f"Roundtrip failed for: {text}"


def test_encode_empty_string(tokenizer):
    """Test encoding empty string"""
    token_ids = tokenizer.encode("")
    assert token_ids == []


def test_decode_empty_list(tokenizer):
    """Test decoding empty token list"""
    decoded = tokenizer.decode([])
    assert decoded == ""


def test_encode_unicode(tokenizer):
    """Test encoding Unicode text"""
    text = "Hello, 世界! 🌍"
    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)
    assert decoded == text
