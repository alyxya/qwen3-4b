"""Tests for tokenizer"""

import pytest


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


def test_special_tokens_loaded(tokenizer):
    """Test that special tokens are loaded properly"""
    assert len(tokenizer.special_tokens) > 0
    assert "<|im_start|>" in tokenizer.special_tokens
    assert "<|im_end|>" in tokenizer.special_tokens
    assert "<|endoftext|>" in tokenizer.special_tokens


def test_special_token_properties(tokenizer):
    """Test special token property accessors"""
    assert tokenizer.im_start_id > 0
    assert tokenizer.im_end_id > 0
    assert tokenizer.endoftext_id > 0
    assert tokenizer.im_start_id == tokenizer.special_tokens["<|im_start|>"]
    assert tokenizer.im_end_id == tokenizer.special_tokens["<|im_end|>"]
    assert tokenizer.endoftext_id == tokenizer.special_tokens["<|endoftext|>"]


def test_special_token_strings(tokenizer):
    """Test special token string accessors"""
    assert tokenizer.im_start == "<|im_start|>"
    assert tokenizer.im_end == "<|im_end|>"
    assert tokenizer.endoftext == "<|endoftext|>"

    # Verify strings match what's in special_tokens dict
    assert tokenizer.im_start in tokenizer.special_tokens
    assert tokenizer.im_end in tokenizer.special_tokens
    assert tokenizer.endoftext in tokenizer.special_tokens

    # Verify ID and string correspond to each other
    assert tokenizer.special_tokens[tokenizer.im_start] == tokenizer.im_start_id
    assert tokenizer.special_tokens[tokenizer.im_end] == tokenizer.im_end_id
    assert tokenizer.special_tokens[tokenizer.endoftext] == tokenizer.endoftext_id


def test_encode_with_special_tokens(tokenizer):
    """Test encoding text containing special tokens"""
    text = "<|im_start|>user\nHello!<|im_end|>"
    token_ids = tokenizer.encode(text)

    # Should contain the special token IDs
    assert tokenizer.im_start_id in token_ids
    assert tokenizer.im_end_id in token_ids


def test_encode_special_tokens_always_recognized(tokenizer):
    """Test that special tokens are always recognized"""
    text = "<|im_start|>user\nHello!<|im_end|>"

    token_ids = tokenizer.encode(text)

    # Special tokens should be recognized as single tokens
    assert tokenizer.im_start_id in token_ids
    assert tokenizer.im_end_id in token_ids

    # Should be much shorter than if they were encoded as regular text
    assert len(token_ids) < 15  # Would be ~20+ if special tokens weren't recognized


def test_decode_with_special_tokens(tokenizer):
    """Test decoding preserves special tokens"""
    text = "<|im_start|>user\nHello, world!<|im_end|>"
    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)

    assert decoded == text


def test_apply_chat_template(tokenizer):
    """Test chat template formatting"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    # Get formatted text (not tokenized)
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)

    assert isinstance(formatted, str)
    assert "<|im_start|>system" in formatted
    assert "You are a helpful assistant." in formatted
    assert "<|im_end|>" in formatted
    assert "<|im_start|>user" in formatted
    assert "Hello!" in formatted
    assert "<|im_start|>assistant" in formatted  # generation prompt


def test_apply_chat_template_tokenized(tokenizer):
    """Test chat template returns token IDs when tokenize=True"""
    messages = [
        {"role": "user", "content": "Hello!"},
    ]

    token_ids = tokenizer.apply_chat_template(messages, tokenize=True)

    assert isinstance(token_ids, list)
    assert all(isinstance(tid, int) for tid in token_ids)
    assert tokenizer.im_start_id in token_ids
    assert tokenizer.im_end_id in token_ids


def test_apply_chat_template_no_generation_prompt(tokenizer):
    """Test chat template without generation prompt"""
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False
    )

    # Should end with </im_end>, not with <im_start>assistant
    assert formatted.rstrip().endswith("<|im_end|>")
    assert not formatted.endswith("<|im_start|>assistant\n")


def test_chat_template_roundtrip(tokenizer):
    """Test that chat template output can be decoded correctly"""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
    ]

    token_ids = tokenizer.apply_chat_template(messages, tokenize=True)
    decoded = tokenizer.decode(token_ids)

    # Verify the structure is preserved
    assert "<|im_start|>system" in decoded
    assert "You are helpful." in decoded
    assert "<|im_start|>user" in decoded
    assert "What is 2+2?" in decoded
    assert "<|im_start|>assistant" in decoded
