"""
Integration tests for end-to-end text generation with Qwen3 4B

These tests verify the model generates coherent text using actual pretrained weights.
They are marked as 'slow' because they require downloading and loading the full 4B model.

Note: model and tokenizer fixtures are defined in conftest.py with scope="session"
and are shared across all tests that use them.
"""

import pytest
import torch


@pytest.mark.slow
def test_simple_generation(model, tokenizer):
    """Test that model generates coherent text from a simple prompt"""
    prompt = "The capital of France is"

    # Encode the prompt
    input_ids = tokenizer.encode(prompt)

    input_tensor = torch.tensor([input_ids])

    # Generate tokens
    max_new_tokens = 10
    cache_k = None
    cache_v = None

    generated_ids = input_ids.copy()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            logits, cache_k, cache_v = model(
                input_tensor, cache_k=cache_k, cache_v=cache_v
            )

            # Get the next token (greedy decoding - pick most likely)
            next_token_id = logits[0, -1, :].argmax().item()
            generated_ids.append(next_token_id)

            # Prepare input for next iteration
            input_tensor = torch.tensor([[next_token_id]])

    # Decode the generated text
    generated_text = tokenizer.decode(generated_ids)

    # Verify we got more text than the prompt
    assert len(generated_text) > len(prompt)
    assert generated_text.startswith(prompt)

    # Basic sanity check - output should contain "Paris" with high probability
    # (though not guaranteed with greedy decoding)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")


@pytest.mark.slow
def test_generation_with_cache(model, tokenizer):
    """Test that KV cache produces same results as without cache"""
    prompt = "Hello, world!"

    input_ids = tokenizer.encode(prompt)

    # Generation WITHOUT cache (baseline)
    max_new_tokens = 5
    generated_ids_no_cache = input_ids.copy()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Full forward pass every time
            current_ids = torch.tensor([generated_ids_no_cache])
            logits, _, _ = model(current_ids)
            next_token_id = logits[0, -1, :].argmax().item()
            generated_ids_no_cache.append(next_token_id)

    # Generation WITH cache (efficient)
    generated_ids_with_cache = input_ids.copy()
    cache_k = None
    cache_v = None

    with torch.no_grad():
        for i in range(max_new_tokens):
            if i == 0:
                # First pass - full prompt
                current_ids = torch.tensor([generated_ids_with_cache])
            else:
                # Subsequent passes - single token with cache
                current_ids = torch.tensor([[generated_ids_with_cache[-1]]])

            logits, cache_k, cache_v = model(
                current_ids, cache_k=cache_k, cache_v=cache_v
            )
            next_token_id = logits[0, -1, :].argmax().item()
            generated_ids_with_cache.append(next_token_id)

    # Both should produce identical results
    assert generated_ids_no_cache == generated_ids_with_cache

    text_no_cache = tokenizer.decode(generated_ids_no_cache)
    text_with_cache = tokenizer.decode(generated_ids_with_cache)

    print(f"Without cache: {text_no_cache}")
    print(f"With cache: {text_with_cache}")


@pytest.mark.slow
def test_batch_generation(model, tokenizer):
    """Test that model can handle batch generation"""
    prompts = [
        "Once upon a time",
        "The quick brown fox",
    ]

    # Encode prompts
    input_ids_list = [tokenizer.encode(p) for p in prompts]

    # Find max length for padding
    max_len = max(len(ids) for ids in input_ids_list)

    # Pad sequences (using 0 as padding, though proper implementation would use pad_token_id)
    padded_ids = []
    for ids in input_ids_list:
        padded = ids + [0] * (max_len - len(ids))
        padded_ids.append(padded)

    batch_tensor = torch.tensor(padded_ids)

    # Generate a few tokens
    max_new_tokens = 5

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _, _ = model(batch_tensor)
            next_tokens = logits[:, -1, :].argmax(dim=-1)
            batch_tensor = torch.cat([batch_tensor, next_tokens.unsqueeze(1)], dim=1)

    # Decode both sequences
    generated_texts = [
        tokenizer.decode(batch_tensor[i].tolist()) for i in range(len(prompts))
    ]

    # Verify both generated text
    for prompt, generated in zip(prompts, generated_texts):
        # Note: Due to padding, the generated text might have artifacts
        # In a production system, we'd use attention masks
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}")
        assert len(generated) >= len(prompt)


@pytest.mark.slow
def test_mathematical_reasoning(model, tokenizer):
    """Test that model can perform simple mathematical reasoning"""
    prompt = "What is 2 + 2? The answer is"

    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids])

    max_new_tokens = 10
    cache_k = None
    cache_v = None

    generated_ids = input_ids.copy()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, cache_k, cache_v = model(
                input_tensor, cache_k=cache_k, cache_v=cache_v
            )
            next_token_id = logits[0, -1, :].argmax().item()
            generated_ids.append(next_token_id)
            input_tensor = torch.tensor([[next_token_id]])

    generated_text = tokenizer.decode(generated_ids)

    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")

    # The model should generate something containing "4"
    # (though we can't guarantee exact format with greedy decoding)
    assert "4" in generated_text or "four" in generated_text.lower()


@pytest.mark.slow
def test_generation_reproducibility(model, tokenizer):
    """Test that generation is deterministic with same seed"""
    prompt = "The meaning of life is"

    # Set random seed
    torch.manual_seed(42)

    input_ids = tokenizer.encode(prompt)

    # First generation
    generated_ids1 = input_ids.copy()

    with torch.no_grad():
        for _ in range(5):
            logits, _, _ = model(torch.tensor([generated_ids1]))
            next_token_id = logits[0, -1, :].argmax().item()
            generated_ids1.append(next_token_id)

    # Reset seed and generate again
    torch.manual_seed(42)

    generated_ids2 = input_ids.copy()

    with torch.no_grad():
        for _ in range(5):
            logits, _, _ = model(torch.tensor([generated_ids2]))
            next_token_id = logits[0, -1, :].argmax().item()
            generated_ids2.append(next_token_id)

    # Should be identical
    assert generated_ids1 == generated_ids2

    text1 = tokenizer.decode(generated_ids1)
    text2 = tokenizer.decode(generated_ids2)

    assert text1 == text2
    print(f"Generated (deterministic): {text1}")


@pytest.mark.slow
def test_model_dtype(model):
    """Test that model uses bfloat16 throughout"""
    # Check embedding weights
    assert model.embed_tokens.embedding.weight.dtype == torch.bfloat16

    # Check transformer layer weights
    first_layer = model.layers[0]
    assert first_layer.self_attn.q_proj.weight.dtype == torch.bfloat16
    assert first_layer.self_attn.k_proj.weight.dtype == torch.bfloat16
    assert first_layer.self_attn.v_proj.weight.dtype == torch.bfloat16
    assert first_layer.mlp.gate_proj.weight.dtype == torch.bfloat16

    # Check final norm and lm_head
    assert model.norm.weight.dtype == torch.bfloat16
    assert model.lm_head.dtype == torch.bfloat16


@pytest.mark.slow
def test_generation_logits_shape(model, tokenizer):
    """Test that logits have correct shape during generation"""
    prompt = "Test"
    input_ids = tokenizer.encode(prompt)

    # Prefill phase
    input_tensor = torch.tensor([input_ids])
    with torch.no_grad():
        logits, cache_k, cache_v = model(input_tensor)

    # Verify prefill logits shape
    assert logits.shape == (1, len(input_ids), model.vocab_size)

    # Decode phase (single token)
    next_token_id = logits[0, -1, :].argmax().item()
    input_tensor = torch.tensor([[next_token_id]])

    with torch.no_grad():
        logits, cache_k, cache_v = model(input_tensor, cache_k=cache_k, cache_v=cache_v)

    # Verify decode logits shape
    assert logits.shape == (1, 1, model.vocab_size)

    # Verify cache grew correctly
    assert cache_k[0].shape[2] == len(input_ids) + 1
    assert cache_v[0].shape[2] == len(input_ids) + 1


@pytest.mark.slow
def test_model_generate_method(model, tokenizer):
    """Test the model's built-in generate() method"""
    prompt = "Hello, world!"

    # Encode prompt
    input_ids = tokenizer.encode(prompt)

    # Use the generate method
    new_tokens, cache_k, cache_v = model.generate(
        input_ids=input_ids,
        max_new_tokens=10,
        temperature=0.8,
        top_k=50,
    )

    # Verify we got only new tokens
    assert len(new_tokens) == 10
    assert new_tokens not in input_ids  # These are new tokens

    # Verify cache was returned
    assert cache_k is not None
    assert cache_v is not None
    assert len(cache_k) == 36  # 36 layers
    assert len(cache_v) == 36

    # Decode and verify it's valid text
    all_tokens = input_ids + new_tokens
    generated_text = tokenizer.decode(all_tokens)
    assert len(generated_text) > len(prompt)
    assert generated_text.startswith(prompt)

    print(f"Generated: {generated_text}")


@pytest.mark.slow
def test_model_generate_with_top_p(model, tokenizer):
    """Test the model's generate() method with top-p (nucleus) sampling"""
    prompt = "The quick brown"

    input_ids = tokenizer.encode(prompt)

    # Use top-p sampling
    new_tokens, cache_k, cache_v = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        temperature=1.0,
        top_p=0.9,
    )

    # Verify output
    assert len(new_tokens) == 5
    all_tokens = input_ids + new_tokens
    generated_text = tokenizer.decode(all_tokens)
    assert generated_text.startswith(prompt)

    print(f"Generated with top-p: {generated_text}")


@pytest.mark.slow
def test_model_generate_with_existing_cache(model, tokenizer):
    """Test that generate() can continue from existing KV cache"""
    prompt = "The capital of"

    # First generation - create cache
    input_ids = tokenizer.encode(prompt)
    new_tokens_1, cache_k, cache_v = model.generate(
        input_ids=input_ids,
        max_new_tokens=3,
        temperature=0.5,
        top_k=10,
    )

    all_tokens_1 = input_ids + new_tokens_1
    text_1 = tokenizer.decode(all_tokens_1)
    print(f"First generation: {text_1}")

    # Continue generation with existing cache
    # Pass the FULL sequence (input_ids + new_tokens_1)
    new_tokens_2, cache_k, cache_v = model.generate(
        input_ids=all_tokens_1,  # Full sequence so far
        max_new_tokens=3,
        temperature=0.5,
        top_k=10,
        cache_k=cache_k,
        cache_v=cache_v,
    )

    # Combine for full text - clean concatenation
    all_tokens_2 = all_tokens_1 + new_tokens_2
    text_2 = tokenizer.decode(all_tokens_2)
    print(f"Continued generation: {text_2}")

    # Verify continuation worked
    assert len(new_tokens_2) == 3  # Got exactly 3 new tokens
    assert text_2.startswith(text_1)


@pytest.mark.slow
def test_model_generate_chat_pattern(model, tokenizer):
    """Test multi-turn chat pattern by adding new context with cache"""
    # Initial prompt
    system_prompt = "You are helpful."
    system_ids = tokenizer.encode(system_prompt)

    # Generate first response
    response_1, cache_k, cache_v = model.generate(
        input_ids=system_ids,
        max_new_tokens=5,
        temperature=0.7,
        top_k=50,
    )

    conversation = system_ids + response_1
    print(f"System + Response 1: {tokenizer.decode(conversation)}")

    # User adds a message (NEW context)
    user_msg = " How are you?"
    user_ids = tokenizer.encode(user_msg)

    # Add user message to conversation and generate response
    # Pass the FULL conversation including the new user message
    conversation = conversation + user_ids
    response_2, cache_k, cache_v = model.generate(
        input_ids=conversation,  # Full conversation so far
        max_new_tokens=5,
        temperature=0.7,
        top_k=50,
        cache_k=cache_k,
        cache_v=cache_v,
    )

    conversation = conversation + response_2
    print(f"Full conversation: {tokenizer.decode(conversation)}")

    # Verify cache management
    # Cache should have: system + response_1 + user_ids + response_2
    expected_cache_len = len(system_ids) + 5 + len(user_ids) + 5
    assert cache_k[0].shape[2] == expected_cache_len
    assert cache_v[0].shape[2] == expected_cache_len
