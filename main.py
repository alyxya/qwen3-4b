"""
Simple text generation example using Qwen3 4B

This script demonstrates how to use the model for basic text generation.
"""

import torch
from src.model import Qwen3Model
from src.tokenizer import Tokenizer


def generate_text(
    model: Qwen3Model,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> str:
    """
    Generate text from a prompt using the Qwen3 model

    Args:
        model: The Qwen3Model instance
        tokenizer: The Tokenizer instance
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (1.0 = no change, < 1.0 = more deterministic, > 1.0 = more random)
        top_k: If set, only sample from top k tokens (None = no filtering)

    Returns:
        Generated text including the prompt
    """
    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    print(f"Prompt tokens: {len(input_ids)}")

    # Initialize cache
    cache_k = None
    cache_v = None

    # Track all generated token IDs
    generated_ids = input_ids.copy()

    # Generate tokens one at a time
    with torch.no_grad():
        for i in range(max_new_tokens):
            # Prepare input tensor (only the new token for decode phase)
            if i == 0:
                # Prefill phase - process entire prompt
                input_tensor = torch.tensor([generated_ids])
            else:
                # Decode phase - process single token with cache
                input_tensor = torch.tensor([[generated_ids[-1]]])

            # Forward pass
            logits, cache_k, cache_v = model(input_tensor, cache_k=cache_k, cache_v=cache_v)

            # Get logits for the last token
            next_token_logits = logits[0, -1, :]  # (vocab_size,)

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                # Set all other logits to -inf
                next_token_logits = torch.full_like(next_token_logits, float("-inf"))
                next_token_logits[top_k_indices] = top_k_logits

            # Sample from the distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

            # Add to generated sequence
            generated_ids.append(next_token_id)

            # Optional: decode and print token as we generate
            # (useful for seeing real-time generation)
            # token_text = tokenizer.decode([next_token_id])
            # print(token_text, end="", flush=True)

    # Decode the full generated sequence
    generated_text = tokenizer.decode(generated_ids)
    return generated_text


def main():
    """Main function demonstrating text generation"""
    print("=" * 80)
    print("Qwen3 4B Text Generation Demo")
    print("=" * 80)
    print()

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = Qwen3Model()
    tokenizer = Tokenizer()
    print("Model loaded successfully!")
    print()

    # Example prompts
    prompts = [
        "The capital of France is",
        "Once upon a time, in a faraway land,",
        "The meaning of life is",
        "To be or not to be,",
    ]

    # Generate text for each prompt
    for i, prompt in enumerate(prompts, 1):
        print(f"Example {i}/{len(prompts)}")
        print("-" * 80)
        print(f"Prompt: {prompt}")
        print()

        # Generate with greedy decoding (temperature=0.0 approximated by low value)
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=30,
            temperature=0.7,  # Slightly less random
            top_k=50,  # Sample from top 50 tokens
        )

        print(f"Generated text:")
        print(generated)
        print()
        print()


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    main()
