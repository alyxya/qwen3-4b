"""
Simple text generation example using Qwen3 4B

This script demonstrates how to use the model for basic text generation.
"""

import torch
from src.model import Qwen3Model
from src.tokenizer import Tokenizer


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

        # Encode the prompt
        input_ids = tokenizer.encode(prompt)
        print(f"Prompt tokens: {len(input_ids)}")

        # Generate using the model's generate method
        new_tokens, _, _ = model.generate(
            input_ids=input_ids,
            max_new_tokens=30,
            temperature=0.7,  # Slightly less random
            top_k=50,  # Sample from top 50 tokens
        )

        # Combine prompt and generated tokens
        all_tokens = input_ids + new_tokens
        generated_text = tokenizer.decode(all_tokens)

        print(f"Generated text:")
        print(generated_text)
        print()
        print()


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    main()
