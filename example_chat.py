#!/usr/bin/env python3
"""
Example: Using the tokenizer with special tokens and chat templates

This script demonstrates:
1. Loading special tokens
2. Using apply_chat_template() for conversation formatting
3. Encoding/decoding with special tokens
"""

from src.tokenizer import Tokenizer


def main():
    print("Loading tokenizer...")
    tokenizer = Tokenizer()

    # Show loaded special tokens
    print(f"\nLoaded {len(tokenizer.special_tokens)} special tokens")
    print(f"  <|im_start|> ID: {tokenizer.im_start_id}")
    print(f"  <|im_end|> ID: {tokenizer.im_end_id}")
    print(f"  <|endoftext|> ID: {tokenizer.endoftext_id}")

    # Example 1: Using chat template
    print("\n" + "=" * 60)
    print("Example 1: Chat Template")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    # Get formatted text (not tokenized)
    formatted_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    print("\nFormatted chat template:")
    print(formatted_text)

    # Get token IDs
    token_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    print(f"Token count: {len(token_ids)}")
    print(f"First 10 token IDs: {token_ids[:10]}")

    # Example 2: Encoding with special tokens
    print("\n" + "=" * 60)
    print("Example 2: Encoding with Special Tokens")
    print("=" * 60)

    text_with_special = "<|im_start|>user\nHello, world!<|im_end|>"
    print(f"Text: {text_with_special}")

    # Encode - special tokens are automatically recognized
    token_ids = tokenizer.encode(text_with_special)
    print(f"Token IDs: {token_ids}")
    print(f"Token count: {len(token_ids)}")
    print(
        f"Note: <|im_start|> and <|im_end|> are encoded as single tokens "
        f"({tokenizer.im_start_id} and {tokenizer.im_end_id})"
    )

    # Example 3: Decoding
    print("\n" + "=" * 60)
    print("Example 3: Decoding")
    print("=" * 60)

    # Decode back to text
    decoded = tokenizer.decode(token_ids)
    print(f"Decoded text: {decoded}")
    print(f"Matches original? {decoded == text_with_special}")

    # Example 4: Multi-turn conversation
    print("\n" + "=" * 60)
    print("Example 4: Multi-turn Conversation")
    print("=" * 60)

    conversation = [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "2 + 2 equals 4."},
        {"role": "user", "content": "What about 3 + 3?"},
    ]

    formatted_conv = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    print("Multi-turn conversation format:")
    print(formatted_conv)

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
