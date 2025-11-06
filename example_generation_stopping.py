#!/usr/bin/env python3
"""
Example: Using stop tokens during generation

This shows how <|endoftext|> and <|im_end|> are used as stop tokens
to control when generation should stop.
"""

from src.tokenizer import Tokenizer


def simulate_generation_with_stop_tokens():
    """
    Simulate what happens during model generation when stop tokens appear
    """
    tokenizer = Tokenizer()

    print("=" * 70)
    print("Stop Token Example")
    print("=" * 70)
    print()

    # Scenario 1: Generation stops at <|im_end|>
    print("### Scenario 1: Chat Response ###")
    print()
    print("Prompt: <|im_start|>user\\nHello!<|im_end|><|im_start|>assistant\\n")
    print()
    print("Model generates tokens one by one:")

    # Simulated generation (what model might produce)
    generated_tokens = [
        9707,      # "Hello"
        0,         # "!"
        1768,      # " How"
        646,       # " can"
        358,       # " I"
        1492,      # " help"
        30,        # "?"
        151645,    # <|im_end|>  ← STOP HERE!
        # Model would generate more, but we stop at im_end
    ]

    print()
    for i, token_id in enumerate(generated_tokens):
        decoded = tokenizer.decode([token_id])

        # Check if it's a stop token
        if token_id == tokenizer.im_end_id:
            print(f"  Step {i+1}: Generated token {token_id} = <|im_end|>")
            print(f"          → This is a STOP TOKEN for chat!")
            print(f"          → Stop generation and return accumulated text")
            break
        else:
            print(f"  Step {i+1}: Generated token {token_id} = {repr(decoded)}")

    # Decode only the tokens BEFORE the stop token
    response_tokens = generated_tokens[:generated_tokens.index(tokenizer.im_end_id)]
    response = tokenizer.decode(response_tokens)
    print()
    print(f"Final response (without stop token): {repr(response)}")
    print()

    # Scenario 2: Generation stops at <|endoftext|>
    print()
    print("### Scenario 2: Raw Text Generation ###")
    print()
    print("Prompt: 'Once upon a time'")
    print()
    print("Model generates tokens one by one:")

    generated_tokens_2 = [
        1052,      # " there"
        572,       # " was"
        264,       # " a"
        1896,      # " princess"
        13,        # "."
        151643,    # <|endoftext|>  ← STOP HERE!
    ]

    print()
    for i, token_id in enumerate(generated_tokens_2):
        decoded = tokenizer.decode([token_id])

        if token_id == tokenizer.endoftext_id:
            print(f"  Step {i+1}: Generated token {token_id} = <|endoftext|>")
            print(f"          → This is a STOP TOKEN for raw text!")
            print(f"          → Model has finished the story")
            break
        else:
            print(f"  Step {i+1}: Generated token {token_id} = {repr(decoded)}")

    response_tokens_2 = generated_tokens_2[:generated_tokens_2.index(tokenizer.endoftext_id)]
    response_2 = tokenizer.decode(response_tokens_2)
    print()
    print(f"Final text (without stop token): {repr(response_2)}")
    print()


def show_stop_token_configuration():
    """Show how you'd configure stop tokens for generation"""
    tokenizer = Tokenizer()

    print()
    print("=" * 70)
    print("Configuring Stop Tokens")
    print("=" * 70)
    print()

    print("For chat mode (using chat template):")
    print(f"  stop_tokens = [{tokenizer.im_end_id}]  # Stop at <|im_end|>")
    print()

    print("For raw text generation:")
    print(f"  stop_tokens = [{tokenizer.endoftext_id}]  # Stop at <|endoftext|>")
    print()

    print("For safety (stop at multiple tokens):")
    print(f"  stop_tokens = [")
    print(f"      {tokenizer.im_end_id},    # <|im_end|>")
    print(f"      {tokenizer.endoftext_id}, # <|endoftext|>")
    print(f"  ]")
    print()

    print("In your generation loop:")
    print("""
    stop_tokens = [tokenizer.im_end_id, tokenizer.endoftext_id]

    generated = []
    for _ in range(max_tokens):
        next_token = model.generate_next_token(...)

        if next_token in stop_tokens:
            break  # Stop generation!

        generated.append(next_token)

    response = tokenizer.decode(generated)  # Don't include stop token
    """)


def main():
    simulate_generation_with_stop_tokens()
    show_stop_token_configuration()

    print()
    print("=" * 70)
    print("Key Takeaways")
    print("=" * 70)
    print()
    print("1. Stop tokens are checked DURING generation, not decoding")
    print("2. When you see a stop token, you stop generating more tokens")
    print("3. You DON'T include the stop token in the final decoded text")
    print("4. <|im_end|> is for chat, <|endoftext|> is for raw text")
    print("5. Different contexts use different stop tokens")
    print()


if __name__ == "__main__":
    main()
