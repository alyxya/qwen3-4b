"""
Debug generation step-by-step to see what tokens are being produced
"""

import torch
from src.model import Qwen3Model
from src.tokenizer import Tokenizer

print("Loading model and tokenizer...")
model = Qwen3Model()
tokenizer = Tokenizer()
print("✓ Loaded\n")

simple_system = """You are a calculator. Use tools for calculations.
Available tools:
- add(a, b): Add two numbers

To call a tool, respond with:
<tool_call>
{"name": "add", "arguments": {"a": value1, "b": value2}}
</tool_call>"""

simple_question = "What is 2 + 3?"

messages = [
    {"role": "system", "content": simple_system},
    {"role": "user", "content": simple_question},
]

# Apply chat template
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
input_ids = tokenizer.encode(text)
input_tensor = torch.tensor([input_ids])

print(f"Prompt has {len(input_ids)} tokens\n")
print("=" * 80)
print("GENERATING TOKEN BY TOKEN (first 30 tokens)")
print("=" * 80)

torch.manual_seed(42)
batch_size = 1
cache_k = None
cache_v = None
next_input = input_tensor
generated_tokens = []

for i in range(30):
    # Forward pass
    logits, cache_k, cache_v = model(next_input, cache_k, cache_v)

    # Sample next token
    next_token = model._sample_token(logits[0, -1, :], temperature=0.01, top_k=None, top_p=None)
    token_id = next_token.item()
    generated_tokens.append(token_id)

    # Decode this single token
    token_str = tokenizer.decode([token_id])
    token_repr = repr(token_str)

    # Print debug info
    print(f"Step {i+1:2d}: token_id={token_id:6d}, token={token_repr:20s}, cache_len={cache_k[0].shape[2]}")

    # Set up next input
    next_input = next_token

print("\n" + "=" * 80)
print("FULL GENERATED TEXT")
print("=" * 80)
full_text = tokenizer.decode(generated_tokens)
print(full_text)
print("=" * 80)
