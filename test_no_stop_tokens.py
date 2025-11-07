"""
Quick test: Generate with and without stop tokens
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

print("=" * 80)
print("TEST 1: Generate WITHOUT stop tokens")
print("=" * 80)
torch.manual_seed(42)
new_tokens1, _, _ = model.generate(
    input_ids=input_tensor,
    max_new_tokens=50,  # Reduced to 50 for faster test
    temperature=0.01,
    stop_token_ids=None,  # NO STOP TOKENS
)

response1 = tokenizer.decode(new_tokens1[0].tolist())
print(f"Generated {len(new_tokens1[0])} tokens")
print("\nResponse:")
print("-" * 40)
print(response1)
print("-" * 40)

print("\n" + "=" * 80)
print("TEST 2: Generate WITH stop tokens")
print("=" * 80)
torch.manual_seed(42)
new_tokens2, _, _ = model.generate(
    input_ids=input_tensor,
    max_new_tokens=50,
    temperature=0.01,
    stop_token_ids=[tokenizer.im_end_id, tokenizer.endoftext_id],
)

response2 = tokenizer.decode(new_tokens2[0].tolist())
print(f"Generated {len(new_tokens2[0])} tokens")
print("\nResponse:")
print("-" * 40)
print(response2)
print("-" * 40)

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
if response1 == response2:
    print("✓ Responses are identical")
else:
    print("✗ Responses differ!")
    print(f"  Without stop tokens: {len(new_tokens1[0])} tokens")
    print(f"  With stop tokens: {len(new_tokens2[0])} tokens")
