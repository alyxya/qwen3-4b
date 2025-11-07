"""
Test generation with pure greedy (argmax) instead of temperature sampling
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

print("Generating with GREEDY decoding (pure argmax)...")
print("=" * 80)

batch_size = 1
cache_k = None
cache_v = None
next_input = input_tensor
generated_tokens = []

# Generate 30 tokens using PURE GREEDY (argmax)
for i in range(30):
    # Forward pass
    logits, cache_k, cache_v = model(next_input, cache_k, cache_v)

    # GREEDY: Just take argmax, no sampling
    next_token_id = torch.argmax(logits[0, -1, :]).item()
    next_token = torch.tensor([[next_token_id]])
    generated_tokens.append(next_token_id)

    # Decode this single token
    token_str = tokenizer.decode([next_token_id])
    token_repr = repr(token_str)

    # Print debug info
    print(f"Step {i+1:2d}: token_id={next_token_id:6d}, token={token_repr:20s}")

    # Set up next input
    next_input = next_token

print("\n" + "=" * 80)
print("FULL GENERATED TEXT (with greedy decoding)")
print("=" * 80)
full_text = tokenizer.decode(generated_tokens)
print(full_text)
print("=" * 80)

# Try to parse as tool call
import json

def parse_tool_call(text):
    if "<tool_call>" not in text or "</tool_call>" not in text:
        return None
    start = text.find("<tool_call>") + len("<tool_call>")
    end = text.find("</tool_call>")
    tool_text = text[start:end].strip()
    try:
        return json.loads(tool_text)
    except json.JSONDecodeError:
        return None

tool_call = parse_tool_call(full_text + "</tool_call>")  # Add closing tag for testing
print(f"\nTool call parsing: {tool_call if tool_call else 'FAILED'}")
