"""
Test generation WITHOUT KV caching (like HuggingFace does)
Regenerate from full context each time
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

print(f"Prompt has {len(input_ids)} tokens\n")
print("=" * 80)
print("GENERATING WITHOUT KV CACHE (regenerate full context each time)")
print("=" * 80)

generated_tokens = []

# Generate 30 tokens WITHOUT caching - regenerate full context each time
for i in range(30):
    # Build full sequence: prompt + all generated tokens so far
    full_sequence = input_ids + generated_tokens
    full_tensor = torch.tensor([full_sequence])

    # Forward pass on FULL sequence (no cache)
    with torch.no_grad():
        logits, _, _ = model(full_tensor, cache_k=None, cache_v=None)

    # Greedy: take argmax from last position
    next_token_id = torch.argmax(logits[0, -1, :]).item()
    generated_tokens.append(next_token_id)

    # Decode this single token
    token_str = tokenizer.decode([next_token_id])
    token_repr = repr(token_str)

    # Print debug info
    print(f"Step {i+1:2d}: token_id={next_token_id:6d}, token={token_repr:20s}, full_seq_len={len(full_sequence)+1}")

print("\n" + "=" * 80)
print("FULL GENERATED TEXT (no cache)")
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
