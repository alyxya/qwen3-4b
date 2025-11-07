"""
Test after RMSNorm fix - fresh model loading
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model import Qwen3Model
from src.tokenizer import Tokenizer

print("Loading models (with fixed RMSNorm)...")
hf_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    dtype=torch.bfloat16,
    device_map="cpu"
)
hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

custom_model = Qwen3Model()
custom_tokenizer = Tokenizer()
print("✓ Models loaded\n")

# Test prompt
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

# Tokenize
hf_text = hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
custom_text = custom_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

hf_tokens = hf_tokenizer.encode(hf_text)
custom_tokens = custom_tokenizer.encode(custom_text)

print(f"Tokenization matches: {hf_tokens == custom_tokens}")

# Forward pass
hf_input = torch.tensor([hf_tokens])
custom_input = torch.tensor([custom_tokens])

print("\nRunning forward passes...")
with torch.no_grad():
    hf_outputs = hf_model(hf_input)
    hf_logits = hf_outputs.logits

    custom_logits, _, _ = custom_model(custom_input)

# Compare logits
hf_last_logits = hf_logits[0, -1, :]
custom_last_logits = custom_logits[0, -1, :]

hf_next_token = torch.argmax(hf_last_logits).item()
custom_next_token = torch.argmax(custom_last_logits).item()

logits_diff = (hf_last_logits - custom_last_logits).abs()
max_diff = logits_diff.max().item()
mean_diff = logits_diff.mean().item()

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"HF predicts: {hf_next_token} ({repr(hf_tokenizer.decode([hf_next_token]))})")
print(f"Custom predicts: {custom_next_token} ({repr(custom_tokenizer.decode([custom_next_token]))})")
print(f"\nMax logit difference: {max_diff:.6e}")
print(f"Mean logit difference: {mean_diff:.6e}")

if hf_next_token == custom_next_token and max_diff < 0.1:
    print("\n✓✓✓ SUCCESS! Models now match!")
else:
    print(f"\n✗ Still different")

    print("\nTop 5 from HF:")
    hf_top5 = torch.topk(hf_last_logits, 5)
    for logit, token_id in zip(hf_top5.values, hf_top5.indices):
        print(f"  {token_id.item():6d}: {logit.item():8.3f} {repr(hf_tokenizer.decode([token_id.item()]))}")

    print("\nTop 5 from Custom:")
    custom_top5 = torch.topk(custom_last_logits, 5)
    for logit, token_id in zip(custom_top5.values, custom_top5.indices):
        print(f"  {token_id.item():6d}: {logit.item():8.3f} {repr(custom_tokenizer.decode([token_id.item()]))}")
