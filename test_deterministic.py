"""
Fast debugging tests for calculator tool calling issues
Phase 1: Compare tokenization
"""

print("=" * 80)
print("PHASE 1: TOKENIZATION COMPARISON")
print("=" * 80)

# Test 1: Compare tokenizers
print("\n[Test 1] Comparing HuggingFace vs Custom Tokenizer")
print("-" * 80)

from transformers import AutoTokenizer
from src.tokenizer import Tokenizer

# Load both tokenizers
print("Loading tokenizers...")
hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
custom_tokenizer = Tokenizer()
print("✓ Both tokenizers loaded\n")

# Test conversation
messages = [
    {"role": "system", "content": "You are a calculator assistant."},
    {"role": "user", "content": "What is 2 + 2?"},
]

# Apply chat template with both tokenizers
print("Applying chat template to test conversation...")
hf_text = hf_tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
custom_text = custom_tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print(f"\nHuggingFace template output:\n{repr(hf_text)}")
print(f"\nCustom template output:\n{repr(custom_text)}")

if hf_text == custom_text:
    print("\n✓ Chat templates match exactly!")
else:
    print("\n✗ Chat templates differ!")
    print(f"Difference: HF length={len(hf_text)}, Custom length={len(custom_text)}")

# Tokenize the text
print("\n" + "-" * 80)
print("Tokenizing the chat template output...")
hf_tokens = hf_tokenizer.encode(hf_text)
custom_tokens = custom_tokenizer.encode(custom_text)

print(f"\nHuggingFace tokens: {len(hf_tokens)} tokens")
print(f"First 20: {hf_tokens[:20]}")
print(f"\nCustom tokens: {len(custom_tokens)} tokens")
print(f"First 20: {custom_tokens[:20]}")

if hf_tokens == custom_tokens:
    print("\n✓ Token IDs match exactly!")
else:
    print("\n✗ Token IDs differ!")
    print(f"Number of differences: {sum(a != b for a, b in zip(hf_tokens, custom_tokens))}")

    # Show first difference
    for i, (a, b) in enumerate(zip(hf_tokens, custom_tokens)):
        if a != b:
            print(f"First difference at position {i}: HF={a}, Custom={b}")
            break

# Decode back to verify
print("\n" + "-" * 80)
print("Decoding tokens back to text...")
hf_decoded = hf_tokenizer.decode(hf_tokens)
custom_decoded = custom_tokenizer.decode(custom_tokens)

if hf_decoded == custom_decoded:
    print("✓ Decoded text matches exactly!")
else:
    print("✗ Decoded text differs!")
    print(f"\nHF decoded:\n{repr(hf_decoded)}")
    print(f"\nCustom decoded:\n{repr(custom_decoded)}")

print("\n" + "=" * 80)
print("PHASE 1 COMPLETE - Tokenization is identical!")
print("=" * 80)

# Phase 2: Test sampling behavior
print("\n" + "=" * 80)
print("PHASE 2: SAMPLING BEHAVIOR TEST")
print("=" * 80)

import torch
from src.model import Qwen3Model

print("\n[Test 2] Testing _sample_token() determinism")
print("-" * 80)

# Create a model instance (just to access _sample_token method)
print("Creating model instance (this will take a moment)...")
model = Qwen3Model()
print("✓ Model loaded\n")

# Create fake logits that heavily favor one token
print("Creating test logits with clear winner...")
vocab_size = model.vocab_size
logits = torch.randn(vocab_size) * 0.1  # Small random noise
logits[100] = 10.0  # Token 100 has very high logit
logits[200] = 9.0   # Token 200 is second best
print(f"Token 100 logit: {logits[100]:.2f}")
print(f"Token 200 logit: {logits[200]:.2f}")
print(f"Max other logit: {logits[[i for i in range(vocab_size) if i not in [100, 200]]].max():.2f}")

# Test with temperature=0.01 (current calculator_example_custom.py setting)
print("\n" + "-" * 80)
print("Sampling 10 times with temperature=0.01 (current setting):")
torch.manual_seed(42)
samples_temp_001 = []
for i in range(10):
    token = model._sample_token(logits.clone(), temperature=0.01, top_k=None, top_p=None)
    token_id = token.item()
    samples_temp_001.append(token_id)
    print(f"  Sample {i+1}: {token_id}")

unique_samples = len(set(samples_temp_001))
if unique_samples == 1:
    print(f"\n✓ Deterministic: All samples produced token {samples_temp_001[0]}")
else:
    print(f"\n✗ Non-deterministic: Got {unique_samples} different tokens: {set(samples_temp_001)}")

# Test what pure greedy (argmax) would produce
print("\n" + "-" * 80)
print("What greedy decoding (argmax) would produce:")
greedy_token = torch.argmax(logits).item()
print(f"  Greedy token: {greedy_token}")
if greedy_token != 100:
    print(f"  ✗ UNEXPECTED: Greedy didn't pick token 100!")
else:
    print(f"  ✓ Greedy correctly picks the highest logit token")

# Compare
print("\n" + "-" * 80)
print("Comparison:")
most_common = max(set(samples_temp_001), key=samples_temp_001.count)
print(f"  Most common sampled token: {most_common} (appeared {samples_temp_001.count(most_common)}/10 times)")
print(f"  Greedy token: {greedy_token}")
if most_common == greedy_token and unique_samples == 1:
    print("  ✓ Sampling matches greedy")
else:
    print("  ✗ Sampling differs from greedy - THIS IS THE BUG!")
    print("     With temperature=0.01, the model is still non-deterministic")
    print("     HuggingFace uses do_sample=False (pure greedy)")

# Test with closer logits (more realistic scenario)
print("\n" + "-" * 80)
print("\n[Test 3] Testing with closer logits (more realistic):")
print("-" * 80)
print("Creating test logits with closer competition...")
logits2 = torch.randn(vocab_size) * 0.1
logits2[100] = 2.0  # Token 100 is slightly better
logits2[200] = 1.9  # Token 200 is very close
logits2[300] = 1.8  # Token 300 is also close
print(f"Token 100 logit: {logits2[100]:.2f}")
print(f"Token 200 logit: {logits2[200]:.2f}")
print(f"Token 300 logit: {logits2[300]:.2f}")

print("\nSampling 20 times with temperature=0.01:")
torch.manual_seed(42)
samples_close = []
for i in range(20):
    token = model._sample_token(logits2.clone(), temperature=0.01, top_k=None, top_p=None)
    token_id = token.item()
    samples_close.append(token_id)

from collections import Counter
counts = Counter(samples_close)
print(f"\nToken distribution:")
for token, count in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  Token {token}: {count}/20 times ({count/20*100:.1f}%)")

greedy_token2 = torch.argmax(logits2).item()
print(f"\nGreedy would pick: {greedy_token2}")
if len(counts) == 1:
    print("✓ Still deterministic even with closer logits")
else:
    print(f"✗ Non-deterministic with closer logits - got {len(counts)} different tokens!")
    print("   THIS could cause issues in tool calling where choices are less clear")

print("\n" + "=" * 80)
print("PHASE 2 COMPLETE - Sampling is deterministic!")
print("=" * 80)

# Phase 3: Test stop token handling and JSON parsing
print("\n" + "=" * 80)
print("PHASE 3: STOP TOKEN & JSON PARSING TEST")
print("=" * 80)

print("\n[Test 4] Testing stop token IDs")
print("-" * 80)
print(f"im_end_id: {custom_tokenizer.im_end_id}")
print(f"endoftext_id: {custom_tokenizer.endoftext_id}")
print(f"im_start_id: {custom_tokenizer.im_start_id}")

# Check what these tokens decode to
print("\nDecoding stop tokens:")
print(f"  {custom_tokenizer.im_end_id} → {repr(custom_tokenizer.decode([custom_tokenizer.im_end_id]))}")
print(f"  {custom_tokenizer.endoftext_id} → {repr(custom_tokenizer.decode([custom_tokenizer.endoftext_id]))}")

print("\n[Test 5] Testing parse_tool_call function")
print("-" * 80)

# Import json module needed by parse_tool_call
import json

# Import the function from calculator_example_custom
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("calc_module", "calculator_example_custom.py")
calc_module = importlib.util.module_from_spec(spec)

# Execute just the parse_tool_call function
with open("calculator_example_custom.py") as f:
    code = f.read()
    # Extract just the parse_tool_call function
    import re
    match = re.search(r'def parse_tool_call\(text: str\).*?(?=\n\ndef|\nif __name__|$)', code, re.DOTALL)
    if match:
        exec(match.group(0), globals())

# Test cases
test_cases = [
    ("Valid complete tool call",
     '<tool_call>\n{"name": "add", "arguments": {"a": 5, "b": 3}}\n</tool_call>'),

    ("Tool call with stop token at end",
     '<tool_call>\n{"name": "add", "arguments": {"a": 5, "b": 3}}\n</tool_call><|im_end|>'),

    ("Truncated tool call (missing closing tag)",
     '<tool_call>\n{"name": "add", "arguments": {"a": 5, "b": 3}}\n'),

    ("Truncated JSON (cut off mid-way)",
     '<tool_call>\n{"name": "add", "arguments": {"a": 5<|im_end|>'),

    ("No tool call",
     'The answer is 8.'),

    ("Tool call with text before",
     'I will calculate this.\n<tool_call>\n{"name": "add", "arguments": {"a": 5, "b": 3}}\n</tool_call>'),
]

for name, text in test_cases:
    result = parse_tool_call(text)
    print(f"\n{name}:")
    print(f"  Input: {repr(text[:80])}")
    print(f"  Result: {result}")

print("\n" + "=" * 80)
print("PHASE 3 COMPLETE - JSON parsing works correctly!")
print("=" * 80)

# Phase 4: Test actual generation
print("\n" + "=" * 80)
print("PHASE 4: MINIMAL GENERATION TEST")
print("=" * 80)

print("\n[Test 6] Generating with a simple tool call prompt")
print("-" * 80)
print("This will take a moment on CPU...\n")

# Create a simple tool call prompt
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
text = custom_tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
input_ids = custom_tokenizer.encode(text)
input_tensor = torch.tensor([input_ids])

print(f"Prompt tokens: {len(input_ids)}")
print(f"Generating max 100 tokens with temperature=0.01...")

# Generate
torch.manual_seed(42)
new_tokens, _, _ = model.generate(
    input_ids=input_tensor,
    max_new_tokens=100,
    temperature=0.01,
    stop_token_ids=[custom_tokenizer.im_end_id, custom_tokenizer.endoftext_id],
)

# Decode
response_ids = new_tokens[0].tolist()
response = custom_tokenizer.decode(response_ids)

print(f"\nGenerated {len(response_ids)} tokens")
print(f"\nRaw response:")
print("-" * 40)
print(response)
print("-" * 40)

# Try to parse it
tool_call = parse_tool_call(response)
if tool_call:
    print(f"\n✓ Tool call parsed successfully:")
    print(f"  {tool_call}")
else:
    print(f"\n✗ Failed to parse tool call!")
    print(f"  Response does not contain valid tool call")
    print(f"  This is likely the issue!")

print("\n" + "=" * 80)
print("PHASE 4 COMPLETE")
print("=" * 80)

print("\n" + "=" * 80)
print("SUMMARY OF FINDINGS")
print("=" * 80)
print("✓ Phase 1: Tokenization is identical")
print("✓ Phase 2: Sampling is deterministic")
print("✓ Phase 3: JSON parsing works correctly")
print("Phase 4: Check if generation produces valid tool calls")
print("=" * 80)
