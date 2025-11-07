"""
Compare HuggingFace vs Custom Model systematically:
1. Same tokenized prompt?
2. Same logit distributions?
3. Precision differences?
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model import Qwen3Model
from src.tokenizer import Tokenizer

print("=" * 80)
print("LOADING MODELS")
print("=" * 80)
print("Loading HuggingFace model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    dtype=torch.bfloat16,
    device_map="cpu"  # Force CPU for fair comparison
)
hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
print("✓ HuggingFace loaded")

print("\nLoading Custom model...")
custom_model = Qwen3Model()
custom_tokenizer = Tokenizer()
print("✓ Custom loaded")

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

# TEST 1: Same tokenized prompt?
print("\n" + "=" * 80)
print("TEST 1: COMPARING TOKENIZED PROMPTS")
print("=" * 80)

hf_text = hf_tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
custom_text = custom_tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print(f"Template text matches: {hf_text == custom_text}")

hf_tokens = hf_tokenizer.encode(hf_text)
custom_tokens = custom_tokenizer.encode(custom_text)

print(f"HF tokens: {len(hf_tokens)}")
print(f"Custom tokens: {len(custom_tokens)}")
print(f"Tokens match: {hf_tokens == custom_tokens}")

if hf_tokens != custom_tokens:
    print("\n✗ TOKENS DIFFER!")
    for i, (a, b) in enumerate(zip(hf_tokens, custom_tokens)):
        if a != b:
            print(f"  First diff at position {i}: HF={a}, Custom={b}")
            break
else:
    print("✓ Tokenized prompts are identical")

# TEST 2: Compare logits from first forward pass
print("\n" + "=" * 80)
print("TEST 2: COMPARING LOGITS FROM FIRST FORWARD PASS")
print("=" * 80)

hf_input = torch.tensor([hf_tokens])
custom_input = torch.tensor([custom_tokens])

print("Running forward pass on HuggingFace model...")
with torch.no_grad():
    hf_outputs = hf_model(hf_input)
    hf_logits = hf_outputs.logits  # (1, seq_len, vocab_size)

print("Running forward pass on Custom model...")
with torch.no_grad():
    custom_logits, _, _ = custom_model(custom_input)  # (1, seq_len, vocab_size)

print(f"\nHF logits shape: {hf_logits.shape}")
print(f"Custom logits shape: {custom_logits.shape}")
print(f"Shapes match: {hf_logits.shape == custom_logits.shape}")

# Compare logits at the last position (where we'd generate from)
hf_last_logits = hf_logits[0, -1, :]  # (vocab_size,)
custom_last_logits = custom_logits[0, -1, :]  # (vocab_size,)

print(f"\nHF last logits: shape={hf_last_logits.shape}, dtype={hf_last_logits.dtype}")
print(f"Custom last logits: shape={custom_last_logits.shape}, dtype={custom_last_logits.dtype}")

# TEST 3: Do they predict the same next token?
print("\n" + "=" * 80)
print("TEST 3: COMPARING PREDICTED NEXT TOKENS")
print("=" * 80)

hf_next_token = torch.argmax(hf_last_logits).item()
custom_next_token = torch.argmax(custom_last_logits).item()

print(f"HF predicts token: {hf_next_token} ({repr(hf_tokenizer.decode([hf_next_token]))})")
print(f"Custom predicts token: {custom_next_token} ({repr(custom_tokenizer.decode([custom_next_token]))})")

if hf_next_token == custom_next_token:
    print("✓ Both models predict the same first token")
else:
    print("✗ Models predict DIFFERENT first tokens!")

# TEST 4: Precision/numerical differences
print("\n" + "=" * 80)
print("TEST 4: CHECKING PRECISION/NUMERICAL DIFFERENCES")
print("=" * 80)

# Compare logits numerically
logits_diff = (hf_last_logits - custom_last_logits).abs()
max_diff = logits_diff.max().item()
mean_diff = logits_diff.mean().item()

print(f"Max absolute difference in logits: {max_diff:.6e}")
print(f"Mean absolute difference in logits: {mean_diff:.6e}")

# Show top-5 predictions from each
print("\n" + "-" * 80)
print("Top 5 predictions from HF:")
hf_top5 = torch.topk(hf_last_logits, 5)
for i, (logit, token_id) in enumerate(zip(hf_top5.values, hf_top5.indices)):
    token_str = hf_tokenizer.decode([token_id.item()])
    print(f"  {i+1}. token {token_id.item():6d}: logit={logit.item():8.3f}, text={repr(token_str)}")

print("\nTop 5 predictions from Custom:")
custom_top5 = torch.topk(custom_last_logits, 5)
for i, (logit, token_id) in enumerate(zip(custom_top5.values, custom_top5.indices)):
    token_str = custom_tokenizer.decode([token_id.item()])
    print(f"  {i+1}. token {token_id.item():6d}: logit={logit.item():8.3f}, text={repr(token_str)}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Tokenization matches: {hf_tokens == custom_tokens}")
print(f"Logit shapes match: {hf_logits.shape == custom_logits.shape}")
print(f"First token matches: {hf_next_token == custom_next_token}")
print(f"Max logit difference: {max_diff:.6e}")
print("=" * 80)
