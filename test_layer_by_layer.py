"""
Compare HF vs Custom model layer-by-layer to find where divergence happens
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model import Qwen3Model
from src.tokenizer import Tokenizer

print("Loading models...")
hf_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    dtype=torch.bfloat16,
    device_map="cpu"
)
hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

custom_model = Qwen3Model()
custom_tokenizer = Tokenizer()
print("✓ Models loaded\n")

# Simple test input
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hi"},
]

text = custom_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
tokens = custom_tokenizer.encode(text)
input_tensor = torch.tensor([tokens])

print(f"Input: {len(tokens)} tokens")
print("=" * 80)
print("LAYER-BY-LAYER COMPARISON")
print("=" * 80)

# Compare embeddings
print("\n[1] Embeddings")
with torch.no_grad():
    hf_embeds = hf_model.model.embed_tokens(input_tensor)
    custom_embeds = custom_model.embed_tokens(input_tensor)

hf_norm = hf_embeds.norm().item()
custom_norm = custom_embeds.norm().item()
diff = (hf_embeds - custom_embeds).abs().max().item()

print(f"  HF norm: {hf_norm:.6f}")
print(f"  Custom norm: {custom_norm:.6f}")
print(f"  Max diff: {diff:.6e}")
print(f"  Match: {'✓' if diff < 1e-2 else '✗'}")

# Compare first transformer layer
print("\n[2] First Transformer Layer")
with torch.no_grad():
    hf_layer0_out, _, _ = hf_model.model.layers[0](hf_embeds)
    custom_layer0_out, _, _ = custom_model.layers[0](custom_embeds)

hf_norm = hf_layer0_out.norm().item()
custom_norm = custom_layer0_out.norm().item()
diff = (hf_layer0_out - custom_layer0_out).abs().max().item()

print(f"  HF norm: {hf_norm:.6f}")
print(f"  Custom norm: {custom_norm:.6f}")
print(f"  Max diff: {diff:.6e}")
print(f"  Match: {'✓' if diff < 1e-2 else '✗'}")

# Run through all layers
print("\n[3] All Layers")
hf_hidden = hf_embeds
custom_hidden = custom_embeds

for i in range(min(3, len(custom_model.layers))):  # Just check first 3 layers
    with torch.no_grad():
        hf_hidden, _, _ = hf_model.model.layers[i](hf_hidden)
        custom_hidden, _, _ = custom_model.layers[i](custom_hidden)

    diff = (hf_hidden - custom_hidden).abs().max().item()
    print(f"  After layer {i}: diff={diff:.6e}, match={'✓' if diff < 1e-2 else '✗'}")

print("\n[4] Final Norm")
with torch.no_grad():
    # Run through ALL layers first
    hf_hidden = hf_embeds
    custom_hidden = custom_embeds

    for i in range(len(custom_model.layers)):
        hf_hidden, _, _ = hf_model.model.layers[i](hf_hidden)
        custom_hidden, _, _ = custom_model.layers[i](custom_hidden)

    hf_normed = hf_model.model.norm(hf_hidden)
    custom_normed = custom_model.norm(custom_hidden)

diff = (hf_normed - custom_normed).abs().max().item()
print(f"  Max diff: {diff:.6e}")
print(f"  Match: {'✓' if diff < 1e-2 else '✗'}")

print("\n[5] LM Head")
with torch.no_grad():
    # HF lm_head
    hf_logits = hf_model.lm_head(hf_normed)

    # Custom lm_head (uses einsum)
    custom_logits = torch.einsum("bsd,vd->bsv", custom_normed, custom_model.lm_head)

diff = (hf_logits - custom_logits).abs().max().item()
print(f"  Max diff: {diff:.6e}")
print(f"  Match: {'✓' if diff < 1e-2 else '✗'}")

# Check if lm_head weights are the same
print("\n[6] LM Head Weights")
hf_lm_head_weight = hf_model.lm_head.weight
custom_lm_head_weight = custom_model.lm_head

weight_diff = (hf_lm_head_weight - custom_lm_head_weight).abs().max().item()
print(f"  HF shape: {hf_lm_head_weight.shape}")
print(f"  Custom shape: {custom_lm_head_weight.shape}")
print(f"  Weight diff: {weight_diff:.6e}")
print(f"  Match: {'✓' if weight_diff < 1e-4 else '✗'}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("Check which layer shows first significant divergence")
print("=" * 80)
