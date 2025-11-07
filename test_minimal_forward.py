"""
Minimal forward pass test - just embeddings + first attention
"""

import torch
from transformers import AutoModelForCausalLM
from src.model import Qwen3Model
from src.tokenizer import Tokenizer

print("Loading models...")
hf_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    dtype=torch.bfloat16,
    device_map="cpu"
)

custom_model = Qwen3Model()
custom_tokenizer = Tokenizer()
print("✓ Models loaded\n")

# Very simple input
tokens = [1, 2, 3, 4, 5]  # Just 5 tokens
input_tensor = torch.tensor([tokens])

print("Input tokens:", tokens)
print("=" * 80)

# Step 1: Embeddings
print("\n[STEP 1] Embeddings")
with torch.no_grad():
    hf_embeds = hf_model.model.embed_tokens(input_tensor)
    custom_embeds = custom_model.embed_tokens(input_tensor)

print(f"HF embeds shape: {hf_embeds.shape}")
print(f"Custom embeds shape: {custom_embeds.shape}")
diff = (hf_embeds - custom_embeds).abs().max().item()
print(f"Max diff: {diff:.6e} {'✓' if diff < 1e-6 else '✗'}")

# Step 2: Apply input layernorm
print("\n[STEP 2] Input Layernorm")
with torch.no_grad():
    hf_normed = hf_model.model.layers[0].input_layernorm(hf_embeds)
    custom_normed = custom_model.layers[0].input_layernorm(custom_embeds)

print(f"HF normed shape: {hf_normed.shape}")
print(f"Custom normed shape: {custom_normed.shape}")
diff = (hf_normed - custom_normed).abs().max().item()
print(f"Max diff: {diff:.6e} {'✓' if diff < 1e-6 else '✗'}")

# Step 3: Q, K, V projections
print("\n[STEP 3] Q, K, V Projections")
batch_size, seq_len, d_model = custom_normed.shape

with torch.no_grad():
    # HuggingFace
    hf_q = hf_model.model.layers[0].self_attn.q_proj(hf_normed)
    hf_k = hf_model.model.layers[0].self_attn.k_proj(hf_normed)
    hf_v = hf_model.model.layers[0].self_attn.v_proj(hf_normed)

    # Custom
    custom_q = custom_model.layers[0].self_attn.q_proj(custom_normed)
    custom_k = custom_model.layers[0].self_attn.k_proj(custom_normed)
    custom_v = custom_model.layers[0].self_attn.v_proj(custom_normed)

print(f"Q - HF shape: {hf_q.shape}, Custom shape: {custom_q.shape}")
diff_q = (hf_q - custom_q).abs().max().item()
print(f"    Max diff: {diff_q:.6e} {'✓' if diff_q < 1e-4 else '✗'}")

print(f"K - HF shape: {hf_k.shape}, Custom shape: {custom_k.shape}")
diff_k = (hf_k - custom_k).abs().max().item()
print(f"    Max diff: {diff_k:.6e} {'✓' if diff_k < 1e-4 else '✗'}")

print(f"V - HF shape: {hf_v.shape}, Custom shape: {custom_v.shape}")
diff_v = (hf_v - custom_v).abs().max().item()
print(f"    Max diff: {diff_v:.6e} {'✓' if diff_v < 1e-4 else '✗'}")

# Step 4: Reshape to heads
print("\n[STEP 4] Reshape to Heads")
num_heads = 32
num_kv_heads = 8
head_dim = 128

with torch.no_grad():
    # HuggingFace reshape
    hf_q_heads = hf_q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    hf_k_heads = hf_k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    hf_v_heads = hf_v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # Custom reshape
    custom_q_heads = custom_q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    custom_k_heads = custom_k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    custom_v_heads = custom_v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

print(f"Q heads - HF shape: {hf_q_heads.shape}, Custom shape: {custom_q_heads.shape}")
diff = (hf_q_heads - custom_q_heads).abs().max().item()
print(f"          Max diff: {diff:.6e} {'✓' if diff < 1e-4 else '✗'}")

# Step 5: Apply Q/K norm
print("\n[STEP 5] Q/K Norm")
with torch.no_grad():
    # HuggingFace
    hf_q_normed = hf_model.model.layers[0].self_attn.q_norm(hf_q_heads)
    hf_k_normed = hf_model.model.layers[0].self_attn.k_norm(hf_k_heads)

    # Custom
    custom_q_normed = custom_model.layers[0].self_attn.q_norm(custom_q_heads)
    custom_k_normed = custom_model.layers[0].self_attn.k_norm(custom_k_heads)

print(f"Q normed - HF shape: {hf_q_normed.shape}, Custom shape: {custom_q_normed.shape}")
diff_q = (hf_q_normed - custom_q_normed).abs().max().item()
print(f"           Max diff: {diff_q:.6e} {'✓' if diff_q < 1e-4 else '✗'}")

print(f"K normed - HF shape: {hf_k_normed.shape}, Custom shape: {custom_k_normed.shape}")
diff_k = (hf_k_normed - custom_k_normed).abs().max().item()
print(f"           Max diff: {diff_k:.6e} {'✓' if diff_k < 1e-4 else '✗'}")

print("\n" + "=" * 80)
print("If everything matches up to here, the bug is in RoPE or attention computation")
print("=" * 80)
