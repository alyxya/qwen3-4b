"""
Compare actual weight values between HF and Custom models
"""

import torch
from transformers import AutoModelForCausalLM
from src.model import Qwen3Model

print("Loading models...")
hf_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    dtype=torch.bfloat16,
    device_map="cpu"
)

custom_model = Qwen3Model()
print("✓ Models loaded\n")

print("=" * 80)
print("WEIGHT COMPARISON")
print("=" * 80)

# Check embeddings
print("\n[1] Embeddings")
hf_embed_weight = hf_model.model.embed_tokens.weight
custom_embed_weight = custom_model.embed_tokens.weight

diff = (hf_embed_weight - custom_embed_weight).abs().max().item()
print(f"  HF shape: {hf_embed_weight.shape}")
print(f"  Custom shape: {custom_embed_weight.shape}")
print(f"  Max weight diff: {diff:.6e}")
print(f"  Match: {'✓ PERFECT' if diff < 1e-6 else '✗'}")

# Check first layer attention Q projection
print("\n[2] Layer 0 - Attention Q Projection")
hf_q_weight = hf_model.model.layers[0].self_attn.q_proj.weight
custom_q_weight = custom_model.layers[0].self_attn.q_proj.weight

diff = (hf_q_weight - custom_q_weight).abs().max().item()
print(f"  HF shape: {hf_q_weight.shape}")
print(f"  Custom shape: {custom_q_weight.shape}")
print(f"  Max weight diff: {diff:.6e}")
print(f"  Match: {'✓' if diff < 1e-6 else '✗'}")

# Check first layer attention K projection
print("\n[3] Layer 0 - Attention K Projection")
hf_k_weight = hf_model.model.layers[0].self_attn.k_proj.weight
custom_k_weight = custom_model.layers[0].self_attn.k_proj.weight

diff = (hf_k_weight - custom_k_weight).abs().max().item()
print(f"  HF shape: {hf_k_weight.shape}")
print(f"  Custom shape: {custom_k_weight.shape}")
print(f"  Max weight diff: {diff:.6e}")
print(f"  Match: {'✓' if diff < 1e-6 else '✗'}")

# Check Q/K norm weights
print("\n[4] Layer 0 - Q Norm")
hf_q_norm = hf_model.model.layers[0].self_attn.q_norm.weight
custom_q_norm = custom_model.layers[0].self_attn.q_norm.weight

diff = (hf_q_norm - custom_q_norm).abs().max().item()
print(f"  HF shape: {hf_q_norm.shape}")
print(f"  Custom shape: {custom_q_norm.shape}")
print(f"  Max weight diff: {diff:.6e}")
print(f"  Match: {'✓' if diff < 1e-6 else '✗'}")

print("\n[5] Layer 0 - K Norm")
hf_k_norm = hf_model.model.layers[0].self_attn.k_norm.weight
custom_k_norm = custom_model.layers[0].self_attn.k_norm.weight

diff = (hf_k_norm - custom_k_norm).abs().max().item()
print(f"  HF shape: {hf_k_norm.shape}")
print(f"  Custom shape: {custom_k_norm.shape}")
print(f"  Max weight diff: {diff:.6e}")
print(f"  Match: {'✓' if diff < 1e-6 else '✗'}")

# Check MLP weights
print("\n[6] Layer 0 - MLP Gate Projection")
hf_gate = hf_model.model.layers[0].mlp.gate_proj.weight
custom_gate = custom_model.layers[0].mlp.gate_proj.weight

diff = (hf_gate - custom_gate).abs().max().item()
print(f"  HF shape: {hf_gate.shape}")
print(f"  Custom shape: {custom_gate.shape}")
print(f"  Max weight diff: {diff:.6e}")
print(f"  Match: {'✓' if diff < 1e-6 else '✗'}")

# Check input layernorm
print("\n[7] Layer 0 - Input Layernorm")
hf_input_norm = hf_model.model.layers[0].input_layernorm.weight
custom_input_norm = custom_model.layers[0].input_layernorm.weight

diff = (hf_input_norm - custom_input_norm).abs().max().item()
print(f"  HF shape: {hf_input_norm.shape}")
print(f"  Custom shape: {custom_input_norm.shape}")
print(f"  Max weight diff: {diff:.6e}")
print(f"  Match: {'✓' if diff < 1e-6 else '✗'}")

# Check final norm
print("\n[8] Final Norm")
hf_final_norm = hf_model.model.norm.weight
custom_final_norm = custom_model.norm.weight

diff = (hf_final_norm - custom_final_norm).abs().max().item()
print(f"  HF shape: {hf_final_norm.shape}")
print(f"  Custom shape: {custom_final_norm.shape}")
print(f"  Max weight diff: {diff:.6e}")
print(f"  Match: {'✓' if diff < 1e-6 else '✗'}")

# Check lm_head
print("\n[9] LM Head")
hf_lm_head = hf_model.lm_head.weight
custom_lm_head = custom_model.lm_head

diff = (hf_lm_head - custom_lm_head).abs().max().item()
print(f"  HF shape: {hf_lm_head.shape}")
print(f"  Custom shape: {custom_lm_head.shape}")
print(f"  Max weight diff: {diff:.6e}")
print(f"  Match: {'✓' if diff < 1e-6 else '✗'}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("If all weights match but logits don't, the bug is in the forward pass logic!")
print("=" * 80)
