# Qwen3-4B Implementation

A from-scratch implementation of the Qwen3 4B language model in PyTorch, built for educational purposes and inference.

## Features

- ✅ Complete transformer architecture (36 layers, 4B parameters)
- ✅ Grouped Query Attention (GQA) with efficient KV caching
- ✅ RoPE (Rotary Position Embeddings) with θ = 5,000,000
- ✅ RMSNorm layers with Q/K normalization
- ✅ SwiGLU MLP with ~3.8x expansion ratio
- ✅ BPE tokenizer with byte-level encoding
- ✅ Weight loading from HuggingFace Hub
- ✅ Flexible `generate()` method with temperature, top-k, and top-p sampling
- ✅ Comprehensive test suite (42 fast + 12 slow tests)

## Quick Start

### Installation

```bash
pip install torch huggingface_hub safetensors regex
```

### Basic Usage

```python
from src.model import Qwen3Model
from src.tokenizer import Tokenizer

# Load model and tokenizer
model = Qwen3Model()
tokenizer = Tokenizer()

# Generate text
prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt)

new_tokens, cache_k, cache_v = model.generate(
    input_ids=input_ids,
    max_new_tokens=20,
    temperature=0.7,
    top_k=50,
)

# Decode output
full_text = tokenizer.decode(input_ids + new_tokens)
print(full_text)  # "The capital of France is Paris..."
```

### Demo Script

Run the included demo:

```bash
python main.py
```

This generates text for 4 example prompts showing various model capabilities.

## Model Architecture

- **Parameters**: 4,022,468,096 (~4.02B)
- **Layers**: 36 transformer blocks
- **Hidden Size**: 2560
- **Attention**: 32 query heads, 8 key/value heads (GQA with 4:1 ratio)
- **Head Dimension**: 128
- **MLP Hidden Size**: 9728 (~3.8x expansion)
- **Vocabulary**: 151,936 tokens
- **Max Context**: 262,144 tokens (256K)
- **RoPE Theta**: 5,000,000
- **Precision**: bfloat16 throughout

## Generation API

### The `generate()` Method

```python
new_tokens, cache_k, cache_v = model.generate(
    input_ids=tokens,          # Required for first call, optional with cache
    max_new_tokens=50,         # Number of tokens to generate
    temperature=1.0,           # Sampling temperature (lower = more deterministic)
    top_k=None,               # Sample from top k tokens (None = no filtering)
    top_p=None,               # Nucleus sampling threshold (None = disabled)
    cache_k=None,             # Existing key cache (for continuation)
    cache_v=None,             # Existing value cache (for continuation)
)
```

**Returns:**
- `new_tokens`: List of ONLY newly generated token IDs (does NOT include input_ids)
- `cache_k`, `cache_v`: Updated KV cache containing all processed tokens

### Usage Patterns

#### Pattern 1: Simple One-Shot Generation

```python
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt)

new_tokens, _, _ = model.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=0.8,
)

full_text = tokenizer.decode(input_ids + new_tokens)
```

#### Pattern 2: Continuation (No New Input)

```python
# Generate first part
new_tokens_1, cache_k, cache_v = model.generate(
    input_ids=input_ids,
    max_new_tokens=20,
)

# Continue generating from cache (no new input)
new_tokens_2, cache_k, cache_v = model.generate(
    input_ids=None,  # No new input needed
    max_new_tokens=20,
    cache_k=cache_k,
    cache_v=cache_v,
)

# Full output
all_tokens = input_ids + new_tokens_1 + new_tokens_2
```

#### Pattern 3: Multi-Turn Conversation (Adding Context)

```python
# System prompt
system = "You are a helpful assistant."
system_ids = tokenizer.encode(system)

# Generate response
response_1, cache_k, cache_v = model.generate(
    input_ids=system_ids,
    max_new_tokens=30,
)

# User sends a message (NEW context)
user_msg = "What is Python?"
user_ids = tokenizer.encode(user_msg)

# Add user message and generate response
response_2, cache_k, cache_v = model.generate(
    input_ids=user_ids,  # NEW tokens to add to conversation
    max_new_tokens=50,
    cache_k=cache_k,
    cache_v=cache_v,
)

# Full conversation
conversation = system_ids + response_1 + user_ids + response_2
```

### Key Semantics

**Cache** = All previously processed tokens (complete history)

**input_ids** = NEW tokens to process before generating
- `None` or `[]`: Pure continuation, generate from current cache state
- `[tokens]`: Add these NEW tokens to cache, then generate

**Return value** = ONLY the newly generated tokens
- To get full sequence: `input_ids + new_tokens`
- Cache includes everything: `input_ids + new_tokens`

This design enables:
- ✅ Pure continuation without redundant processing
- ✅ Adding new context mid-generation (chat systems)
- ✅ Clean composition of multiple generations

## Testing

Tests are organized by speed:

### Fast Tests (~18 seconds)

Unit tests for individual components without model loading:

```bash
pytest -m "not slow"
```

### Slow Tests (~2-3 minutes)

Integration tests with full model and pretrained weights:

```bash
pytest -m "slow"
```

### All Tests

```bash
pytest
```

## Project Structure

```
qwen3-4b/
├── src/
│   ├── __init__.py
│   ├── model.py              # Qwen3Model with generate() method
│   ├── transformer_block.py  # Single transformer layer
│   ├── attention.py          # Multi-head GQA with KV cache
│   ├── mlp.py               # SwiGLU feed-forward network
│   ├── embedding.py         # Token embeddings
│   ├── rmsnorm.py           # RMSNorm implementation
│   ├── rope.py              # Rotary position embeddings
│   ├── tokenizer.py         # BPE tokenizer
│   └── load_weights.py      # HuggingFace weight loader
├── tests/
│   ├── conftest.py          # Shared fixtures (model, tokenizer, config)
│   ├── test_model.py        # Model configuration and forward pass
│   ├── test_integration.py  # End-to-end generation tests
│   ├── test_attention.py    # Attention mechanism tests
│   ├── test_transformer_block.py
│   ├── test_mlp.py
│   ├── test_rope.py
│   ├── test_embedding.py
│   ├── test_tokenizer.py
│   └── test_load_weights.py
├── main.py                  # Demo script
├── AGENTS.md               # Guide for AI coding agents
├── CLAUDE.md -> AGENTS.md  # Symlink
└── README.md               # This file
```

## Implementation Highlights

### Memory-Efficient Initialization

The model uses PyTorch's `meta` device for efficient weight loading:
- Parameters initialized on meta device (zero memory allocation)
- Pretrained weights loaded directly with `assign=True` (no copying)
- Total memory: ~1x model size instead of 2x

### Grouped Query Attention (GQA)

GQA reduces KV cache size by sharing key/value heads across query heads:
- 32 query heads, 8 KV heads (4:1 ratio)
- Query heads grouped: `(batch, 8 groups, 4 queries/group, seq, 128)`
- Efficient einsum operations: `torch.einsum("bghsd,bgkd->bghsk", q, k)`

### Rotary Position Embeddings (RoPE)

- Precomputed frequency buffers in bfloat16
- Applied per-head after Q/K projection and normalization
- Theta = 5,000,000 for extended context support

### KV Caching

- Cache format: `(batch_size, num_kv_heads, seq_len, head_dim)`
- Automatic position tracking based on cache length
- Efficient concatenation for autoregressive generation

### Data Types

- All model weights and activations: **bfloat16**
- RoPE buffers: bfloat16 (avoids dtype promotion)
- Attention masks: bfloat16 (matches score dtype)

## Development

### For AI Coding Agents

See [AGENTS.md](AGENTS.md) for detailed guidance on:
- Project structure and architecture
- Import conventions (relative within src/, absolute from tests/)
- Weight loading and parameter mapping
- KV cache semantics
- Testing strategy
- Common tasks and debugging tips

### Type Checking

The codebase includes comprehensive type annotations:

```bash
pyright src/
```

### Code Style

- **Type hints**: Required for all function signatures
- **Docstrings**: Required for classes and public methods
- **Inline comments**: Tensor shapes (e.g., `# (batch, seq, dim)`)
- **Naming**: Descriptive names matching academic papers

## Technical Details

### Weight Tying

The language model head (`lm_head`) shares weights with the token embedding layer:

```python
self.lm_head = self.embed_tokens.embedding.weight
```

This is standard practice and reduces parameter count by ~390M.

### Causal Masking

Attention uses causal masking for autoregressive generation:
- Prefill (seq_len > 1): Full causal mask applied
- Decode (seq_len = 1): No mask needed (only attending to past)

### Q/K Normalization

Following Qwen3 architecture, both Q and K projections are normalized with RMSNorm:

```python
q = self.q_norm(q)  # Per-head normalization
k = self.k_norm(k)  # Per-head normalization
```

This improves training stability and performance.

## Resources

- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [Grouped Query Attention Paper](https://arxiv.org/abs/2305.13245)
- [RoFormer (RoPE) Paper](https://arxiv.org/abs/2104.09864)
- [RMSNorm Paper](https://arxiv.org/abs/1910.07467)

## License

This is an educational implementation. The pretrained weights are loaded from HuggingFace and subject to the Qwen3 license.

## Contributing

This implementation is designed to be readable and hackable. When in doubt:
1. Check the corresponding test file for usage examples
2. Look at tensor shape comments throughout the code
3. Refer to AGENTS.md for architecture details
4. Check the Qwen3 model card on HuggingFace
