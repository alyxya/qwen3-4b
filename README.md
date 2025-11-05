# Qwen3-4B from Scratch

A clean PyTorch implementation of Qwen3 4B (4 billion parameters) built for learning and inference.

## Quick Start

```python
from src.model import Qwen3Model
from src.tokenizer import Tokenizer

model = Qwen3Model()
tokenizer = Tokenizer()

# Generate text
prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt)
new_tokens, _, _ = model.generate(input_ids, max_new_tokens=20)

print(tokenizer.decode(input_ids + new_tokens))
# "The capital of France is Paris..."
```

Or run the demo:
```bash
python main.py
```

## Model Details

- **4.02B parameters**: 36 layers × 2560 hidden dim
- **Grouped Query Attention**: 32 query heads, 8 KV heads
- **256K context**: RoPE with θ = 5M
- **bfloat16**: Memory-efficient throughout

## Generation API

```python
new_tokens, cache_k, cache_v = model.generate(
    input_ids=tokens,      # Full sequence (prompt + previous generations)
    max_new_tokens=50,
    temperature=0.7,       # Lower = more deterministic
    top_k=50,             # Sample from top k tokens
    top_p=0.9,            # Nucleus sampling
    cache_k=cache_k,      # For continuation
    cache_v=cache_v,
)
```

**Returns only new tokens** (not including input). To get full text: `input_ids + new_tokens`

### Usage Examples

**Simple generation:**
```python
new_tokens, _, _ = model.generate(input_ids, max_new_tokens=50)
```

**Continue generating:**
```python
# Generate 20 tokens
conversation = input_ids
new_1, cache_k, cache_v = model.generate(conversation, max_new_tokens=20)
conversation = conversation + new_1

# Continue for 20 more - pass full sequence
new_2, cache_k, cache_v = model.generate(
    input_ids=conversation,  # Full sequence so far
    max_new_tokens=20,
    cache_k=cache_k,
    cache_v=cache_v,
)
conversation = conversation + new_2
```

**Multi-turn chat:**
```python
# First response
conversation = system_ids
resp_1, cache_k, cache_v = model.generate(conversation, max_new_tokens=30)
conversation = conversation + resp_1

# Add user message and generate
conversation = conversation + user_msg_ids
resp_2, cache_k, cache_v = model.generate(
    input_ids=conversation,  # Full conversation
    max_new_tokens=50,
    cache_k=cache_k,
    cache_v=cache_v,
)
conversation = conversation + resp_2
```

## Testing

```bash
pytest -m "not slow"  # Fast unit tests (~18s)
pytest -m "slow"      # Integration tests with full model (~3min)
pytest                # All tests
```

## Architecture

- **Attention**: Multi-head GQA with Q/K normalization
- **Position**: RoPE (Rotary Position Embeddings)
- **Normalization**: RMSNorm
- **Activation**: SwiGLU in MLP
- **Tokenizer**: Byte-level BPE (151K vocab)
- **Weights**: Loaded from HuggingFace Hub

## Project Structure

```
src/
├── model.py              # Qwen3Model + generate()
├── attention.py          # GQA with KV cache
├── transformer_block.py  # Single layer
├── mlp.py               # SwiGLU FFN
├── rope.py              # Rotary embeddings
├── rmsnorm.py           # Normalization
├── tokenizer.py         # BPE tokenizer
└── load_weights.py      # Weight loader

tests/                    # Comprehensive test suite
main.py                  # Demo script
AGENTS.md               # Guide for AI agents
```

## Key Features

- **Memory efficient**: Meta device initialization, single allocation
- **KV caching**: Efficient autoregressive generation
- **Type safe**: Full type annotations
- **Well tested**: 54 tests covering all components

See [AGENTS.md](AGENTS.md) for implementation details and development guide.
