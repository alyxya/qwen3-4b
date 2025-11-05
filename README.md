# Qwen3-4B Implementation

A minimal implementation of the Qwen3 4B model built for inference, implemented from scratch in PyTorch.

## Features

- ✅ Complete transformer architecture (36 layers)
- ✅ Grouped Query Attention (GQA) with KV caching
- ✅ RoPE (Rotary Position Embeddings)
- ✅ RMSNorm layers
- ✅ SwiGLU MLP
- ✅ BPE tokenizer
- ✅ Weight loading from HuggingFace

## Testing

Tests are organized into fast and slow categories:

### Run Fast Tests Only (recommended for development)
```bash
pytest -v -m "not slow"
```

### Run Slow Tests Only (model initialization and forward pass)
```bash
pytest -v -m "slow"
```

### Run All Tests
```bash
pytest -v
```

## Project Structure

```
.
├── model.py              # Complete Qwen3Model implementation
├── tokenizer.py          # BPE tokenizer
├── attention.py          # Multi-head GQA with KV cache
├── transformer_block.py  # Transformer layer
├── mlp.py                # Feed-forward network with SwiGLU
├── rmsnorm.py            # RMSNorm implementation
├── rope.py               # Rotary Position Embeddings
├── embedding.py          # Token embeddings
├── load_weights.py       # HuggingFace weight loading
└── tests/                # Pytest test suite
```

## Model Architecture

- **Parameters**: 4.06B
- **Layers**: 36
- **Hidden Size**: 2560
- **Attention Heads**: 32 query, 8 key/value (GQA)
- **Head Dimension**: 80
- **MLP Hidden Size**: 9728
- **Vocabulary Size**: 151,936
- **Max Context**: 262,144 tokens
