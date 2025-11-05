# AI Agents Guide

This document provides guidance for AI coding agents (like Claude, Cursor, etc.) working with this codebase.

## Project Structure

```
qwen3-4b/
├── src/                      # Source code
│   ├── __init__.py
│   ├── model.py             # Main Qwen3Model class
│   ├── transformer_block.py # Transformer layer implementation
│   ├── attention.py         # Multi-head attention with GQA
│   ├── mlp.py              # Feed-forward network
│   ├── embedding.py        # Token embeddings
│   ├── rmsnorm.py          # RMS normalization
│   ├── rope.py             # Rotary position embeddings
│   ├── tokenizer.py        # BPE tokenizer
│   └── load_weights.py     # Weight loading utilities
├── tests/                   # Test suite
└── README.md               # User documentation
```

## Architecture Overview

This is a from-scratch implementation of **Qwen3 4B** (4 billion parameters) with the following architecture:

- **Model**: 36 transformer layers, 2560 hidden dimension
- **Attention**: Grouped Query Attention (GQA) with 32 query heads and 8 KV heads
- **Position**: Rotary Position Embeddings (RoPE) with θ = 5,000,000
- **Normalization**: RMSNorm applied to inputs and Q/K projections
- **MLP**: SwiGLU activation with ~3.8x expansion ratio
- **Context**: Supports up to 262,144 tokens (256K)

## Key Implementation Details

### Import Structure
- **Within src/**: Use relative imports (e.g., `from .attention import Attention`)
- **From tests/**: Use absolute imports (e.g., `from src.model import Qwen3Model`)

### Weight Loading
- Weights are loaded from HuggingFace Hub (`Qwen/Qwen3-4B-Instruct-2507`)
- Uses `meta` device during initialization to avoid allocating memory twice
- Parameter names are mapped from HuggingFace format to our model structure
- `lm_head` uses weight tying (shares weights with `embed_tokens`)

### KV Cache
- All layers support KV caching for efficient autoregressive generation
- Cache format: `(batch_size, num_kv_heads, seq_len, head_dim)`
- Position IDs are automatically derived from cache state

### Grouped Query Attention (GQA)
- 32 query heads, 8 KV heads (4:1 ratio)
- Query heads are grouped and broadcasted to match KV heads
- Uses einsum for efficient grouped attention computation

### Data Types
- Model uses bfloat16 throughout for memory efficiency
- RoPE buffers and position embeddings use bfloat16 to avoid dtype promotion
- Attention masks created with correct dtype to match scores

## Testing

```bash
# Run fast tests (excludes slow model initialization tests)
pytest tests/ -m "not slow"

# Run all tests including model tests
pytest tests/
```

Test coverage includes:
- Unit tests for each component (attention, MLP, RMSNorm, RoPE, etc.)
- Integration tests for full model forward pass
- Cache functionality tests
- Tokenizer roundtrip tests

## Development Guidelines

### When Adding Features
1. Update the relevant module in `src/`
2. Add corresponding tests in `tests/`
3. Ensure all tests pass with `pytest tests/`
4. Update this guide if architecture changes

### When Modifying Imports
- Keep relative imports within `src/` package
- Update test imports if module structure changes
- Verify with `pytest` that all imports resolve correctly

### When Changing Model Architecture
- Update parameter counts in comments
- Verify tensor shapes in docstrings
- Add tests for new components
- Update README.md with user-facing changes

## Common Tasks

### Adding a New Layer Type
1. Create new module in `src/` (e.g., `src/new_layer.py`)
2. Implement with proper type hints and docstrings
3. Add to `TransformerBlock` or `Qwen3Model` as needed
4. Create test file `tests/test_new_layer.py`
5. Update imports in parent modules

### Debugging Shape Mismatches
- Check comments in code showing expected tensor shapes
- Verify batch_size, seq_len, and dimension parameters
- Ensure RoPE and attention mask shapes align
- Check GQA head grouping logic

### Optimizing Performance
- Model uses bfloat16 for memory efficiency
- KV cache reduces computation in generation
- `meta` device initialization avoids double allocation
- Consider torch.compile() for additional speedup

## Code Style

- **Type hints**: Required for all function signatures
- **Docstrings**: Required for classes and public methods
- **Comments**: Inline shape comments (e.g., `# (batch, seq, dim)`)
- **Naming**: Use descriptive names matching academic papers
- **Testing**: Pytest with fixtures in `conftest.py`

## Resources

- [Qwen3 Technical Report](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [Grouped Query Attention Paper](https://arxiv.org/abs/2305.13245)
- [RoFormer (RoPE) Paper](https://arxiv.org/abs/2104.09864)
- [RMSNorm Paper](https://arxiv.org/abs/1910.07467)

## Questions?

This implementation is designed to be educational and hackable. When in doubt:
1. Check the corresponding test file for usage examples
2. Look at tensor shape comments throughout the code
3. Refer to the Qwen3 model card on HuggingFace
4. Ask the human developer for clarification
