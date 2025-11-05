# Text Generation Semantics

This document explains the semantics of the `Qwen3Model.generate()` method, including how `input_ids`, `cache_k`, `cache_v`, and return values work.

## Method Signature

```python
generated_ids, cache_k, cache_v = model.generate(
    input_ids: list[int],
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    cache_k: list[torch.Tensor] | None = None,
    cache_v: list[torch.Tensor] | None = None,
)
```

## Parameters

### `input_ids: list[int]`

**Semantics**: Token IDs that will be processed **before** generating new tokens.

**Behavior**:
- **Without cache** (`cache_k=None`): These are the prompt tokens. The model processes all tokens in a single forward pass (prefill), then generates `max_new_tokens` new tokens.
- **With cache** (`cache_k` provided): These tokens are **already in the cache**. Only the **last token** from `input_ids` is used as input to continue generation. This avoids re-processing tokens already in the cache.

**Important**: When using cache, `input_ids` should contain the tokens that correspond to the cache state. Typically this is just `[last_generated_token]`.

### `cache_k: list[torch.Tensor] | None`
### `cache_v: list[torch.Tensor] | None`

**Semantics**: Key/Value cache from previous generation, representing already-processed tokens.

**Structure**:
- List of 36 tensors (one per transformer layer)
- Each tensor shape: `(batch_size, num_kv_heads, seq_len, head_dim)` = `(1, 8, seq_len, 128)`
- `seq_len` is the number of tokens already processed

**Behavior**:
- `None`: Start fresh generation (prefill phase processes all `input_ids`)
- Provided: Continue from cached state (only process new tokens incrementally)

## Return Values

### `generated_ids: list[int]`

**Semantics**: **All** token IDs for the generated sequence.

**Content**: `input_ids + newly_generated_tokens`

**Example**:
```python
input_ids = [1, 2, 3]  # "The capital"
generated_ids, _, _ = model.generate(input_ids, max_new_tokens=2)
# generated_ids = [1, 2, 3, 4, 5]
#                 ^^^^^^^^  ^^^^
#                 input     new tokens
```

**Important**: The returned `generated_ids` **always includes** the original `input_ids`.

### `cache_k: list[torch.Tensor]`
### `cache_v: list[torch.Tensor]`

**Semantics**: Updated KV cache representing **all processed tokens** (input + generated).

**Content**: Cache for tokens corresponding to `generated_ids`

**Cache sequence length**: `len(input_ids) + max_new_tokens`

## Usage Patterns

### Pattern 1: Simple Generation (No Cache)

Generate text from a prompt in one call:

```python
from src.model import Qwen3Model
from src.tokenizer import Tokenizer

model = Qwen3Model()
tokenizer = Tokenizer()

# Encode prompt
prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt)  # [1, 2, 3, 4, 5]

# Generate
generated_ids, cache_k, cache_v = model.generate(
    input_ids=input_ids,
    max_new_tokens=10,
    temperature=0.7,
    top_k=50,
)

# generated_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
#                 ^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                 input (5)      new tokens (10)

text = tokenizer.decode(generated_ids)
print(text)  # "The capital of France is Paris, the largest..."
```

### Pattern 2: Multi-Turn Conversation (With Cache)

Continue generation efficiently by reusing cache:

```python
# First turn: Generate initial response
prompt = "Hello, my name is"
input_ids = tokenizer.encode(prompt)

generated_ids_1, cache_k, cache_v = model.generate(
    input_ids=input_ids,
    max_new_tokens=5,
)
# generated_ids_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# cache represents tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9]

text_1 = tokenizer.decode(generated_ids_1)
print(text_1)  # "Hello, my name is Alice"

# Second turn: Continue generation with cache
# IMPORTANT: Only pass the last token, cache already has the rest
last_token_id = generated_ids_1[-1]
generated_ids_2, cache_k, cache_v = model.generate(
    input_ids=[last_token_id],  # Just [9]
    max_new_tokens=5,
    cache_k=cache_k,  # Cache has [1,2,3,4,5,6,7,8,9]
    cache_v=cache_v,
)
# generated_ids_2 = [9, 10, 11, 12, 13, 14]
#                   ^  ^^^^^^^^^^^^^^^^^^
#                   |  new tokens (5)
#                   last token from input_ids

# Combine results (skip duplicate last token)
full_sequence = generated_ids_1 + generated_ids_2[1:]
# full_sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

text_2 = tokenizer.decode(full_sequence)
print(text_2)  # "Hello, my name is Alice and I"
```

### Pattern 3: Iterative Generation

Generate one token at a time with full control:

```python
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt)

cache_k, cache_v = None, None
all_tokens = input_ids.copy()

for _ in range(20):
    # Generate one token
    if cache_k is None:
        # First iteration: process full prompt
        gen_ids, cache_k, cache_v = model.generate(
            input_ids=all_tokens,
            max_new_tokens=1,
        )
    else:
        # Subsequent iterations: only process last token
        gen_ids, cache_k, cache_v = model.generate(
            input_ids=[all_tokens[-1]],
            max_new_tokens=1,
            cache_k=cache_k,
            cache_v=cache_v,
        )

    # Get the newly generated token
    new_token = gen_ids[-1]
    all_tokens.append(new_token)

    # Optional: decode and print incrementally
    print(tokenizer.decode([new_token]), end="", flush=True)

final_text = tokenizer.decode(all_tokens)
```

## Key Insights

### 1. **`input_ids` semantics change based on cache**

| Cache State | `input_ids` Meaning | Model Processes |
|-------------|---------------------|-----------------|
| `cache_k=None` | Prompt tokens to start from | All tokens in `input_ids` (prefill) |
| `cache_k` provided | Tokens already in cache | Only last token from `input_ids` (decode) |

### 2. **Return `generated_ids` always includes input**

The returned `generated_ids` is **cumulative**, not just the new tokens:
- Length: `len(input_ids) + max_new_tokens`
- Content: Original input + newly sampled tokens

### 3. **Cache represents ALL processed tokens**

After generation, cache contains KV states for every token in `generated_ids`:
- Cache sequence length: `len(generated_ids)`
- Corresponds exactly to the tokens in `generated_ids`

### 4. **Efficient continuation requires careful handling**

When continuing with cache:
```python
# ✅ CORRECT: Pass only last token as input
generated_ids_2, cache_k, cache_v = model.generate(
    input_ids=[generated_ids_1[-1]],
    cache_k=cache_k,
    cache_v=cache_v,
)

# ❌ WRONG: Re-passing all tokens wastes computation
generated_ids_2, cache_k, cache_v = model.generate(
    input_ids=generated_ids_1,  # Cache already has these!
    cache_k=cache_k,
    cache_v=cache_v,
)
```

### 5. **Combining sequences must avoid duplication**

When concatenating multiple generation results:
```python
# The last token of generated_ids_1 is the first token of generated_ids_2
full_sequence = generated_ids_1 + generated_ids_2[1:]  # Skip duplicate
```

## Implementation Details

From the source code (src/model.py:156-168):

```python
for i in range(max_new_tokens):
    if i == 0 and cache_k is None:
        # Prefill: Process all input_ids
        input_tensor = torch.tensor([generated_ids])
    else:
        # Decode: Process only last token with cache
        input_tensor = torch.tensor([[generated_ids[-1]]])

    logits, cache_k, cache_v = self(input_tensor, cache_k=cache_k, cache_v=cache_v)
    # ... sample next token ...
    generated_ids.append(next_token_id)
```

**Key observations**:
- First iteration without cache: Full prefill
- All other iterations: Single-token decode
- Cache automatically accumulates all processed tokens
- `generated_ids` grows by appending each sampled token

## Common Mistakes

### Mistake 1: Not skipping duplicate tokens when concatenating

```python
# ❌ WRONG
full = generated_ids_1 + generated_ids_2  # Duplicate last/first token!

# ✅ CORRECT
full = generated_ids_1 + generated_ids_2[1:]  # Skip duplicate
```

### Mistake 2: Re-processing cached tokens

```python
# ❌ WRONG (inefficient)
generated_ids_2, cache_k, cache_v = model.generate(
    input_ids=generated_ids_1,  # All already cached!
    cache_k=cache_k,
    cache_v=cache_v,
)

# ✅ CORRECT (efficient)
generated_ids_2, cache_k, cache_v = model.generate(
    input_ids=[generated_ids_1[-1]],  # Only last token
    cache_k=cache_k,
    cache_v=cache_v,
)
```

### Mistake 3: Expecting only new tokens in return

```python
generated_ids, _, _ = model.generate(input_ids, max_new_tokens=5)

# ❌ WRONG assumption
assert len(generated_ids) == 5  # Fails!

# ✅ CORRECT
assert len(generated_ids) == len(input_ids) + 5  # Passes
```

## Summary

- **`input_ids`**: Tokens to process (full prompt if no cache, last token if cache provided)
- **`cache_k/v` (input)**: Previous KV cache to continue from (None = start fresh)
- **`generated_ids` (return)**: **All tokens** (input + generated), length = `len(input_ids) + max_new_tokens`
- **`cache_k/v` (return)**: Updated cache for all tokens in `generated_ids`

The API is designed for flexibility: simple one-shot generation without cache management, or efficient multi-turn generation by manually passing cache between calls.
