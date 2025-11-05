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

**Semantics**: **Only newly generated** token IDs.

**Content**: `newly_generated_tokens` (does NOT include input_ids)

**Example**:
```python
input_ids = [1, 2, 3]  # "The capital"
new_tokens, _, _ = model.generate(input_ids, max_new_tokens=2)
# new_tokens = [4, 5]  # Only the new tokens!
#
# To get full sequence: input_ids + new_tokens = [1, 2, 3, 4, 5]
```

**Important**: The returned `generated_ids` contains **ONLY** newly generated tokens, NOT the input!

### `cache_k: list[torch.Tensor]`
### `cache_v: list[torch.Tensor]`

**Semantics**: Updated KV cache representing **all processed tokens** (input + generated).

**Content**: Cache for tokens corresponding to `input_ids + generated_ids`

**Cache sequence length**: `len(input_ids) + len(generated_ids)` = `len(input_ids) + max_new_tokens`

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
new_tokens, cache_k, cache_v = model.generate(
    input_ids=input_ids,
    max_new_tokens=10,
    temperature=0.7,
    top_k=50,
)

# new_tokens = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # Only new tokens!

# Combine to get full sequence
all_tokens = input_ids + new_tokens
# all_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

text = tokenizer.decode(all_tokens)
print(text)  # "The capital of France is Paris, the largest..."
```

### Pattern 2: Multi-Turn Conversation (With Cache) - **IMPROVED!**

Continue generation efficiently by reusing cache. The new API is much cleaner:

```python
# First turn: Generate initial response
prompt = "Hello, my name is"
input_ids = tokenizer.encode(prompt)  # [1, 2, 3, 4]

new_tokens_1, cache_k, cache_v = model.generate(
    input_ids=input_ids,
    max_new_tokens=5,
)
# new_tokens_1 = [5, 6, 7, 8, 9]
# cache represents tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9]

all_tokens_1 = input_ids + new_tokens_1
text_1 = tokenizer.decode(all_tokens_1)
print(text_1)  # "Hello, my name is Alice"

# Second turn: Continue generation with cache
# Beautiful! Just pass new_tokens_1 back in with the cache
new_tokens_2, cache_k, cache_v = model.generate(
    input_ids=new_tokens_1,  # Pass previous output directly!
    max_new_tokens=5,
    cache_k=cache_k,  # Cache already has [1,2,3,4,5,6,7,8,9]
    cache_v=cache_v,
)
# new_tokens_2 = [10, 11, 12, 13, 14]  # Only new tokens

# Combine results - clean concatenation, no duplicates!
all_tokens_2 = input_ids + new_tokens_1 + new_tokens_2
# all_tokens_2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

text_2 = tokenizer.decode(all_tokens_2)
print(text_2)  # "Hello, my name is Alice and I love"
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
        new_tokens, cache_k, cache_v = model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
        )
    else:
        # Subsequent iterations: just pass the previous output!
        new_tokens, cache_k, cache_v = model.generate(
            input_ids=new_tokens,  # Pass previous output directly
            max_new_tokens=1,
            cache_k=cache_k,
            cache_v=cache_v,
        )

    # Add newly generated token
    all_tokens.extend(new_tokens)

    # Optional: decode and print incrementally
    print(tokenizer.decode(new_tokens), end="", flush=True)

final_text = tokenizer.decode(all_tokens)
```

## Key Insights

### 1. **`input_ids` semantics change based on cache**

| Cache State | `input_ids` Meaning | Model Processes |
|-------------|---------------------|-----------------|
| `cache_k=None` | Prompt tokens to start from | All tokens in `input_ids` (prefill) |
| `cache_k` provided | Tokens already in cache | Only last token from `input_ids` (decode) |

### 2. **Return `generated_ids` contains ONLY new tokens**

The returned `generated_ids` contains **only newly generated tokens**:
- Length: `max_new_tokens`
- Content: Only newly sampled tokens (NOT including input)
- To get full sequence: `input_ids + generated_ids`

### 3. **Cache represents ALL processed tokens**

After generation, cache contains KV states for input + generated tokens:
- Cache sequence length: `len(input_ids) + len(generated_ids)`
- Corresponds to `input_ids + generated_ids`

### 4. **Efficient continuation is now trivial!**

The output can be passed directly as input with cache:
```python
# ✅ BEAUTIFUL: Pass previous output directly
new_tokens_2, cache_k, cache_v = model.generate(
    input_ids=new_tokens_1,  # Just pass previous output!
    cache_k=cache_k,
    cache_v=cache_v,
)
```

### 5. **Combining sequences is clean**

No need to skip duplicates - just concatenate:
```python
# ✅ CLEAN: Direct concatenation
full_sequence = input_ids + new_tokens_1 + new_tokens_2
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

### Mistake 1: Expecting input_ids in the return value

```python
new_tokens, _, _ = model.generate(input_ids, max_new_tokens=5)

# ❌ WRONG assumption
full_sequence = new_tokens  # Missing the input!

# ✅ CORRECT
full_sequence = input_ids + new_tokens  # Combine input and output
```

### Mistake 2: Forgetting output only includes new tokens

```python
new_tokens, _, _ = model.generate(input_ids, max_new_tokens=5)

# ❌ WRONG assumption
assert len(new_tokens) == len(input_ids) + 5  # Fails!

# ✅ CORRECT
assert len(new_tokens) == 5  # Passes - only new tokens
assert len(input_ids + new_tokens) == len(input_ids) + 5  # Full sequence
```

## Summary

- **`input_ids`**: Tokens to process (full prompt if no cache, previous output tokens if cache provided)
- **`cache_k/v` (input)**: Previous KV cache to continue from (None = start fresh)
- **`generated_ids` (return)**: **ONLY newly generated tokens** (does NOT include input), length = `max_new_tokens`
- **`cache_k/v` (return)**: Updated cache for `input_ids + generated_ids`

The improved API makes continuation trivial: just pass the previous output as input along with the cache. No duplicate handling needed, clean concatenation!
