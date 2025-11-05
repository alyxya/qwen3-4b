# KV Cache and generate() Semantics

## Quick Answer

**The KV cache contains all previously processed tokens. When you call `generate()` with a cache:**

- `input_ids`: Optional new tokens to process before generating (e.g., user input in a conversation)
- If `input_ids=None` or `[]`: Continue generating directly from cache state
- If `input_ids` provided: Process these tokens first, THEN generate `max_new_tokens`

## The generate() Method

```python
new_tokens, cache_k, cache_v = model.generate(
    input_ids=[1, 2, 3],      # Tokens to process before generating
    max_new_tokens=5,          # How many NEW tokens to generate
    cache_k=cache_k,           # Existing cache (or None)
    cache_v=cache_v,
)
```

**Returns:**
- `new_tokens`: ONLY the newly generated tokens (length = `max_new_tokens`)
- `cache_k/v`: Updated cache containing ALL processed tokens

## Usage Patterns

### Pattern 1: Simple Generation (No Cache)

```python
prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt)  # [1, 2, 3, 4, 5]

new_tokens, cache_k, cache_v = model.generate(
    input_ids=input_ids,
    max_new_tokens=10,
)
# new_tokens = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# cache contains: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

full_text = tokenizer.decode(input_ids + new_tokens)
# "The capital of France is Paris"
```

### Pattern 2: Pure Continuation (No New Input)

When you just want to generate more tokens from the current state:

```python
# First generation
new_tokens_1, cache_k, cache_v = model.generate(
    input_ids=[1, 2, 3],
    max_new_tokens=5,
)
# new_tokens_1 = [4, 5, 6, 7, 8]
# cache = [1, 2, 3, 4, 5, 6, 7, 8]

# Continue generating WITHOUT any new input
new_tokens_2, cache_k, cache_v = model.generate(
    input_ids=None,  # or []  - no new input!
    max_new_tokens=5,
    cache_k=cache_k,
    cache_v=cache_v,
)
# new_tokens_2 = [9, 10, 11, 12, 13]
# cache = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# Full sequence is: [1,2,3] + new_tokens_1 + new_tokens_2
```

### Pattern 3: Multi-Turn Conversation (New Input + Cache)

This is the power pattern - adding new context mid-generation:

```python
# System prompt
system = "You are a helpful assistant."
system_ids = tokenizer.encode(system)

# Generate initial response
response_1, cache_k, cache_v = model.generate(
    input_ids=system_ids,
    max_new_tokens=20,
)
# cache = system_ids + response_1

conversation = system_ids + response_1

# User asks a follow-up
user_msg = "What is Python?"
user_ids = tokenizer.encode(user_msg)

# Process user message and generate response
# The cache already has system + response_1
# Now we add user_ids and generate more
response_2, cache_k, cache_v = model.generate(
    input_ids=user_ids,  # NEW tokens to process
    max_new_tokens=30,
    cache_k=cache_k,      # Previous conversation
    cache_v=cache_v,
)
# cache = system_ids + response_1 + user_ids + response_2

conversation = conversation + user_ids + response_2

# Another turn
user_msg_2 = "Give me an example"
user_ids_2 = tokenizer.encode(user_msg_2)

response_3, cache_k, cache_v = model.generate(
    input_ids=user_ids_2,  # NEW tokens
    max_new_tokens=50,
    cache_k=cache_k,
    cache_v=cache_v,
)
# cache = system_ids + response_1 + user_ids + response_2 + user_ids_2 + response_3

conversation = conversation + user_ids_2 + response_3
```

## Key Insights

### 1. Cache Contains ALL Processed Tokens

After calling `generate()`, the cache contains:
```
cache = [all previous tokens] + [input_ids you just passed] + [newly generated tokens]
```

### 2. input_ids Means "Process These Before Generating"

- `input_ids=None` or `[]`: Don't process anything new, just generate from current state
- `input_ids=[...]`: Process these tokens first, add to cache, THEN generate

### 3. The Return Value is Always New Tokens Only

```python
new_tokens, cache = model.generate(input_ids=[1,2,3], max_new_tokens=5, cache=old_cache)
# len(new_tokens) == 5  (always!)
# len(cache) == len(old_cache) + len(input_ids) + max_new_tokens
```

### 4. Why This Design?

This design enables:

✅ **Pure continuation**: `generate(input_ids=None, cache=cache)`
✅ **Adding context**: `generate(input_ids=user_message, cache=cache)`
✅ **Composable**: Chain multiple generations easily
✅ **Efficient**: Never reprocess tokens already in cache

## Common Patterns

### Streaming Generation (Token-by-Token)

```python
cache_k, cache_v = None, None
all_tokens = input_ids.copy()

for _ in range(50):
    new_tokens, cache_k, cache_v = model.generate(
        input_ids=input_ids if cache_k is None else None,
        max_new_tokens=1,
        cache_k=cache_k,
        cache_v=cache_v,
    )

    all_tokens.extend(new_tokens)
    print(tokenizer.decode(new_tokens), end="", flush=True)
```

### Constrained Generation (Inject Tokens Mid-Stream)

```python
# Generate first part
new_tokens_1, cache = model.generate(input_ids=prompt_ids, max_new_tokens=10)

# Force a specific continuation (e.g., "However,")
forced_tokens = tokenizer.encode("However,")

# Process forced tokens (no generation yet)
_, cache = model.generate(
    input_ids=forced_tokens,
    max_new_tokens=0,  # Don't generate, just update cache
    cache_k=cache,
    cache_v=cache,
)

# Continue generating after forced tokens
new_tokens_2, cache = model.generate(
    input_ids=None,
    max_new_tokens=10,
    cache_k=cache,
    cache_v=cache,
)
```

### Beam Search / Parallel Sampling

```python
# Start with same prompt
prompt_ids = tokenizer.encode("Once upon a time")

# Generate 3 different completions
for beam in range(3):
    new_tokens, _, _ = model.generate(
        input_ids=prompt_ids,
        max_new_tokens=20,
        temperature=1.0,
    )
    print(f"Beam {beam}: {tokenizer.decode(prompt_ids + new_tokens)}")
```

## Implementation Details

Internally, `generate()` does:

```python
def generate(input_ids, max_new_tokens, cache_k, cache_v):
    # Step 1: Process input_ids if provided
    if input_ids is not None and len(input_ids) > 0:
        if cache_k is None:
            # Prefill: process all at once
            logits, cache_k, cache_v = model(input_ids)
        else:
            # Incremental: process one by one
            for token in input_ids:
                logits, cache_k, cache_v = model([token], cache_k, cache_v)

    # Step 2: Generate max_new_tokens
    new_tokens = []
    for _ in range(max_new_tokens):
        logits, cache_k, cache_v = model([last_token], cache_k, cache_v)
        next_token = sample(logits)
        new_tokens.append(next_token)
        last_token = next_token

    return new_tokens, cache_k, cache_v
```

## Summary

**Think of it this way:**

- **Cache** = conversation history (all tokens so far)
- **input_ids** = new context to add (user message, system prompt, etc.)
- **new_tokens** = what the model generates

```
[cache] + [input_ids] → process → [updated_cache]
[updated_cache] → generate → [new_tokens] + [final_cache]
```

The API is designed to make multi-turn conversations and constrained generation natural and efficient!
