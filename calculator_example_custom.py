"""
Simple calculator tool calling example using custom Qwen3-4B-Instruct implementation

This demonstrates:
- Defining calculator tools (add, subtract, multiply, divide)
- Having the model call tools to perform calculations
- Chaining multiple tool calls together
- Using the custom model implementation instead of HuggingFace transformers
"""

import json
import torch
from src.model import Qwen3Model
from src.tokenizer import Tokenizer


# Calculator functions
def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a"""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


# Tool registry
TOOLS = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide,
}


def parse_tool_call(text: str) -> dict | None:
    """Extract tool call JSON from model response"""
    if "<tool_call>" not in text or "</tool_call>" not in text:
        return None

    start = text.find("<tool_call>") + len("<tool_call>")
    end = text.find("</tool_call>")
    tool_text = text[start:end].strip()

    try:
        return json.loads(tool_text)
    except json.JSONDecodeError:
        print(f"Failed to parse tool call: {tool_text}")
        return None


def execute_tool(tool_call: dict) -> float:
    """Execute a tool call and return the result"""
    name = tool_call["name"]
    args = tool_call["arguments"]

    if name not in TOOLS:
        raise ValueError(f"Unknown tool: {name}")

    result = TOOLS[name](**args)
    print(f"  → {name}({', '.join(f'{k}={v}' for k, v in args.items())}) = {result}")
    return result


def main():
    # Load model and tokenizer
    print("Loading custom Qwen3-4B-Instruct model...")
    model = Qwen3Model()
    tokenizer = Tokenizer()
    print("Model loaded!\n")

    # System prompt explaining tool calling
    system_prompt = """You are a calculator assistant. You MUST use tools for ALL arithmetic operations. Never calculate numbers directly.

Available tools:
- add(a, b): Add two numbers
- subtract(a, b): Subtract b from a
- multiply(a, b): Multiply two numbers
- divide(a, b): Divide a by b

IMPORTANT RULES:
1. You MUST call a tool for EVERY arithmetic operation - never compute numbers yourself
2. For multi-step calculations, call one tool at a time and wait for the result
3. After receiving a tool result, if more calculations are needed, call another tool
4. Only provide the final answer after all tool calls are complete

To call a tool, respond with ONLY:
<tool_call>
{"name": "tool_name", "arguments": {"a": value1, "b": value2}}
</tool_call>

Example for "What is 5 + 3 multiplied by 2?":
Step 1: Call add(5, 3) -> get result 8
Step 2: Call multiply(8, 2) -> get result 16
Step 3: Provide final answer: 16"""

    # Test questions - starting with complex multi-step calculations
    test_questions = [
        "What is (15 + 27) multiplied by 3?",
        "Calculate 100 divided by 5, then add 30 to the result",
        "What is (50 minus 10) divided by (2 plus 3)?",
        "First multiply 8 by 9, then subtract 20 from that result",
        "What is 144 divided by 12, then multiply that by 5?",
        "Calculate (25 + 15) times (10 - 4)",
    ]

    print("=" * 70)
    print("CALCULATOR TOOL CALLING EXAMPLES (Custom Model)")
    print("=" * 70)

    for i, question in enumerate(test_questions, 1):
        print(f"\n[Question {i}] {question}")

        # Build conversation
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Track cache for efficient generation
        cache_k = None
        cache_v = None

        # Maximum 10 turns to avoid infinite loops
        for turn in range(10):
            # Add user message (first turn only)
            if turn == 0:
                messages.append({"role": "user", "content": question})

            # Generate model response using custom tokenizer
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            input_ids = tokenizer.encode(text)
            input_tensor = torch.tensor([input_ids])  # (1, seq)

            # Generate response
            new_tokens, cache_k, cache_v = model.generate(
                input_ids=input_tensor,
                max_new_tokens=200,
                temperature=0.01,  # Nearly deterministic (our model doesn't support temperature=0)
                cache_k=cache_k,
                cache_v=cache_v,
                stop_token_ids=[
                    tokenizer.special_tokens["<|im_end|>"],
                    tokenizer.special_tokens["<|endoftext|>"],
                ],
            )

            # Decode the response
            response_ids = new_tokens[0].tolist()
            response = tokenizer.decode(response_ids)

            # Check if response contains a tool call
            tool_call = parse_tool_call(response)

            if tool_call:
                # Execute the tool
                result = execute_tool(tool_call)

                # Add assistant's tool call and result to conversation
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "tool",
                    "content": json.dumps({"result": result})
                })

                # Reset cache since we're adding new context
                cache_k = None
                cache_v = None

                # Continue to let model process the result
                continue
            else:
                # No tool call - this is the final answer
                # Clean up the response
                final_response = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
                print(f"  Answer: {final_response}")
                break

        print()


if __name__ == "__main__":
    main()
