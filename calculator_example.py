"""
Simple calculator tool calling example using HuggingFace Qwen3-4B-Instruct

This demonstrates:
- Defining calculator tools (add, subtract, multiply, divide)
- Having the model call tools to perform calculations
- Chaining multiple tool calls together
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subtract",
            "description": "Subtract b from a",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "divide",
            "description": "Divide a by b",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    },
]


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
    print("Loading Qwen3-4B-Instruct model...")
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    print("Model loaded!\n")

    # System prompt explaining tool calling
    system_prompt = """You are a calculator assistant. You MUST use tools for ALL arithmetic operations. Never calculate numbers directly.

IMPORTANT RULES:
1. You MUST call a tool for EVERY arithmetic operation - never compute numbers yourself
2. For multi-step calculations, call one tool at a time and wait for the result
3. After receiving a tool result, if more calculations are needed, call another tool
4. Only provide the final answer after all tool calls are complete

Example: For "What is 5 + 3 multiplied by 2?" call add(5, 3), then multiply(8, 2), then answer."""

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
    print("CALCULATOR TOOL CALLING EXAMPLES")
    print("=" * 70)

    for i, question in enumerate(test_questions, 1):
        print(f"\n[Question {i}] {question}")

        # Build conversation
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Maximum 10 turns to avoid infinite loops
        for turn in range(10):
            # Add user message (first turn only)
            if turn == 0:
                messages.append({"role": "user", "content": question})

            # Generate model response
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=TOOL_SPECS,
            )
            inputs = tokenizer([text], return_tensors="pt").to(model.device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                )

            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=False
            )

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

                # Continue to let model process the result
                continue
            else:
                # No tool call - this is the final answer
                # Clean up the response
                final_response = response.replace("<|im_end|>", "").strip()
                print(f"  Answer: {final_response}")
                break

        print()


if __name__ == "__main__":
    main()
