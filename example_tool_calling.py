#!/usr/bin/env python3
"""
Example: Tool Calling with Qwen3

This demonstrates the full tool calling flow:
1. Define tools (functions)
2. Format conversation with tools
3. Model generates a tool call
4. Execute the tool
5. Return result to model
6. Model generates final answer
"""

import json
from src.tokenizer import Tokenizer


# Step 1: Define your actual Python functions
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Simulated weather API"""
    # In reality, this would call a real weather API
    weather_data = {
        "Paris": {"temperature": 22, "condition": "sunny"},
        "London": {"temperature": 15, "condition": "rainy"},
        "Tokyo": {"temperature": 28, "condition": "cloudy"},
    }
    return weather_data.get(location, {"temperature": 20, "condition": "unknown"})


def calculate(operation: str, a: float, b: float) -> float:
    """Simple calculator"""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else None,
    }
    return operations.get(operation, lambda x, y: None)(a, b)


# Step 2: Define tool specifications (JSON schema format)
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g., 'Paris', 'London'",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "calculate",
        "description": "Perform basic arithmetic operations",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                },
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["operation", "a", "b"],
        },
    },
]


def format_system_message_with_tools(tools: list[dict]) -> str:
    """Format system message following Qwen3's tool calling template"""
    tools_json = json.dumps(tools, indent=2)
    return f"""You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_json}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


def parse_tool_call(text: str) -> dict | None:
    """Extract tool call JSON from model output"""
    if "<tool_call>" in text and "</tool_call>" in text:
        start = text.find("<tool_call>") + len("<tool_call>")
        end = text.find("</tool_call>")
        tool_call_json = text[start:end].strip()
        try:
            return json.loads(tool_call_json)
        except json.JSONDecodeError:
            return None
    return None


def execute_tool(tool_call: dict) -> str:
    """Execute the requested tool and return result"""
    function_name = tool_call["name"]
    arguments = tool_call["arguments"]

    if function_name == "get_weather":
        result = get_weather(**arguments)
    elif function_name == "calculate":
        result = calculate(**arguments)
    else:
        result = {"error": f"Unknown function: {function_name}"}

    return json.dumps(result)


def main():
    print("=" * 70)
    print("Tool Calling Example with Qwen3")
    print("=" * 70)

    tokenizer = Tokenizer()

    # Example 1: Weather query
    print("\n### EXAMPLE 1: Weather Query ###\n")

    # Build conversation with tools
    messages = [
        {"role": "system", "content": format_system_message_with_tools(TOOLS)},
        {"role": "user", "content": "What's the weather like in Paris?"},
    ]

    # Show the formatted conversation
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    print("Formatted conversation sent to model:")
    print("-" * 70)
    print(formatted)
    print("-" * 70)

    # Simulate what the model would generate
    # (In reality, you'd call model.generate() here)
    model_response = """<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}
</tool_call>"""

    print("\nModel generates tool call:")
    print(model_response)

    # Parse and execute the tool
    tool_call = parse_tool_call(model_response)
    if tool_call:
        print(f"\nParsed tool call: {tool_call}")
        tool_result = execute_tool(tool_call)
        print(f"Tool execution result: {tool_result}")

        # Add tool response to conversation
        print("\nSending tool result back to model...")
        # Note: In Qwen's format, tool responses go in a user message
        tool_response_text = f"<tool_response>\n{tool_result}\n</tool_response>"
        print(tool_response_text)

        # Model would then generate final answer based on tool result
        print("\nModel would then generate:")
        print(
            "The weather in Paris is currently sunny with a temperature of 22°C."
        )

    # Example 2: Calculator query
    print("\n\n### EXAMPLE 2: Calculator Query ###\n")

    messages = [
        {"role": "system", "content": format_system_message_with_tools(TOOLS)},
        {"role": "user", "content": "What is 156 multiplied by 23?"},
    ]

    print("User asks: What is 156 multiplied by 23?")

    # Model would generate
    model_response = """<tool_call>
{"name": "calculate", "arguments": {"operation": "multiply", "a": 156, "b": 23}}
</tool_call>"""

    print(f"\nModel generates: {model_response}")

    tool_call = parse_tool_call(model_response)
    if tool_call:
        tool_result = execute_tool(tool_call)
        print(f"Tool result: {tool_result}")
        print(f"\nModel final answer: 156 × 23 = {json.loads(tool_result)}")

    # Show token IDs for tool tokens
    print("\n\n### SPECIAL TOKEN IDs ###\n")
    print(f"<tool_call>      -> ID {tokenizer.special_tokens['<tool_call>']}")
    print(f"</tool_call>     -> ID {tokenizer.special_tokens['</tool_call>']}")
    print(f"<tool_response>  -> ID {tokenizer.special_tokens['<tool_response>']}")
    print(f"</tool_response> -> ID {tokenizer.special_tokens['</tool_response>']}")

    # Show how it encodes
    print("\n### ENCODING EXAMPLE ###\n")
    text = "<tool_call>\n{\"name\": \"test\"}\n</tool_call>"
    token_ids = tokenizer.encode(text)
    print(f"Text: {text}")
    print(f"Token IDs: {token_ids}")
    print(
        f"Note: {tokenizer.special_tokens['<tool_call>']} and {tokenizer.special_tokens['</tool_call>']} are the special token IDs"
    )


if __name__ == "__main__":
    main()
