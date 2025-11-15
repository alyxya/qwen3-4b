"""
Example: Run Qwen3 on Modal serverless GPU

This example shows how to run Qwen3 inference entirely on Modal's serverless infrastructure.
All model loading and inference happens in the cloud - your local machine just triggers it.

Prerequisites:
    pip install modal
    modal setup

Usage:
    python modal_example.py
"""

import modal
import torch
import time

from src.model import Qwen3Model
from src.tokenizer import Tokenizer

# Create Modal app
app = modal.App("qwen3-inference")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "huggingface_hub",
        "safetensors",
        "regex",
    )
    # Add local src/ directory to the image so imports work
    .add_local_dir("src", remote_path="/root/src")
)


@app.function(
    image=image,
    gpu="A100",  # Use A100 GPU (can also use "T4", "A10G", etc.)
    timeout=600,  # 10 minute timeout
)
def run_qwen3_inference(prompt: str, max_new_tokens: int = 100, temperature: float = 0.7):
    """
    Run Qwen3 inference on Modal GPU.

    Args:
        prompt: Text prompt to generate from
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)

    Returns:
        Dictionary with prompt and generated text
    """

    print("Loading model on Modal GPU...")
    print("(Downloading ~8GB from HuggingFace)")

    # Load model and tokenizer
    device = torch.device("cuda")
    model = Qwen3Model(
        repo_id="Qwen/Qwen3-4B-Instruct-2507",
        device=device
    )
    tokenizer = Tokenizer(repo_id="Qwen/Qwen3-4B-Instruct-2507")

    print(f"✓ Model loaded on {device}")
    print(f"\nGenerating text for prompt: {prompt}")

    # Tokenize input
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate

    start_time = time.time()
    new_tokens, _, _ = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    duration = time.time() - start_time

    # Decode output
    generated_text = tokenizer.decode(new_tokens[0].cpu().tolist())

    print(f"✓ Generated {len(new_tokens)} tokens")

    return {
        "prompt": prompt,
        "generated": generated_text,
        "tokens_generated": len(new_tokens),
        "duration": duration,
    }


if __name__ == "__main__":
    print("🚀 Starting Qwen3 inference on Modal...")
    print("=" * 60)

    # Run the function using Modal's programmatic API
    with app.run():
        result = run_qwen3_inference.remote(
            prompt="Explain quantum computing in simple terms:",
            max_new_tokens=100,
            temperature=0.7,
        )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nPrompt: {result['prompt']}")
    print(f"\nGenerated ({result['tokens_generated']} tokens):")
    print(result['generated'])
    print(f"\nDuration: {result['duration']}")
    print("\n✓ Done!")
