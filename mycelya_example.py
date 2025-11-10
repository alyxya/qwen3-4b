"""
Example: Load Qwen3 on remote GPU with mycelya-torch

This example shows how to use mycelya-torch to load and run Qwen3 on a cloud GPU.
The weights are downloaded directly to the remote GPU without passing through your
local machine.

Prerequisites:
    pip install mycelya-torch
    modal setup

Usage:
    python mycelya_example.py
"""

import torch
import mycelya_torch
from src.model import Qwen3Model
from src.tokenizer import Tokenizer


def main():
    # 1. Create remote machine with A100 GPU
    print("Creating remote machine with A100 GPU...")
    machine = mycelya_torch.RemoteMachine(
        provider="modal",
        gpu_type="A100",
        gpu_count=1,
        packages=["torch", "huggingface_hub", "safetensors", "regex"],
    )

    # 2. Get mycelya device
    device = machine.device("cuda")
    print(f"Remote device: {device}")

    try:
        # 3. Load model - weights download directly to remote GPU!
        print("\nLoading model on remote GPU...")
        print("(HuggingFace downloads ~8GB directly to cloud GPU)")
        model = Qwen3Model(
            repo_id="Qwen/Qwen3-4B-Instruct-2507",
            device=device
        )
        tokenizer = Tokenizer(repo_id="Qwen/Qwen3-4B-Instruct-2507")
        print("✓ Model loaded on remote GPU!")

        # 4. Generate text - all operations execute on remote GPU
        print("\nGenerating text on remote GPU...")
        prompt = "Explain quantum computing in simple terms:"
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)

        new_tokens, _, _ = model.generate(
            input_tensor,
            max_new_tokens=100,
            temperature=0.7,
        )

        # 5. Decode result
        generated_text = tokenizer.decode(new_tokens[0].cpu().tolist())
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")

    finally:
        # 6. Clean up
        print("\nStopping machine...")
        machine.stop()
        print("✓ Done!")


if __name__ == "__main__":
    main()
