"""
Load model weights from HuggingFace for Qwen3 4B
"""

from huggingface_hub import hf_hub_download
from safetensors import safe_open
import torch


def list_weight_files(repo_id: str = "Qwen/Qwen3-4B-Instruct-2507") -> list[str]:
    """
    List all weight files in the HuggingFace repo

    Returns:
        List of safetensors filenames
    """
    # Qwen3 4B weights are split across multiple files
    # They follow the pattern: model-00001-of-00004.safetensors, etc.
    # We need to check how many files there are

    # For now, let's just check the first file
    try:
        first_file = hf_hub_download(repo_id, "model.safetensors.index.json")

        import json
        with open(first_file, "r") as f:
            index = json.load(f)

        # Get unique weight files
        weight_files = list(set(index["weight_map"].values()))
        return sorted(weight_files)
    except Exception:
        # If no index, might be a single file
        return ["model.safetensors"]


def load_weights(repo_id: str = "Qwen/Qwen3-4B-Instruct-2507") -> dict[str, torch.Tensor]:
    """
    Load all model weights from HuggingFace

    Returns:
        Dictionary mapping parameter names to tensors
        Example: {"model.embed_tokens.weight": tensor(...), ...}
    """
    weight_files = list_weight_files(repo_id)

    all_weights: dict[str, torch.Tensor] = {}

    for weight_file in weight_files:
        weight_path = hf_hub_download(repo_id, weight_file)

        # Load tensors from safetensors file
        with safe_open(weight_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                all_weights[key] = f.get_tensor(key)

    return all_weights
