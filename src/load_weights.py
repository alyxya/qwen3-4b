"""
Load model weights from HuggingFace for Qwen3 4B
"""

from huggingface_hub import hf_hub_download
from safetensors import safe_open
import torch


def list_weight_files(repo_id: str) -> list[str]:
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


def _load_weights_impl(repo_id: str, device: torch.device) -> dict[str, torch.Tensor]:
    """
    Implementation of weight loading (used by both local and remote paths)

    Args:
        repo_id: HuggingFace model repository ID
        device: Device to load weights to

    Returns:
        Dictionary mapping parameter names to tensors
    """
    weight_files = list_weight_files(repo_id)

    all_weights: dict[str, torch.Tensor] = {}

    for weight_file in weight_files:
        weight_path = hf_hub_download(repo_id, weight_file)

        # Load tensors from safetensors file
        # Weights are natively bfloat16, loaded directly to target device
        # Note: safe_open requires device as string, not torch.device object
        with safe_open(weight_path, framework="pt", device=str(device)) as f:
            for key in f.keys():
                all_weights[key] = f.get_tensor(key)

    return all_weights


def load_weights(
    repo_id: str,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    """
    Load all model weights from HuggingFace

    Automatically detects mycelya devices and loads weights directly on remote GPU
    without transferring through local machine.

    Args:
        repo_id: HuggingFace model repository ID
        device: Device to load weights to ("cpu", "mps", "cuda", mycelya device, etc.)

    Returns:
        Dictionary mapping parameter names to tensors
        Example: {"model.embed_tokens.weight": tensor(...), ...}
    """
    # Normalize to device object
    if isinstance(device, str):
        device = torch.device(device)

    # Check if this is a mycelya device
    if device.type == "mycelya":
        import mycelya_torch

        # Create remote version of weight loader and execute on remote GPU
        # mycelya_torch will automatically map the device to the remote machine
        remote_load = mycelya_torch.remote(_load_weights_impl)
        return remote_load(repo_id, device)
    else:
        # Local loading
        return _load_weights_impl(repo_id, device)
