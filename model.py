"""
Minimal Qwen3 4B model implementation - Built for inference
"""

import json
from huggingface_hub import hf_hub_download


def load_config() -> dict:
    """Load the model configuration from HuggingFace"""
    repo_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    config_path: str = hf_hub_download(repo_id, "config.json")

    with open(config_path, "r", encoding="utf-8") as f:
        config: dict = json.load(f)

    return config


if __name__ == "__main__":
    # Load and display the config
    config = load_config()

    print("Qwen3 4B Configuration:")
    print("=" * 50)

    # Display key parameters
    important_keys = [
        "hidden_size",  # d_model = 2560
        "num_hidden_layers",  # 36
        "num_attention_heads",  # 32
        "num_key_value_heads",  # 8
        "intermediate_size",  # 9728, MLP dimension size
        "vocab_size",  # 151936
        "max_position_embeddings",  # 262144, max context size
        "rms_norm_eps",  # 1e-06
        "rope_theta",  # 5000000
    ]

    for key in important_keys:
        if key in config:
            print(f"{key:30s}: {config[key]}")

    print("\nFull config:")
    print(json.dumps(config, indent=2))
