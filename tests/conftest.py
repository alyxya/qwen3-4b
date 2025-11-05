"""Shared test fixtures"""

import json
import pytest
from huggingface_hub import hf_hub_download


@pytest.fixture
def config():
    """Load model config for testing"""
    repo_id = "Qwen/Qwen3-4B-Instruct-2507"
    config_path = hf_hub_download(repo_id, "config.json")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)
