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


@pytest.fixture(scope="session")
def model():
    """
    Load the full Qwen3 model (shared across entire test session)

    Only loaded once when first needed by any test that uses this fixture.
    Tests that don't use this fixture won't trigger model loading.
    """
    from src.model import Qwen3Model

    return Qwen3Model(
        repo_id="Qwen/Qwen3-4B-Instruct-2507",
        device="cpu"
    )


@pytest.fixture(scope="session")
def tokenizer():
    """
    Load the tokenizer (shared across entire test session)

    Only loaded once when first needed by any test that uses this fixture.
    """
    from src.tokenizer import Tokenizer

    return Tokenizer(repo_id="Qwen/Qwen3-4B-Instruct-2507")
