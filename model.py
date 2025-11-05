"""
Complete Qwen3 4B model implementation - Built for inference
"""

import json
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from embedding import Embedding
from transformer_block import TransformerBlock
from rmsnorm import RMSNorm


def load_config() -> dict:
    """Load the model configuration from HuggingFace"""
    repo_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    config_path: str = hf_hub_download(repo_id, "config.json")

    with open(config_path, "r", encoding="utf-8") as f:
        config: dict = json.load(f)

    return config


class Qwen3Model(nn.Module):
    """
    Complete Qwen3 4B Language Model

    Architecture:
    1. Token Embedding
    2. 36 Transformer Blocks (each with Attention + MLP)
    3. Final RMSNorm
    4. Output Projection (LM head)
    """

    def __init__(
        self,
        vocab_size: int = 151936,
        d_model: int = 2560,
        num_layers: int = 36,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        intermediate_size: int = 9728,
        max_position_embeddings: int = 262144,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 5000000.0,
    ) -> None:
        """
        Initialize Qwen3 4B model

        Args:
            vocab_size: Size of vocabulary (151936 for Qwen3 4B)
            d_model: Model dimension (2560 for Qwen3 4B)
            num_layers: Number of transformer layers (36 for Qwen3 4B)
            num_heads: Number of query attention heads (32 for Qwen3 4B)
            num_kv_heads: Number of key/value heads for GQA (8 for Qwen3 4B)
            intermediate_size: MLP hidden dimension (9728 for Qwen3 4B)
            max_position_embeddings: Maximum sequence length (262144 for Qwen3 4B)
            rms_norm_eps: RMSNorm epsilon (1e-6 for Qwen3 4B)
            rope_theta: RoPE base frequency (5000000 for Qwen3 4B)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        # Calculate head dimension
        self.head_dim = d_model // num_heads  # 2560 // 32 = 80

        # 1. Token embedding layer
        self.embed_tokens = Embedding(vocab_size=vocab_size, d_model=d_model)

        # 2. Stack of transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=self.head_dim,
                intermediate_size=intermediate_size,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
            )
            for _ in range(num_layers)
        ])

        # 3. Final layer normalization
        self.norm = RMSNorm(d_model, eps=rms_norm_eps)

        # 4. Output projection (LM head) - maps from d_model to vocab_size
        # Note: In many models, this shares weights with the embedding layer (weight tying)
        # For now, we'll keep them separate
        self.lm_head = nn.Parameter(torch.randn(vocab_size, d_model))

    @classmethod
    def from_pretrained(cls, repo_id: str = "Qwen/Qwen3-4B-Instruct-2507") -> "Qwen3Model":
        """
        Create model from HuggingFace config

        Args:
            repo_id: HuggingFace model repository ID

        Returns:
            Initialized Qwen3Model with architecture matching the pretrained model
        """
        config_path = hf_hub_download(repo_id, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        return cls(
            vocab_size=config["vocab_size"],
            d_model=config["hidden_size"],
            num_layers=config["num_hidden_layers"],
            num_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            intermediate_size=config["intermediate_size"],
            max_position_embeddings=config["max_position_embeddings"],
            rms_norm_eps=config["rms_norm_eps"],
            rope_theta=config["rope_theta"],
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_k: list[torch.Tensor] | None = None,
        cache_v: list[torch.Tensor] | None = None,
        return_logits: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        Forward pass through the model

        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            cache_k: List of cached key tensors for each layer (length = num_layers) or None
            cache_v: List of cached value tensors for each layer (length = num_layers) or None
            return_logits: If True, return logits. If False, return hidden states before lm_head.

        Returns:
            Tuple of (output, new_cache_k, new_cache_v)
            - output: Logits of shape (batch_size, seq_len, vocab_size) if return_logits=True,
                     else hidden states of shape (batch_size, seq_len, d_model)
            - new_cache_k: List of updated key caches for each layer
            - new_cache_v: List of updated value caches for each layer
        """
        # 1. Token embedding
        # input_ids: (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        hidden_states = self.embed_tokens(input_ids)

        # Initialize cache lists if not provided
        if cache_k is None:
            cache_k = [None] * self.num_layers
        if cache_v is None:
            cache_v = [None] * self.num_layers

        # Storage for new caches
        new_cache_k = []
        new_cache_v = []

        # 2. Pass through all transformer blocks
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, new_k, new_v = layer(
                hidden_states,
                cache_k=cache_k[layer_idx],
                cache_v=cache_v[layer_idx],
            )
            new_cache_k.append(new_k)
            new_cache_v.append(new_v)

        # 3. Final normalization
        hidden_states = self.norm(hidden_states)

        # 4. Project to vocabulary (LM head)
        if return_logits:
            # hidden_states: (batch_size, seq_len, d_model) - "bsd"
            # lm_head: (vocab_size, d_model) - "vd"
            # logits: (batch_size, seq_len, vocab_size) - "bsv"
            logits = torch.einsum("bsd,vd->bsv", hidden_states, self.lm_head)
            return logits, new_cache_k, new_cache_v
        else:
            return hidden_states, new_cache_k, new_cache_v
