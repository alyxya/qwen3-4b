"""Qwen3 4B model implementation"""

import json
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from embedding import Embedding
from transformer_block import TransformerBlock
from rmsnorm import RMSNorm


class Qwen3Model(nn.Module):
    """Qwen3 4B Language Model"""

    def __init__(self, repo_id: str = "Qwen/Qwen3-4B-Instruct-2507") -> None:
        super().__init__()

        config_path = hf_hub_download(repo_id, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.vocab_size = config["vocab_size"]
        self.d_model = config["hidden_size"]
        self.num_layers = config["num_hidden_layers"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.max_position_embeddings = config["max_position_embeddings"]
        self.head_dim = self.d_model // self.num_heads

        self.embed_tokens = Embedding(self.vocab_size, self.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                intermediate_size=config["intermediate_size"],
                rope_theta=config["rope_theta"],
                rms_norm_eps=config["rms_norm_eps"],
            )
            for _ in range(self.num_layers)
        ])
        self.norm = RMSNorm(self.d_model, eps=config["rms_norm_eps"])
        self.lm_head = nn.Parameter(torch.randn(self.vocab_size, self.d_model))

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_k: list[torch.Tensor] | None = None,
        cache_v: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        hidden_states = self.embed_tokens(input_ids)

        if cache_k is None:
            cache_k = [None] * self.num_layers
        if cache_v is None:
            cache_v = [None] * self.num_layers

        new_cache_k = []
        new_cache_v = []

        for layer_idx, layer in enumerate(self.layers):
            hidden_states, new_k, new_v = layer(
                hidden_states,
                cache_k=cache_k[layer_idx],
                cache_v=cache_v[layer_idx],
            )
            new_cache_k.append(new_k)
            new_cache_v.append(new_v)

        hidden_states = self.norm(hidden_states)
        logits = torch.einsum("bsd,vd->bsv", hidden_states, self.lm_head)

        return logits, new_cache_k, new_cache_v
