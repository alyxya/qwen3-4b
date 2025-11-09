"""Qwen3 4B model implementation"""

import json
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .embedding import Embedding
from .transformer_block import TransformerBlock
from .rmsnorm import RMSNorm
from .load_weights import load_weights


class Qwen3Model(nn.Module):
    """Qwen3 4B Language Model"""

    def __init__(
        self,
        repo_id: str,
        device: str,
    ) -> None:
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
        self.head_dim = config["head_dim"]
        self.device = device

        # Use meta device to avoid allocating memory for initial weights
        # that will be immediately overwritten by pretrained weights
        with torch.device("meta"):
            self.embed_tokens = Embedding(self.vocab_size, self.d_model)
            self.layers = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=self.d_model,
                        num_heads=self.num_heads,
                        num_kv_heads=self.num_kv_heads,
                        head_dim=self.head_dim,
                        intermediate_size=config["intermediate_size"],
                        max_position_embeddings=config["max_position_embeddings"],
                        rope_theta=config["rope_theta"],
                        rms_norm_eps=config["rms_norm_eps"],
                    )
                    for _ in range(self.num_layers)
                ]
            )
            self.norm = RMSNorm(self.d_model, eps=config["rms_norm_eps"])

        # Load pretrained weights
        self._load_pretrained_weights(repo_id)

    def _load_pretrained_weights(self, repo_id: str) -> None:
        """Load pretrained weights from HuggingFace"""
        hf_weights = load_weights(repo_id, device=self.device)

        # Strip "model." prefix from HuggingFace parameter names
        mapped_weights = {
            name[6:] if name.startswith("model.") else name: tensor
            for name, tensor in hf_weights.items()
        }

        # Load weights (assign=True for efficient meta device → real device transfer)
        missing_keys, unexpected_keys = self.load_state_dict(
            mapped_weights, strict=False, assign=True
        )

        # Weight tying: lm_head shares weights with embed_tokens
        self.lm_head = self.embed_tokens.weight

        # Warn about mismatches (lm_head is expected to be missing due to weight tying)
        missing_keys = [k for k in missing_keys if "lm_head" not in k]
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")

        # Move entire model to target device (handles RoPE buffers and any other CPU tensors)
        self.to(self.device)

    def forward(
        self,
        input_ids: torch.Tensor,  # (batch, seq)
        cache_k: list[torch.Tensor] | None = None,  # [(batch, num_kv_heads, cache_len, head_dim)] * num_layers
        cache_v: list[torch.Tensor] | None = None,  # [(batch, num_kv_heads, cache_len, head_dim)] * num_layers
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass through the model

        Returns:
            logits: (batch, seq, vocab_size)
            cache_k: [(batch, num_kv_heads, new_cache_len, head_dim)] * num_layers
            cache_v: [(batch, num_kv_heads, new_cache_len, head_dim)] * num_layers
        """
        hidden_states = self.embed_tokens(input_ids)  # (batch, seq, d_model)

        if cache_k is None:
            cache_k = [None] * self.num_layers
        if cache_v is None:
            cache_v = [None] * self.num_layers

        new_cache_k = []
        new_cache_v = []

        for layer_idx, layer in enumerate(self.layers):
            hidden_states, new_k, new_v = layer(
                hidden_states,  # (batch, seq, d_model)
                cache_k=cache_k[layer_idx],
                cache_v=cache_v[layer_idx],
            )
            # hidden_states: (batch, seq, d_model)
            # new_k: (batch, num_kv_heads, cache_len + seq, head_dim)
            # new_v: (batch, num_kv_heads, cache_len + seq, head_dim)
            new_cache_k.append(new_k)
            new_cache_v.append(new_v)

        hidden_states = self.norm(hidden_states)  # (batch, seq, d_model)
        logits = torch.einsum(
            "bsd,vd->bsv", hidden_states, self.lm_head
        )  # (batch, seq, vocab_size)

        return logits, new_cache_k, new_cache_v

    def generate(
        self,
        input_ids: torch.Tensor,  # (batch, seq) or (seq,)
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        cache_k: list[torch.Tensor] | None = None,  # [(batch, num_kv_heads, cache_len, head_dim)] * num_layers
        cache_v: list[torch.Tensor] | None = None,  # [(batch, num_kv_heads, cache_len, head_dim)] * num_layers
        stop_token_ids: list[int] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Generate tokens autoregressively with optional KV caching

        Args:
            input_ids: Full sequence (batch, seq_len) or (seq_len,). Pass complete conversation each time.
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_k: Sample from top k tokens only
            top_p: Nucleus sampling threshold
            cache_k/cache_v: Optional KV cache for continuation
            stop_token_ids: Stop if any of these tokens are generated

        Returns:
            new_tokens: (batch, num_generated) - only the newly generated tokens
            cache_k: [(batch, num_kv_heads, new_cache_len, head_dim)] * num_layers
            cache_v: [(batch, num_kv_heads, new_cache_len, head_dim)] * num_layers
        """
        if input_ids.numel() == 0:
            raise ValueError("input_ids cannot be empty")

        # Move input to model device
        input_ids = input_ids.to(self.device)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # (seq,) -> (1, seq)

        batch_size, input_len = input_ids.shape  # (batch, seq)
        cache_len = 0 if cache_k is None else cache_k[0].shape[2]

        # Truncate cache if needed to ensure at least last token is reprocessed
        if cache_len >= input_len:
            truncate_to = max(0, input_len - 1)
            if truncate_to > 0:
                cache_k = [k[:, :, :truncate_to, :] for k in cache_k]
                cache_v = [v[:, :, :truncate_to, :] for v in cache_v]
            else:
                cache_k = cache_v = None
            cache_len = truncate_to

        next_input = input_ids[:, cache_len:]  # (batch, uncached_len)
        new_tokens = torch.empty((batch_size, 0), dtype=torch.long, device=input_ids.device)  # (batch, 0)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, cache_k, cache_v = self(next_input, cache_k, cache_v)  # logits: (batch, seq, vocab_size)
                next_token = self._sample_token(logits[0, -1, :], temperature, top_k, top_p)  # (1, 1)
                new_tokens = torch.cat([new_tokens, next_token], dim=1)  # (batch, num_generated)

                if stop_token_ids and next_token.item() in stop_token_ids:
                    break

                next_input = next_token  # (1, 1)

        return new_tokens, cache_k, cache_v

    def _sample_token(
        self, logits: torch.Tensor, temperature: float, top_k: int | None, top_p: float | None
    ) -> torch.Tensor:
        """Sample token with temperature, top-k, and top-p filtering

        Args:
            logits: (vocab_size,) - logits for next token
            temperature: Sampling temperature (0 for greedy/deterministic)
            top_k: Sample from top k tokens only
            top_p: Nucleus sampling threshold

        Returns:
            next_token: (1, 1) - sampled token ID
        """
        # Greedy decoding (deterministic) when temperature is 0
        if temperature == 0.0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True).unsqueeze(0)  # (1,) -> (1, 1)
            return next_token

        if temperature != 1.0:
            logits = logits / temperature  # (vocab_size,)

        if top_k:
            top_k_logits, top_k_indices = torch.topk(logits, top_k)  # (top_k,), (top_k,)
            logits = torch.full_like(logits, float("-inf"))  # (vocab_size,)
            logits[top_k_indices] = top_k_logits  # (vocab_size,)

        probs = torch.softmax(logits, dim=-1)  # (vocab_size,)

        if top_p:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)  # (vocab_size,), (vocab_size,)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # (vocab_size,)
            sorted_indices_to_remove = cumulative_probs > top_p  # (vocab_size,)
            sorted_indices_to_remove[0] = False
            probs[sorted_indices[sorted_indices_to_remove]] = 0.0  # (vocab_size,)
            probs = probs / probs.sum()  # (vocab_size,)

        return torch.multinomial(probs, num_samples=1).unsqueeze(0)  # (1,) -> (1, 1)
