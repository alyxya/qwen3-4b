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
        self.head_dim = config["head_dim"]

        # Use meta device to avoid allocating memory for initial weights
        # that will be immediately overwritten by pretrained weights
        with torch.device("meta"):
            self.embed_tokens = Embedding(self.vocab_size, self.d_model)
            self.layers = nn.ModuleList([
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
            ])
            self.norm = RMSNorm(self.d_model, eps=config["rms_norm_eps"])

        # Load pretrained weights
        print("Loading pretrained weights from HuggingFace...")
        self._load_pretrained_weights(repo_id)
        print("Weights loaded successfully!")

    def _load_pretrained_weights(self, repo_id: str) -> None:
        """Load pretrained weights from HuggingFace and map them to model parameters"""
        hf_weights = load_weights(repo_id)

        # Map HuggingFace parameter names to our model's parameter names
        # HuggingFace uses "model." prefix which we need to strip
        mapped_weights = {}
        for name, tensor in hf_weights.items():
            if name.startswith("model."):
                new_name = name[6:]  # Remove "model." prefix
                # Map embed_tokens.weight to embed_tokens.embedding.weight
                if new_name == "embed_tokens.weight":
                    new_name = "embed_tokens.embedding.weight"
                mapped_weights[new_name] = tensor
            else:
                mapped_weights[name] = tensor

        # Load weights directly into the model
        # assign=True replaces the meta tensor parameters with the loaded tensors
        # (no copying or extra allocation needed)
        # strict=False allows missing keys (lm_head will be handled separately)
        missing_keys, unexpected_keys = self.load_state_dict(
            mapped_weights, strict=False, assign=True
        )

        # Handle lm_head with weight tying (shares weights with embed_tokens)
        # Qwen3 uses weight tying, so lm_head uses the same weights as embed_tokens
        self.lm_head = self.embed_tokens.embedding.weight

        if missing_keys:
            # Filter out lm_head since we handle it with weight tying
            missing_keys = [k for k in missing_keys if "lm_head" not in k]
            if missing_keys:
                print(f"Warning: Missing keys in state dict: {missing_keys}")

        if unexpected_keys:
            print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_k: list[torch.Tensor] | None = None,
        cache_v: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        hidden_states = self.embed_tokens(input_ids)  # (batch, seq, dim)

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
            )  # hidden_states: (batch, seq, dim)
            new_cache_k.append(new_k)
            new_cache_v.append(new_v)

        hidden_states = self.norm(hidden_states)  # (batch, seq, dim)
        logits = torch.einsum("bsd,vd->bsv", hidden_states, self.lm_head)  # (batch, seq, vocab)

        return logits, new_cache_k, new_cache_v

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        cache_k: list[torch.Tensor] | None = None,
        cache_v: list[torch.Tensor] | None = None,
    ) -> tuple[list[int], list[torch.Tensor], list[torch.Tensor]]:
        """
        Generate tokens autoregressively from input token IDs

        Args:
            input_ids: List of input token IDs to condition generation on
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (1.0 = no change, < 1.0 = more deterministic, > 1.0 = more random)
            top_k: If set, only sample from top k tokens (None = no filtering)
            top_p: If set, nucleus sampling - sample from smallest set of tokens with cumulative probability >= top_p
            cache_k: Optional existing KV cache (key) to continue generation from
            cache_v: Optional existing KV cache (value) to continue generation from

        Returns:
            Tuple of (generated_ids, new_cache_k, new_cache_v)
            - generated_ids: List of ONLY newly generated token IDs (does NOT include input_ids)
            - new_cache_k: Updated key cache (includes input_ids + generated_ids)
            - new_cache_v: Updated value cache (includes input_ids + generated_ids)
        """

        # Track token IDs for processing
        current_ids = input_ids.copy()
        # Track only the newly generated tokens
        new_tokens = []

        # Generate tokens one at a time
        with torch.no_grad():
            for i in range(max_new_tokens):
                # Prepare input tensor
                if i == 0 and cache_k is None:
                    # Prefill phase - process entire prompt (only if no existing cache)
                    input_tensor = torch.tensor([current_ids])
                else:
                    # Decode phase - process single token with cache
                    input_tensor = torch.tensor([[current_ids[-1]]])

                # Forward pass
                logits, cache_k, cache_v = self(input_tensor, cache_k=cache_k, cache_v=cache_v)

                # Get logits for the last token
                next_token_logits = logits[0, -1, :]  # (vocab_size,)

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-k filtering if specified
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    # Set all other logits to -inf
                    next_token_logits = torch.full_like(next_token_logits, float("-inf"))
                    next_token_logits[top_k_indices] = top_k_logits

                # Convert to probabilities
                probs = torch.softmax(next_token_logits, dim=-1)

                # Apply top-p (nucleus) sampling if specified
                if top_p is not None:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least one token
                    sorted_indices_to_remove[0] = False

                    # Create mask for original indices
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs[indices_to_remove] = 0.0
                    # Renormalize
                    probs = probs / probs.sum()

                # Sample from the distribution
                next_token_id = torch.multinomial(probs, num_samples=1).item()

                # Add to tracking lists
                current_ids.append(next_token_id)
                new_tokens.append(next_token_id)

        return new_tokens, cache_k, cache_v
