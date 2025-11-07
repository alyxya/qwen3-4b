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
        """Load pretrained weights from HuggingFace and map them to model parameters"""
        hf_weights = load_weights(repo_id)

        # Map HuggingFace parameter names to our model's parameter names
        # HuggingFace uses "model." prefix which we need to strip
        mapped_weights = {}
        for name, tensor in hf_weights.items():
            if name.startswith("model."):
                new_name = name[6:]  # Remove "model." prefix
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
        self.lm_head = self.embed_tokens.weight

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
        logits = torch.einsum(
            "bsd,vd->bsv", hidden_states, self.lm_head
        )  # (batch, seq, vocab)

        return logits, new_cache_k, new_cache_v

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        cache_k: list[torch.Tensor] | None = None,
        cache_v: list[torch.Tensor] | None = None,
        stop_token_ids: list[int] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        Generate tokens autoregressively

        Args:
            input_ids: Tensor of ALL token IDs in the sequence so far (prompt + any previous generations).
                       Shape: (batch, seq_len) or (seq_len,) - will be converted to (1, seq_len) if 1D.
                       The cache length is compared with input_ids to determine what needs processing.
            max_new_tokens: Maximum number of new tokens to generate (may stop earlier if stop token is encountered)
            temperature: Sampling temperature (1.0 = no change, < 1.0 = more deterministic, > 1.0 = more random)
            top_k: If set, only sample from top k tokens (None = no filtering)
            top_p: If set, nucleus sampling - sample from smallest set of tokens with cumulative probability >= top_p
            cache_k: Optional existing KV cache (key) to continue generation from.
                     If provided, cache is truncated/extended to match input_ids.
            cache_v: Optional existing KV cache (value) to continue generation from
            stop_token_ids: Optional list of token IDs to stop generation on (e.g., [im_end_id, endoftext_id]).
                           If a generated token matches any ID in this list, generation stops immediately.

        Returns:
            Tuple of (generated_ids, new_cache_k, new_cache_v)
            - generated_ids: Tensor of ONLY newly generated token IDs (does NOT include input_ids).
                           Shape: (batch, num_generated) where num_generated <= max_new_tokens.
                           Generation stops early if a stop token is encountered (if stop_token_ids provided).
            - new_cache_k: Updated key cache (includes all tokens from input_ids + generated tokens)
            - new_cache_v: Updated value cache (includes all tokens from input_ids + generated tokens)

        Examples:
            # Simple generation
            input_ids = torch.tensor([[1, 2, 3]])
            new_tokens, cache_k, cache_v = model.generate(input_ids, max_new_tokens=5)
            all_tokens = torch.cat([input_ids, new_tokens], dim=1)  # shape: (1, 8)

            # Continue generation - pass full sequence
            conversation = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
            more_tokens, cache_k, cache_v = model.generate(
                conversation, max_new_tokens=5, cache_k=cache_k, cache_v=cache_v
            )
            conversation = torch.cat([conversation, more_tokens], dim=1)

            # Multi-turn chat - add new context
            conversation = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])  # system + response_1
            user_msg = torch.tensor([[9, 10]])  # add user message
            conversation = torch.cat([conversation, user_msg], dim=1)
            response_2, cache_k, cache_v = model.generate(
                conversation, max_new_tokens=5, cache_k=cache_k, cache_v=cache_v
            )
        """
        # Validate inputs
        if input_ids is None or input_ids.numel() == 0:
            raise ValueError("input_ids must be provided and non-empty")

        # Convert 1D tensor to 2D (batch, seq)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        batch_size, input_len = input_ids.shape

        # Get current cache length
        cache_len = 0 if cache_k is None else cache_k[0].shape[2]

        # Truncate cache if it's >= input length
        # This ensures we always reprocess at least the last token to get fresh logits
        if cache_len >= input_len:
            truncate_to = max(0, input_len - 1)
            if cache_k is not None and truncate_to > 0:
                cache_k = [k[:, :, :truncate_to, :] for k in cache_k]
                cache_v = [v[:, :, :truncate_to, :] for v in cache_v]
            elif truncate_to == 0:
                # Start fresh if input is just 1 token
                cache_k = None
                cache_v = None
            cache_len = truncate_to

        # Process uncached suffix (always includes at least the last token)
        uncached_tokens = input_ids[:, cache_len:]  # (batch, uncached_len)

        with torch.no_grad():
            # Process uncached tokens
            if cache_k is None and uncached_tokens.shape[1] > 0:
                # Prefill: process all uncached tokens at once (efficient)
                logits, cache_k, cache_v = self(
                    uncached_tokens, cache_k=None, cache_v=None
                )
            else:
                # Add new tokens incrementally to existing cache
                for i in range(uncached_tokens.shape[1]):
                    single_token = uncached_tokens[:, i : i + 1]  # (batch, 1)
                    logits, cache_k, cache_v = self(
                        single_token, cache_k=cache_k, cache_v=cache_v
                    )

            # Now cache contains all input_ids and we have logits from the last token
            # Generate new tokens one at a time
            new_tokens_list = []
            for _ in range(max_new_tokens):
                # Get logits for the last token (support batch_size=1 for now)
                next_token_logits = logits[0, -1, :]  # (vocab_size,)

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-k filtering if specified
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    # Set all other logits to -inf
                    next_token_logits = torch.full_like(
                        next_token_logits, float("-inf")
                    )
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

                # Add to newly generated tokens
                new_tokens_list.append(next_token_id)

                # Check if we hit a stop token (e.g., <|im_end|> or <|endoftext|>)
                if stop_token_ids is not None and next_token_id in stop_token_ids:
                    break

                # Process this token to get next logits
                input_tensor = torch.tensor([[next_token_id]])
                logits, cache_k, cache_v = self(
                    input_tensor, cache_k=cache_k, cache_v=cache_v
                )

        # Convert list of tokens to tensor (batch, num_generated)
        if len(new_tokens_list) > 0:
            new_tokens = torch.tensor([new_tokens_list])
        else:
            new_tokens = torch.tensor([[]]).long()

        return new_tokens, cache_k, cache_v
