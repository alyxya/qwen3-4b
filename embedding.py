"""
Embedding layer for Qwen3 4B - Custom implementation
"""

import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    Token embedding layer - converts token IDs to dense vectors

    This is a simple lookup table: embeddings[token_id] -> vector of size d_model

    Reimplemented from scratch to understand what nn.Embedding does under the hood.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        """
        Initialize embedding layer

        Args:
            vocab_size: Number of tokens in vocabulary (151936 for Qwen3 4B)
            d_model: Embedding dimension (2560 for Qwen3 4B)
        """
        super().__init__()
        self.vocab_size: int = vocab_size
        self.d_model: int = d_model

        # The embedding weight matrix: shape (vocab_size, d_model)
        # This is just a big lookup table where each row is a token's embedding vector
        # nn.Parameter tells PyTorch this is a learnable weight (even though we're only doing inference)
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings

        Args:
            token_ids: Tensor of token IDs, shape (batch_size, seq_len) or (seq_len,)

        Returns:
            Embeddings tensor of shape (batch_size, seq_len, d_model) or (seq_len, d_model)
        """
        # This is all nn.Embedding does: index into the weight matrix!
        # token_ids are used as indices to select rows from self.weight
        return self.weight[token_ids]
