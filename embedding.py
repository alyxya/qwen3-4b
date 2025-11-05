"""
Token embedding layer for Qwen3 4B
"""

import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    Token embedding layer - converts token IDs to dense vectors

    This is a simple lookup table: embeddings[token_id] -> vector of size d_model
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

        # Use nn.Embedding to match HuggingFace naming convention
        # This creates a parameter called 'weight' which matches the expected name
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings

        Args:
            token_ids: Tensor of token IDs, shape (batch_size, seq_len) or (seq_len,)

        Returns:
            Embeddings tensor of shape (batch_size, seq_len, d_model) or (seq_len, d_model)
        """
        return self.embedding(token_ids)  # (batch, seq, dim) = (batch, seq, 2560)
