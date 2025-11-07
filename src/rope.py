"""
RoPE (Rotary Position Embeddings) for Qwen3 4B

RoPE encodes position information by rotating query and key vectors.
Instead of adding position embeddings (like original Transformer),
RoPE applies a rotation matrix based on the token's position.
"""

import torch
import torch.nn as nn


class RoPE(nn.Module):
    """
    Rotary Position Embeddings (RoPE)

    Applies position-dependent rotations to query and key vectors in attention.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float) -> None:
        """
        Initialize RoPE

        Args:
            head_dim: Dimension of each attention head (128 for Qwen3 4B)
            max_seq_len: Maximum sequence length (262144 for Qwen3 4B)
            theta: Base for the rotation frequencies (5000000 for Qwen3 4B)
        """
        super().__init__()
        self.head_dim: int = head_dim
        self.max_seq_len: int = max_seq_len
        self.theta: float = theta

        # Precompute the rotation frequencies
        # For each pair of dimensions, we have a different frequency
        # Shape: (head_dim // 2,)
        # Compute in float32 for accuracy, then convert to bfloat16 to match model weights
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, head_dim, 2, device="cpu") / head_dim)
        )
        inv_freq = inv_freq.to(torch.bfloat16)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embeddings to input tensor (HuggingFace style)

        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)
            position_ids: Position indices, shape (batch_size, seq_len) or (seq_len,)

        Returns:
            Rotated tensor of same shape as input
        """
        batch_size, num_heads, seq_len, head_dim = x.shape

        # Ensure position_ids has correct shape: (seq_len,)
        if position_ids.dim() == 2:
            position_ids = position_ids[0]  # (seq,)

        # Convert position_ids to match inv_freq dtype
        position_ids = position_ids.to(self.inv_freq.dtype)

        # Compute the rotation angles for each position
        freqs = torch.outer(position_ids, self.inv_freq)  # (seq, head_dim//2)

        # Create cos and sin - duplicate to match full head_dim
        cos = freqs.cos()  # (seq, head_dim//2)
        sin = freqs.sin()  # (seq, head_dim//2)

        # Repeat cos/sin to match head_dim by interleaving
        # HuggingFace expects cos/sin to have shape (seq, head_dim)
        cos = torch.cat([cos, cos], dim=-1)  # (seq, head_dim)
        sin = torch.cat([sin, sin], dim=-1)  # (seq, head_dim)

        # Broadcast to match x shape: (1, 1, seq, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply HuggingFace's rotation: q_embed = (q * cos) + (rotate_half(q) * sin)
        # rotate_half swaps and negates second half
        x_rotated = (x * cos) + (self._rotate_half(x) * sin)

        return x_rotated

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input (HuggingFace implementation)"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
