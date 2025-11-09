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

        # Precompute rotation frequencies in float32 for numerical precision
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device="cpu") / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embeddings

        Args:
            x: (batch_size, num_heads, seq_len, head_dim)
            position_ids: (batch_size, seq_len) or (seq_len,)

        Returns:
            Rotated tensor, same shape as input
        """
        batch_size, num_heads, seq_len, head_dim = x.shape

        # Flatten position_ids if needed
        if position_ids.dim() == 2:
            position_ids = position_ids[0]

        # Compute rotation frequencies (all in float32 for precision)
        position_ids = position_ids.to(torch.float32)
        freqs = torch.outer(position_ids, self.inv_freq)  # (seq, head_dim//2)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).unsqueeze(0).unsqueeze(0)

        # Apply rotation in float32 (better precision than converting cos/sin to bfloat16)
        # PyTorch automatically promotes bfloat16 * float32 -> float32
        x_rotated = (x * cos) + (self._rotate_half(x) * sin)

        return x_rotated.to(x.dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims: [x1, x2] -> [-x2, x1]"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
