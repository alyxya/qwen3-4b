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

    def __init__(self, head_dim: int, max_seq_len: int = 262144, theta: float = 5000000.0) -> None:
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
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embeddings to input tensor

        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)
            position_ids: Position indices, shape (batch_size, seq_len) or (seq_len,)

        Returns:
            Rotated tensor of same shape as input
        """
        # x shape: (batch_size, num_heads, seq_len, head_dim)
        batch_size, num_heads, seq_len, head_dim = x.shape

        # Ensure position_ids has correct shape
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)  # (seq_len,) -> (1, seq_len)

        # Compute the rotation angles for each position
        # position_ids: (batch_size, seq_len)
        # inv_freq: (head_dim // 2,)
        # freqs: (batch_size, seq_len, head_dim // 2)
        freqs = torch.outer(position_ids[0], self.inv_freq)

        # Create the rotation matrix using cos and sin
        # We'll apply: [cos, -sin; sin, cos] rotation to pairs of dimensions
        cos = freqs.cos()  # (seq_len, head_dim // 2)
        sin = freqs.sin()  # (seq_len, head_dim // 2)

        # Reshape x to separate even and odd dimensions
        # Split head_dim into pairs: [x0, x1, x2, x3, ...] -> [[x0, x1], [x2, x3], ...]
        x_reshaped = x.reshape(batch_size, num_heads, seq_len, head_dim // 2, 2)

        # Extract even and odd elements
        x_even = x_reshaped[..., 0]  # (batch_size, num_heads, seq_len, head_dim // 2)
        x_odd = x_reshaped[..., 1]   # (batch_size, num_heads, seq_len, head_dim // 2)

        # Apply rotation: [cos*x_even - sin*x_odd, sin*x_even + cos*x_odd]
        # Broadcast cos/sin from (seq_len, head_dim//2) to match x shape
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim // 2)
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim // 2)

        x_rotated_even = cos * x_even - sin * x_odd
        x_rotated_odd = sin * x_even + cos * x_odd

        # Recombine into original shape
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        x_rotated = x_rotated.reshape(batch_size, num_heads, seq_len, head_dim)

        return x_rotated


if __name__ == "__main__":
    # Test RoPE
    from model import load_config

    config = load_config()

    print("Testing RoPE:")
    print("=" * 50)

    # Parameters from config
    num_heads: int = config["num_attention_heads"]  # 32
    head_dim: int = config["head_dim"]  # 128
    theta: float = config["rope_theta"]  # 5000000

    print(f"num_heads: {num_heads}")
    print(f"head_dim: {head_dim}")
    print(f"theta: {theta}")

    # Create RoPE module
    rope = RoPE(head_dim=head_dim, theta=theta)

    # Test with dummy data
    batch_size = 2
    seq_len = 10
    test_x = torch.randn(batch_size, num_heads, seq_len, head_dim)
    position_ids = torch.arange(seq_len)

    print(f"\nInput shape: {test_x.shape}")
    print(f"Position IDs: {position_ids.tolist()}")

    # Apply RoPE
    output = rope(test_x, position_ids)

    print(f"Output shape: {output.shape}")
    print(f"Shapes match: {test_x.shape == output.shape}")
    print(f"\nFirst position, first head (first 10 values):")
    print(f"  Before RoPE: {test_x[0, 0, 0, :10]}")
    print(f"  After RoPE:  {output[0, 0, 0, :10]}")
