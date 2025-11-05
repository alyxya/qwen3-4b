"""
MLP (Feed-Forward Network) for Qwen3 4B

The MLP uses a gated activation:
- Gate pathway: gate_proj → SiLU activation
- Up pathway: up_proj (no activation)
- Combine: gate * up
- Down pathway: down_proj

This is also called SwiGLU (Swish-Gated Linear Unit)
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Feed-forward network with gated activation (SwiGLU)

    Architecture:
    x → gate_proj → SiLU → (multiply with) up_proj → down_proj → output
    """

    def __init__(self, d_model: int, intermediate_size: int) -> None:
        """
        Initialize MLP layer

        Args:
            d_model: Model dimension (2560 for Qwen3 4B)
            intermediate_size: Hidden dimension (9728 for Qwen3 4B)
        """
        super().__init__()
        self.d_model: int = d_model
        self.intermediate_size: int = intermediate_size

        # Projection layers to match HuggingFace naming
        # Using nn.Linear without bias to match pretrained weights
        # gate_proj: projects from d_model to intermediate_size
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)  # (2560 -> 9728)

        # up_proj: projects from d_model to intermediate_size
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)  # (2560 -> 9728)

        # down_proj: projects from intermediate_size back to d_model
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)  # (9728 -> 2560)

        # SiLU activation (also called Swish)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP

        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor, shape (batch_size, seq_len, d_model)
        """
        # Gate pathway: x → gate_proj → SiLU
        gate = self.gate_proj(x)  # (batch, seq_len, intermediate_size)
        gate = self.activation(gate)

        # Up pathway: x → up_proj (no activation)
        up = self.up_proj(x)  # (batch, seq_len, intermediate_size)

        # Combine: element-wise multiply gate and up
        hidden = gate * up  # (batch, seq_len, intermediate_size)

        # Down pathway: hidden → down_proj
        output = self.down_proj(hidden)  # (batch, seq_len, d_model)

        return output

