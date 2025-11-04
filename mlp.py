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

        # Weight matrices (no bias)
        # gate_proj: projects from d_model to intermediate_size
        self.w_gate = nn.Parameter(torch.randn(intermediate_size, d_model))  # (9728, 2560)

        # up_proj: projects from d_model to intermediate_size
        self.w_up = nn.Parameter(torch.randn(intermediate_size, d_model))  # (9728, 2560)

        # down_proj: projects from intermediate_size back to d_model
        self.w_down = nn.Parameter(torch.randn(d_model, intermediate_size))  # (2560, 9728)

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
        # x: (batch, seq_len, d_model) - "bsd"
        # w_gate: (intermediate_size, d_model) - "id"
        # gate: (batch, seq_len, intermediate_size) - "bsi"
        gate = torch.einsum("bsd,id->bsi", x, self.w_gate)  # (batch, seq_len, 9728)
        gate = self.activation(gate)

        # Up pathway: x → up_proj (no activation)
        # x: (batch, seq_len, d_model) - "bsd"
        # w_up: (intermediate_size, d_model) - "id"
        # up: (batch, seq_len, intermediate_size) - "bsi"
        up = torch.einsum("bsd,id->bsi", x, self.w_up)  # (batch, seq_len, 9728)

        # Combine: element-wise multiply gate and up
        hidden = gate * up  # (batch, seq_len, 9728)

        # Down pathway: hidden → down_proj
        # hidden: (batch, seq_len, intermediate_size) - "bsi"
        # w_down: (d_model, intermediate_size) - "di"
        # output: (batch, seq_len, d_model) - "bsd"
        output = torch.einsum("bsi,di->bsd", hidden, self.w_down)  # (batch, seq_len, 2560)

        return output

