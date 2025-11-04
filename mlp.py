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


if __name__ == "__main__":
    # Test MLP module
    from model import load_config

    config = load_config()

    print("Testing MLP Module:")
    print("=" * 50)

    # Parameters
    d_model: int = config["hidden_size"]  # 2560
    intermediate_size: int = config["intermediate_size"]  # 9728

    print(f"d_model: {d_model}")
    print(f"intermediate_size: {intermediate_size}")
    print(f"expansion ratio: {intermediate_size / d_model:.2f}x")

    # Create MLP module
    mlp = MLP(d_model=d_model, intermediate_size=intermediate_size)

    # Test with dummy data
    print(f"\n" + "=" * 50)
    print("Test: Forward pass")
    print("=" * 50)

    batch_size = 2
    seq_len = 5
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"Input shape: {x.shape}")
    output = mlp(x)
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {d_model})")
    print(f"Match: {output.shape == (batch_size, seq_len, d_model)}")

    # Test that shapes are correct throughout
    print(f"\n" + "=" * 50)
    print("Test: Intermediate shapes")
    print("=" * 50)

    gate = torch.einsum("bsd,id->bsi", x, mlp.w_gate)
    print(f"gate shape: {gate.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {intermediate_size})")
    print(f"Match: {gate.shape == (batch_size, seq_len, intermediate_size)}")

    up = torch.einsum("bsd,id->bsi", x, mlp.w_up)
    print(f"\nup shape: {up.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {intermediate_size})")
    print(f"Match: {up.shape == (batch_size, seq_len, intermediate_size)}")

    hidden = mlp.activation(gate) * up
    print(f"\nhidden shape: {hidden.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {intermediate_size})")
    print(f"Match: {hidden.shape == (batch_size, seq_len, intermediate_size)}")

    output_check = torch.einsum("bsi,di->bsd", hidden, mlp.w_down)
    print(f"\noutput shape: {output_check.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {d_model})")
    print(f"Match: {output_check.shape == (batch_size, seq_len, d_model)}")

    # Verify SiLU activation works
    print(f"\n" + "=" * 50)
    print("Test: SiLU activation")
    print("=" * 50)

    test_input = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    test_output = mlp.activation(test_input)
    print(f"Input: {test_input.tolist()}")
    print(f"SiLU output: {test_output.tolist()}")
    print(f"SiLU(x) = x * sigmoid(x)")
    print(f"SiLU(0) should be ~0: {test_output[2]:.6f}")
    print(f"SiLU(1) should be ~0.731: {test_output[3]:.6f}")
