"""
RMSNorm implementation shared across modules.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm skips mean-centering and scales inputs by their RMS value.
    """

    def __init__(self, d_model: int, eps: float) -> None:
        """
        Initialize RMSNorm.

        Args:
            d_model: Number of features in the input.
            eps: Numerical stability constant.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Args:
            x: Tensor of shape (..., d_model)

        Returns:
            Tensor with the same shape as the input.
        """
        # Store input dtype and convert to float32 for numerical stability
        # This matches HuggingFace's implementation exactly
        input_dtype = x.dtype
        x = x.to(torch.float32)

        # Compute variance and normalize
        variance = x.pow(2).mean(-1, keepdim=True)  # (..., 1)
        x = x * torch.rsqrt(variance + self.eps)  # (..., d_model)

        # Convert back to original dtype and apply learned weight
        return self.weight * x.to(input_dtype)  # (..., d_model)
