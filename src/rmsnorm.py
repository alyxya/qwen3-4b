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
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)  # (..., 1)
        return self.weight * (x / rms)  # (..., dim)
