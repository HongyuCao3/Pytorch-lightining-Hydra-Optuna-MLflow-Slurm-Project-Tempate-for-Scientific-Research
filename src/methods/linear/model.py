"""Pure nn.Module components for the Linear baseline.

A single linear projection: no hidden layers, no nonlinearity.
Useful as a lower-bound baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearModel(nn.Module):
    """Single linear layer classifier.

    Parameters
    ----------
    input_dim  : number of input features
    output_dim : number of output classes (logits)
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
