"""Pure nn.Module components for the MLP method.

Rules (from claude.md)
----------------------
- No training logic, no Hydra, no loggers.
- Only torch.nn.Module subclasses.
- forward() contract: Tensor(N, D) -> Tensor(N, C)  (logits)
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron with LayerNorm + ReLU + Dropout.

    Uses LayerNorm (not BatchNorm) for stability with small batch sizes.

    Parameters
    ----------
    input_dim   : number of input features
    hidden_dims : list of hidden layer sizes
    output_dim  : number of output classes (logits)
    dropout     : dropout probability applied after each hidden activation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
