"""Metric helpers shared across methods and analyzers."""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def accuracy(preds: Tensor, targets: Tensor) -> float:
    """Top-1 accuracy (fraction of correct predictions)."""
    return (preds == targets).float().mean().item()


def top_k_accuracy(logits: Tensor, targets: Tensor, k: int = 3) -> float:
    """Top-k accuracy."""
    _, top_k = logits.topk(k, dim=-1)
    correct = top_k.eq(targets.unsqueeze(-1).expand_as(top_k))
    return correct.any(dim=-1).float().mean().item()


def class_distribution(targets: Tensor, num_classes: int) -> Dict[int, int]:
    """Return per-class sample counts."""
    counts: Dict[int, int] = {}
    for cls in range(num_classes):
        counts[cls] = int((targets == cls).sum().item())
    return counts
