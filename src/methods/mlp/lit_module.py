"""LightningModule for the MLP method.

Implements training / validation / test / predict steps.
predict_step output contract: {"pred": Tensor, "prob": Tensor, "meta": {...}}
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

from src.methods._eval_snapshot_mixin import EvalSnapshotMixin
from src.methods.mlp.model import MLP
from src.methods.registry import register


@register("mlp")
class MLPLitModule(EvalSnapshotMixin, pl.LightningModule):
    """LightningModule wrapping the MLP model for classification.

    Parameters
    ----------
    input_dim   : number of input features
    hidden_dims : list of hidden layer widths
    output_dim  : number of classes
    learning_rate : Adam learning rate
    weight_decay  : Adam L2 penalty
    dropout       : dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Convert OmegaConf ListConfig -> plain list before save_hyperparameters().
        # Ensures the checkpoint contains only native Python types, which is
        # required for torch.load(weights_only=True) in PyTorch 2.6+.
        hidden_dims = list(hidden_dims)
        self.save_hyperparameters()

        self.model = MLP(input_dim, hidden_dims, output_dim, dropout)

        # torchmetrics (reset per epoch automatically in PL)
        self._acc = torchmetrics.Accuracy(task="multiclass", num_classes=output_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    # PL hooks
    # ------------------------------------------------------------------

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, Any]:
        """Standard predict output: {"pred": Tensor, "prob": Tensor, "meta": {...}}."""
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=-1)
        probs = torch.softmax(logits, dim=-1)
        return {
            "pred": preds,
            "prob": probs,
            "meta": {
                "true_label": y,
                "batch_idx": batch_idx,
            },
        }

    # ------------------------------------------------------------------
    # Optimizer / Scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
