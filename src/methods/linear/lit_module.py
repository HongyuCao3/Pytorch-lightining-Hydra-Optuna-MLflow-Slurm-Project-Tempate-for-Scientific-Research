"""LightningModule for the Linear baseline method."""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from src.methods.linear.model import LinearModel
from src.methods.registry import register


@register("linear")
class LinearLitModule(pl.LightningModule):
    """LightningModule wrapping LinearModel for classification.

    Parameters
    ----------
    input_dim     : number of input features
    output_dim    : number of classes
    learning_rate : SGD / Adam learning rate
    weight_decay  : L2 penalty
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = LinearModel(input_dim, output_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()

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
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return {"optimizer": optimizer}
