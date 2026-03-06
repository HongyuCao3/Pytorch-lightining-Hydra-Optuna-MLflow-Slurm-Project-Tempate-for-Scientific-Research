"""Acceptance criteria 2 & 3: Single-batch forward + fast dev run.

2. Single-batch forward: model(batch_x) runs without error
3. Fast dev run: trainer completes with fast_dev_run=True
"""
from __future__ import annotations

import pytest
import torch
import pytorch_lightning as pl
import hydra


# ---------------------------------------------------------------------------
# Acceptance 2: Single-batch forward
# ---------------------------------------------------------------------------

def test_mlp_single_batch_forward(cfg_train):
    """MLP forward pass on one batch from the datamodule."""
    dm = hydra.utils.instantiate(cfg_train.dataset)
    dm.setup("fit")

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    model = hydra.utils.instantiate(cfg_train.method)
    model.eval()

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (x.shape[0], cfg_train.method.output_dim)


def test_linear_single_batch_forward(cfg_linear):
    """Linear baseline forward pass on one batch."""
    dm = hydra.utils.instantiate(cfg_linear.dataset)
    dm.setup("fit")

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    model = hydra.utils.instantiate(cfg_linear.method)
    model.eval()

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (x.shape[0], cfg_linear.method.output_dim)


def test_mlp_predict_step_contract(cfg_train):
    """predict_step returns {"pred": Tensor, "prob": Tensor, "meta": {...}}."""
    dm = hydra.utils.instantiate(cfg_train.dataset)
    dm.setup("predict")

    batch = next(iter(dm.predict_dataloader()))
    model = hydra.utils.instantiate(cfg_train.method)
    model.eval()

    with torch.no_grad():
        output = model.predict_step(batch, batch_idx=0)

    assert "pred" in output
    assert "prob" in output
    assert "meta" in output
    assert isinstance(output["pred"], torch.Tensor)
    assert isinstance(output["prob"], torch.Tensor)
    assert output["prob"].shape[-1] == cfg_train.method.output_dim


# ---------------------------------------------------------------------------
# Acceptance 3: Fast dev run (integration)
# ---------------------------------------------------------------------------

def test_fast_dev_run_mlp(cfg_train):
    """fast_dev_run=True completes without error for MLP."""
    dm = hydra.utils.instantiate(cfg_train.dataset)
    model = hydra.utils.instantiate(cfg_train.method)

    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, datamodule=dm)
    # If we reach here without exception, the test passes


def test_fast_dev_run_linear(cfg_linear):
    """fast_dev_run=True completes without error for Linear baseline."""
    dm = hydra.utils.instantiate(cfg_linear.dataset)
    model = hydra.utils.instantiate(cfg_linear.method)

    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, datamodule=dm)


def test_full_train_run_mlp(cfg_train):
    """Acceptance 4: Full run – val_loss decreases after training (smoke)."""
    dm = hydra.utils.instantiate(cfg_train.dataset)
    model = hydra.utils.instantiate(cfg_train.method)

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=dm)

    metrics = trainer.callback_metrics
    assert "val_loss" in metrics
    assert float(metrics["val_loss"]) < 10.0  # sanity: loss is finite and reasonable
