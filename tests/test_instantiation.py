"""Acceptance criterion 1: Instantiation checks.

Verifies that all major components can be instantiated from config
without errors.
"""
from __future__ import annotations

import pytest
import pytorch_lightning as pl
import hydra


def test_datamodule_instantiation(cfg_train):
    """instantiate(cfg.dataset) succeeds and returns a LightningDataModule."""
    dm = hydra.utils.instantiate(cfg_train.dataset)
    assert isinstance(dm, pl.LightningDataModule)


def test_mlp_model_instantiation(cfg_train):
    """instantiate(cfg.method) with mlp succeeds and returns a LightningModule."""
    model = hydra.utils.instantiate(cfg_train.method)
    assert isinstance(model, pl.LightningModule)


def test_linear_model_instantiation(cfg_linear):
    """instantiate(cfg.method) with linear baseline succeeds."""
    model = hydra.utils.instantiate(cfg_linear.method)
    assert isinstance(model, pl.LightningModule)


def test_logger_instantiation(cfg_train):
    """instantiate(cfg.logger) succeeds and returns a PL logger."""
    logger = hydra.utils.instantiate(cfg_train.logger)
    assert isinstance(logger, pl.loggers.Logger)


def test_trainer_instantiation(cfg_train):
    """instantiate(cfg.trainer) with fast_dev_run succeeds."""
    trainer = hydra.utils.instantiate(cfg_train.trainer, fast_dev_run=True, logger=False)
    assert isinstance(trainer, pl.Trainer)


def test_datamodule_setup(cfg_train):
    """DataModule.setup() creates train / val / test datasets."""
    dm = hydra.utils.instantiate(cfg_train.dataset)
    dm.setup("fit")
    assert dm.train_ds is not None
    assert dm.val_ds is not None
    assert dm.test_ds is not None
    assert len(dm.train_ds) > 0


def test_mlp_method_target(cfg_train):
    """cfg.method._target_ points to MLPLitModule."""
    assert "MLPLitModule" in cfg_train.method._target_


def test_linear_method_target(cfg_linear):
    """cfg.method._target_ points to LinearLitModule."""
    assert "LinearLitModule" in cfg_linear.method._target_
