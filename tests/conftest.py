"""Shared pytest fixtures for the test suite."""
from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
import os


@pytest.fixture(scope="session")
def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def cfg_train(project_root) -> DictConfig:
    """Return a default train config (wine + mlp)."""
    config_dir = os.path.join(project_root, "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["mode=debug", "dataset=wine", "method=mlp", "logger=mlflow"],
        )
    return cfg


@pytest.fixture(scope="session")
def cfg_linear(project_root) -> DictConfig:
    """Return a debug config for the linear baseline."""
    config_dir = os.path.join(project_root, "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["mode=debug", "dataset=wine", "method=linear", "logger=mlflow"],
        )
    return cfg
