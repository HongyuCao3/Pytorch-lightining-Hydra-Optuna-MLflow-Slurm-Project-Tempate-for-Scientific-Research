"""Unified MLflow utilities.

Provides a thin wrapper so the rest of the codebase never imports mlflow
directly (makes it easy to swap or mock in tests).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from src.utils.logging import get_logger

log = get_logger(__name__)


def _get_active_run_id() -> Optional[str]:
    """Return current MLflow run ID, or None if not in a run."""
    try:
        import mlflow
        run = mlflow.active_run()
        return run.info.run_id if run else None
    except Exception:
        return None


def log_artifact_to_run(
    path: str,
    logger: Any = None,
    tag: str = "",
) -> None:
    """Log a local file as an MLflow artifact.

    Falls back silently if MLflow is not active or path doesn't exist.

    Parameters
    ----------
    path   : local file path to log
    logger : PL logger (unused directly, kept for API consistency)
    tag    : sub-directory within MLflow artifact store
    """
    if not Path(path).exists():
        log.debug(f"Artifact not found, skipping MLflow log: {path}")
        return
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.log_artifact(path, artifact_path=tag or None)
            log.debug(f"Logged artifact to MLflow: {path} (tag={tag!r})")
    except Exception as e:
        log.warning(f"MLflow log_artifact failed for '{path}': {e}")


def log_config_to_run(cfg: DictConfig, logger: Any = None) -> None:
    """Serialize and log the full merged Hydra config as an MLflow artifact."""
    try:
        import mlflow
        if not mlflow.active_run():
            return
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="hydra_config_", delete=False
        ) as f:
            f.write(OmegaConf.to_yaml(cfg))
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, artifact_path="config")
        Path(tmp_path).unlink(missing_ok=True)
        log.debug("Hydra config logged to MLflow.")
    except Exception as e:
        log.warning(f"Failed to log config to MLflow: {e}")


def log_metrics(metrics: dict, step: Optional[int] = None) -> None:
    """Log a dict of scalar metrics to MLflow."""
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.log_metrics(metrics, step=step)
    except Exception as e:
        log.warning(f"MLflow log_metrics failed: {e}")


def log_params(params: dict) -> None:
    """Log a dict of hyperparameters to MLflow."""
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.log_params(params)
    except Exception as e:
        log.warning(f"MLflow log_params failed: {e}")
