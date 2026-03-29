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


def log_run_summary(summary: dict, local_dir: str = ".") -> str:
    """Serialize a run-summary dict to JSON, save it locally, and upload to MLflow.

    The file is always written to ``<local_dir>/run_summary.json`` so that
    callers can locate it predictably on the local filesystem regardless of
    whether an MLflow run is active.  When an active run exists the file is
    also uploaded under the ``summary/`` artifact sub-directory.

    This function **never raises** — any error is swallowed and logged as a
    warning so that a tracking failure can never abort a training run.

    Parameters
    ----------
    summary:
        Dict containing the structured run record.  All values must be JSON-
        serialisable (use ``default=str`` internally handles datetimes, tensors
        represented as scalars, etc.).
    local_dir:
        Directory where ``run_summary.json`` is written.  Defaults to ``"."``
        which resolves to Hydra's per-run output directory during normal runs
        and to ``tmp_path`` when tests use ``monkeypatch.chdir``.

    Returns
    -------
    str
        Absolute path of the written JSON file, or an empty string on failure.
    """
    try:
        import json
        from pathlib import Path

        out_dir = Path(local_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "run_summary.json"
        out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        log.debug(f"Run summary written to {out_path}")

        # Upload to MLflow when a run is active (reuse existing helper)
        log_artifact_to_run(str(out_path), tag="summary")
        return str(out_path)
    except Exception as e:
        log.warning(f"log_run_summary failed: {e}")
        return ""


def log_exception_to_run(exc: BaseException) -> None:
    """Record a caught exception as MLflow tags and a traceback text artifact.

    Intended to be called from an ``except`` block **before** re-raising.
    Sets two MLflow tags (``run_status`` and ``exception_type``) and uploads
    the full formatted traceback as ``summary/exception_traceback.txt``.

    Safe to call even when no MLflow run is active — the call becomes a no-op.
    This function **never raises**.

    Parameters
    ----------
    exc:
        The caught exception instance.  The traceback is captured via
        ``traceback.format_exc()`` which reads the current exception context,
        so this must be called from inside an active ``except`` block.
    """
    try:
        import traceback as _tb
        import tempfile
        from pathlib import Path

        import mlflow

        if not mlflow.active_run():
            return

        tb_text = _tb.format_exc()

        # Set lightweight tags visible directly in the MLflow UI
        mlflow.set_tag("run_status", "failed")
        mlflow.set_tag("exception_type", type(exc).__name__)

        # Upload the full traceback as a text artifact for detailed post-mortem
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            prefix="exception_traceback_",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(tb_text)
            tmp_path = f.name

        mlflow.log_artifact(tmp_path, artifact_path="summary")
        Path(tmp_path).unlink(missing_ok=True)
        log.debug(f"Exception traceback logged to MLflow for {type(exc).__name__}.")
    except Exception as inner:
        log.warning(f"log_exception_to_run failed silently: {inner}")
