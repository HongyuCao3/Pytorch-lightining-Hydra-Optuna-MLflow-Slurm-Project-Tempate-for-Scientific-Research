"""Core training / optuna / inference orchestration.

Public API
----------
run_train(cfg)  -> Dict[str, Any]   train + test, return metrics
run_optuna(cfg) -> None             hyperparameter search with Optuna
run_infer(cfg)  -> None             load checkpoint, run predict + analyzers
"""
from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

import hydra

from src.callbacks.oom_handler import OOMHandler
from src.utils.logging import get_logger
from src.utils.mlflow_utils import log_artifact_to_run, log_config_to_run

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_name_from_target(target: str, kind: str = "method") -> str:
    """Derive a short human-readable name from a _target_ string."""
    parts = target.split(".")
    if kind == "method":
        try:
            idx = parts.index("methods")
            return parts[idx + 1]
        except (ValueError, IndexError):
            pass
    elif kind == "dataset":
        return parts[-1].replace("DataModule", "").lower()
    return parts[-1].lower()


def _build_checkpoint_callback(cfg: DictConfig, method_name: str, dataset_name: str) -> ModelCheckpoint:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirpath = Path("checkpoints") / method_name / timestamp

    # Filename template: {method}-{dataset}-{timestamp}-epoch={epoch:02d}-val={val_loss:.4f}
    filename = f"{method_name}-{dataset_name}-{timestamp}-epoch={{epoch:02d}}-val={{val_loss:.4f}}"

    return ModelCheckpoint(
        dirpath=str(dirpath),
        filename=filename,
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=3,
        verbose=True,
    )


def _save_latest_record(ckpt_path: str, output_dir: str = ".") -> None:
    """Write latest checkpoint path to a JSON record (symlink alternative for Windows)."""
    record = {"latest_checkpoint": str(ckpt_path), "updated_at": datetime.now().isoformat()}
    record_path = Path(output_dir) / "latest_checkpoint.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2))
    log.info(f"Latest checkpoint record saved to {record_path}")


def _suggest_params_from_search_space(trial, cfg: DictConfig) -> DictConfig:
    """Apply Optuna trial suggestions based on cfg.optuna.search_space."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_trial = OmegaConf.create(cfg_dict)

    search_space = OmegaConf.select(cfg, "optuna.search_space") or {}
    for param_path, spec in search_space.items():
        kind = spec.get("type", "float")
        if kind == "float":
            value = trial.suggest_float(
                param_path,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
            )
        elif kind == "int":
            value = trial.suggest_int(param_path, spec["low"], spec["high"])
        elif kind == "categorical":
            value = trial.suggest_categorical(param_path, spec["choices"])
        else:
            continue
        OmegaConf.update(cfg_trial, param_path, value)

    return cfg_trial


# ---------------------------------------------------------------------------
# run_train
# ---------------------------------------------------------------------------

def run_train(cfg: DictConfig) -> Dict[str, Any]:
    """Train and optionally test a model.

    Contract
    --------
    - Seeds everything with cfg.seed
    - Instantiates datamodule, model (LightningModule), logger, trainer
    - Calls trainer.fit(); calls trainer.test() if cfg.mode.run_test
    - Returns callback_metrics dict with at least val_loss / val_acc
    """
    pl.seed_everything(cfg.seed, workers=True)

    # --- Instantiate components ------------------------------------------
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.dataset)
    model: pl.LightningModule = hydra.utils.instantiate(cfg.method)
    logger = hydra.utils.instantiate(cfg.logger)

    method_name = _get_name_from_target(cfg.method._target_, "method")
    dataset_name = _get_name_from_target(cfg.dataset._target_, "dataset")

    # --- Callbacks ----------------------------------------------------------
    ckpt_callback = _build_checkpoint_callback(cfg, method_name, dataset_name)
    callbacks = [
        ckpt_callback,
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        OOMHandler(),
    ]

    # --- Trainer ------------------------------------------------------------
    fast_dev_run = bool(OmegaConf.select(cfg, "mode.fast_dev_run", default=False))
    trainer_kwargs: Dict[str, Any] = dict(logger=logger, callbacks=callbacks)
    if fast_dev_run:
        trainer_kwargs["fast_dev_run"] = True

    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, **trainer_kwargs)

    # --- Log merged config as artifact --------------------------------------
    log_config_to_run(cfg, logger)

    # --- Fit ----------------------------------------------------------------
    log.info(f"Starting training: method={method_name}, dataset={dataset_name}")
    trainer.fit(model, datamodule=datamodule)

    # --- Test ---------------------------------------------------------------
    run_test = bool(OmegaConf.select(cfg, "mode.run_test", default=True))
    if run_test and not fast_dev_run:
        trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # --- Save latest checkpoint record -------------------------------------
    if ckpt_callback.best_model_path:
        _save_latest_record(ckpt_callback.best_model_path)
        log_artifact_to_run(ckpt_callback.best_model_path, logger, tag="best_checkpoint")

    # --- Collect metrics ---------------------------------------------------
    metrics: Dict[str, Any] = {
        k: float(v) for k, v in trainer.callback_metrics.items()
    }
    log.info(f"Training complete. Metrics: {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# run_optuna
# ---------------------------------------------------------------------------

def run_optuna(cfg: DictConfig) -> None:
    """Hyperparameter search via Optuna.

    Each trial:
      1. Suggests params from cfg.optuna.search_space
      2. Calls run_train with a deep-copied + patched cfg
      3. Returns cfg.optuna.metric value

    MLflow integration: each trial creates its own MLflow run (via the PL
    MLFlowLogger inside run_train).  The parent study is tracked separately.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.INFO)

    storage = OmegaConf.select(cfg, "optuna.storage")
    if storage in (None, "null", ""):
        storage = None

    study = optuna.create_study(
        direction=cfg.optuna.direction,
        study_name=cfg.optuna.study_name,
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        cfg_trial = _suggest_params_from_search_space(trial, cfg)
        # Ensure mode is train so that run_train runs fully
        OmegaConf.update(cfg_trial, "mode.name", "train")
        OmegaConf.update(cfg_trial, "mode.run_test", False)

        try:
            metrics = run_train(cfg_trial)
        except Exception as e:
            log.warning(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()

        metric_name = cfg.optuna.metric
        return float(metrics.get(metric_name, float("inf")))

    timeout = OmegaConf.select(cfg, "optuna.timeout")
    if timeout in (None, "null"):
        timeout = None

    study.optimize(
        objective,
        n_trials=cfg.optuna.n_trials,
        timeout=timeout,
        gc_after_trial=True,
    )

    log.info(f"Optuna finished. Best value: {study.best_value}")
    log.info(f"Best params: {study.best_params}")

    # Save study summary
    summary_path = Path("optuna_summary.json")
    summary = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info(f"Optuna summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# run_infer
# ---------------------------------------------------------------------------

def run_infer(cfg: DictConfig) -> None:
    """Load a checkpoint and run the inference pipeline + analyzers."""
    from src.inference.pipeline import run_inference_pipeline

    checkpoint_path = OmegaConf.select(cfg, "inference.checkpoint_path")
    if not checkpoint_path:
        raise ValueError("inference.checkpoint_path must be set for infer mode. "
                         "E.g.: python -m src.main mode=infer inference.checkpoint_path=<path>")

    run_inference_pipeline(cfg)
