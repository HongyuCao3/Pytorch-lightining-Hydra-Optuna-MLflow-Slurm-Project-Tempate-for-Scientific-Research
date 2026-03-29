"""Core training / optuna / inference orchestration.

Public API
----------
run_train(cfg)  -> Dict[str, Any]   train + test, return metrics
run_optuna(cfg) -> None             hyperparameter search with Optuna
run_infer(cfg)  -> None             load checkpoint, run predict + analyzers

Tracking contract
-----------------
Every ``run_train`` call writes ``run_summary.json`` to the current working
directory (Hydra's per-run output dir) and uploads it to MLflow under the
``summary/`` artifact path.  The summary always contains the 9 canonical
observability fields regardless of whether the run succeeded or failed:

    trial_id, seed, params, final_metric, best_metric,
    convergence_step, status, wall_time, traceback
"""
from __future__ import annotations

import csv
import json
import time
import traceback as tb_module
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

import hydra

from src.callbacks.oom_handler import OOMHandler
from src.callbacks.run_tracker import RunTrackerCallback
from src.utils.logging import get_logger
from src.utils.mlflow_utils import (
    log_artifact_to_run,
    log_config_to_run,
    log_exception_to_run,
    log_run_summary,
)

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
    - Seeds everything with ``cfg.seed``
    - Instantiates datamodule, model (LightningModule), logger, trainer
    - Calls ``trainer.fit()``; calls ``trainer.test()`` if ``cfg.mode.run_test``
    - **Always** writes ``run_summary.json`` (9 canonical fields) regardless of
      success or failure, then re-raises any exception
    - Returns callback_metrics dict with at least ``val_loss`` / ``val_acc``

    Summary fields
    --------------
    trial_id         : Optuna trial number if called from run_optuna, else None
    seed             : integer seed from cfg.seed
    params           : flat dict of model hyper-parameters from model.hparams
    final_metric     : val_loss at the end of training (None on fast_dev_run)
    best_metric      : best val_loss across all epochs (from RunTrackerCallback)
    convergence_step : first epoch where the metric crossed the threshold (None)
    status           : "completed" or "failed"
    wall_time        : total elapsed seconds (float)
    traceback        : formatted traceback string on failure, None on success
    """
    # ------------------------------------------------------------------
    # Wall-clock start — captured before any work so it includes setup time
    # ------------------------------------------------------------------
    _wall_start = time.monotonic()

    pl.seed_everything(cfg.seed, workers=True)

    # --- Instantiate components ------------------------------------------
    # _convert_="all" tells Hydra to convert every OmegaConf container
    # (ListConfig, DictConfig, ContainerMetadata …) to native Python types
    # (list, dict) before calling __init__.  This prevents save_hyperparameters()
    # from storing OmegaConf objects in the checkpoint, which would cause
    # torch.load(weights_only=True) to fail on PyTorch 2.6+.
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.dataset, _convert_="all")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.method, _convert_="all")
    logger = hydra.utils.instantiate(cfg.logger)

    # Snapshot hyper-parameters immediately after instantiation so they are
    # available for the run summary even if fit() later raises.
    _tracked_params: Dict[str, Any] = dict(model.hparams)

    method_name = _get_name_from_target(cfg.method._target_, "method")
    dataset_name = _get_name_from_target(cfg.dataset._target_, "dataset")

    # --- Callbacks ----------------------------------------------------------
    ckpt_callback = _build_checkpoint_callback(cfg, method_name, dataset_name)
    # RunTrackerCallback is a pure data accumulator — no I/O.
    # run_train reads its public attributes after fit() to build the summary.
    tracker = RunTrackerCallback(monitor="val_loss", mode="min")
    callbacks = [
        ckpt_callback,
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        OOMHandler(),
        tracker,
    ]

    # --- Trainer ------------------------------------------------------------
    fast_dev_run = bool(OmegaConf.select(cfg, "mode.fast_dev_run", default=False))
    trainer_kwargs: Dict[str, Any] = dict(logger=logger, callbacks=callbacks)
    if fast_dev_run:
        trainer_kwargs["fast_dev_run"] = True

    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, **trainer_kwargs)

    # --- Log merged config as artifact --------------------------------------
    log_config_to_run(cfg, logger)

    # --- Fit + Test (with failure capture) ----------------------------------
    # The exception object and formatted traceback are stored so the summary
    # can be written unconditionally before re-raising.
    _exc: Optional[Exception] = None
    _traceback: Optional[str] = None
    _status = "completed"

    log.info(f"Starting training: method={method_name}, dataset={dataset_name}")
    try:
        trainer.fit(model, datamodule=datamodule)

        run_test = bool(OmegaConf.select(cfg, "mode.run_test", default=True))
        if run_test and not fast_dev_run:
            trainer.test(model, datamodule=datamodule, ckpt_path="best")

    except Exception as exc:
        _status = "failed"
        _traceback = tb_module.format_exc()
        _exc = exc
        # Upload traceback + set MLflow tags before the summary is written
        log_exception_to_run(exc)

    # --- Save latest checkpoint record (only on success) -------------------
    if _exc is None and ckpt_callback.best_model_path:
        _save_latest_record(ckpt_callback.best_model_path)
        log_artifact_to_run(ckpt_callback.best_model_path, logger, tag="best_checkpoint")

    # --- Build and log the structured run summary --------------------------
    # This block always runs — both on success and failure.
    metrics: Dict[str, Any] = {
        k: float(v) for k, v in trainer.callback_metrics.items()
    }
    _summary: Dict[str, Any] = {
        # trial_id is injected by run_optuna via cfg.optuna._trial_id;
        # returns None for standalone train runs.
        "trial_id": OmegaConf.select(cfg, "optuna._trial_id", default=None),
        "seed": int(cfg.seed),
        "params": _tracked_params,
        "final_metric": metrics.get("val_loss"),
        "best_metric": tracker.best_value,
        "convergence_step": tracker.convergence_step,
        "status": _status,
        "wall_time": round(time.monotonic() - _wall_start, 3),
        "traceback": _traceback,
    }
    log_run_summary(_summary)
    log.info(f"Training complete | status={_status} | metrics={metrics}")

    # Re-raise any captured exception after the summary has been persisted
    if _exc is not None:
        raise _exc

    return metrics


# ---------------------------------------------------------------------------
# run_optuna
# ---------------------------------------------------------------------------

def run_optuna(cfg: DictConfig) -> None:
    """Hyperparameter search via Optuna.

    Each trial:

    1. Suggests params from ``cfg.optuna.search_space``
    2. Injects the trial number as ``cfg.optuna._trial_id`` into a config copy
    3. Calls ``run_train`` with the patched copy (each trial → separate MLflow run)
    4. Returns the scalar metric specified by ``cfg.optuna.metric``

    After all trials complete, two summary files are written and uploaded to
    the active MLflow run (if any):

    * ``optuna_summary.json`` — full per-trial records (trial_id, state, value,
      params, duration_seconds) plus study-level best_value / best_params
    * ``optuna_summary.csv``  — same data in tabular form for spreadsheet use
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
        """Single Optuna trial: suggest → configure → train → return metric."""
        cfg_trial = _suggest_params_from_search_space(trial, cfg)

        # Switch mode to "train" so run_train executes the full training loop
        OmegaConf.update(cfg_trial, "mode.name", "train")
        OmegaConf.update(cfg_trial, "mode.run_test", False)

        # Propagate fast_dev_run from the parent config so tests remain fast
        fast_dev_run = OmegaConf.select(cfg, "mode.fast_dev_run", default=False)
        OmegaConf.update(cfg_trial, "mode.fast_dev_run", fast_dev_run)

        # Inject trial number so run_train can embed it in run_summary.json
        OmegaConf.update(cfg_trial, "optuna._trial_id", trial.number)

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

    # ------------------------------------------------------------------
    # Build per-trial records (one dict per trial)
    # ------------------------------------------------------------------
    trial_records = []
    for t in study.trials:
        # Duration is None when the trial never completed (PRUNED / FAIL)
        duration: Optional[float] = None
        if t.datetime_start and t.datetime_complete:
            duration = round(
                (t.datetime_complete - t.datetime_start).total_seconds(), 3
            )
        trial_records.append(
            {
                "trial_id": t.number,
                "state": t.state.name,   # "COMPLETE" | "PRUNED" | "FAIL"
                "value": t.value,
                "params": t.params,
                "duration_seconds": duration,
            }
        )

    # ------------------------------------------------------------------
    # Write optuna_summary.json
    # ------------------------------------------------------------------
    summary_path = Path("optuna_summary.json")
    summary = {
        "study_name": cfg.optuna.study_name,
        "direction": cfg.optuna.direction,
        "metric": cfg.optuna.metric,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "trials": trial_records,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(f"Optuna JSON summary saved to {summary_path}")

    # ------------------------------------------------------------------
    # Write optuna_summary.csv (flat table, one row per trial)
    # ------------------------------------------------------------------
    csv_path = Path("optuna_summary.csv")
    if trial_records:
        # Collect all param keys across all trials to form a stable column set
        all_param_keys = sorted({k for r in trial_records for k in r["params"]})
        fieldnames = (
            ["trial_id", "state", "value", "duration_seconds"] + all_param_keys
        )
        rows = []
        for r in trial_records:
            row: Dict[str, Any] = {
                "trial_id": r["trial_id"],
                "state": r["state"],
                "value": r["value"],
                "duration_seconds": r["duration_seconds"],
            }
            # Fill param columns; missing params (e.g. pruned trials) → empty
            for k in all_param_keys:
                row[k] = r["params"].get(k, "")
            rows.append(row)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        log.info(f"Optuna CSV summary saved to {csv_path}")

    # Upload both summaries to the active MLflow run when available
    log_artifact_to_run(str(summary_path), tag="summary")
    if trial_records:
        log_artifact_to_run(str(csv_path), tag="summary")


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
