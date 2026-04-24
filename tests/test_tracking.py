"""Regression tests for the experiment observability / tracking layer.

These tests verify that the 9-field ``run_summary.json`` contract is upheld
for every ``run_train`` and ``run_optuna`` call, and that the
``RunTrackerCallback`` records convergence and best-metric correctly.

Isolation strategy
------------------
Each test uses a **function-scoped** Hydra config fixture that:

* Clears Hydra global state before and after via ``GlobalHydra.instance().clear()``
  so it does not conflict with the session-scoped fixtures in ``conftest.py``.
* Points the MLflow tracking URI to a per-test temp directory so runs never
  pollute a shared ``mlruns/`` directory.

``monkeypatch.chdir(tmp_path)`` ensures that ``run_summary.json`` (written to
``"."`` by default) lands in the isolated temp directory.
"""
from __future__ import annotations

import json
import os
from typing import Generator
from unittest.mock import MagicMock

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Helper: build an isolated Hydra config (function scope)
# ---------------------------------------------------------------------------

def _make_cfg(project_root: str, tmp_path, overrides: list[str]) -> DictConfig:
    """Compose a Hydra config with an isolated MLflow tracking URI.

    ``logger.tracking_uri`` is set to a ``file:///`` URI pointing at
    ``tmp_path/mlruns`` so that:
    * MLflow artefacts never leak between tests or into the project ``mlruns/``.
    * The URI scheme is accepted by MLflowLogger on all platforms (including
      Windows, where bare absolute paths are rejected with an
      ``UnsupportedModelRegistryStoreURIException``).
    """
    from pathlib import Path

    # Path.as_uri() produces "file:///C:/..." on Windows and "file:///..." on Unix
    mlruns_uri = (tmp_path / "mlruns").as_uri()

    config_dir = os.path.join(project_root, "configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=overrides + [f"logger.tracking_uri={mlruns_uri}"],
        )
    GlobalHydra.instance().clear()
    return cfg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cfg_fast(project_root: str, tmp_path) -> DictConfig:
    """Fast-dev-run config for MLP on wine dataset."""
    return _make_cfg(
        project_root,
        tmp_path,
        overrides=["mode=debug", "dataset=wine", "method=mlp", "logger=mlflow"],
    )


@pytest.fixture()
def cfg_optuna_fast(project_root: str, tmp_path) -> DictConfig:
    """Optuna config limited to 2 fast trials for unit-test speed."""
    return _make_cfg(
        project_root,
        tmp_path,
        overrides=[
            "mode=optuna",
            "dataset=wine",
            "method=mlp",
            "logger=mlflow",
            "optuna.n_trials=2",
            "mode.fast_dev_run=true",
        ],
    )


# ---------------------------------------------------------------------------
# Test 1: run_summary.json is written after run_train
# ---------------------------------------------------------------------------

def test_run_summary_artifact_exists(cfg_fast: DictConfig, tmp_path, monkeypatch) -> None:
    """run_train must write run_summary.json to the current working directory."""
    from src.train import run_train

    monkeypatch.chdir(tmp_path)
    run_train(cfg_fast)

    summary_path = tmp_path / "run_summary.json"
    assert summary_path.exists(), (
        "run_summary.json was not written after run_train — "
        "check that log_run_summary() is called unconditionally."
    )


# ---------------------------------------------------------------------------
# Test 2: all 9 canonical fields are present and have correct types
# ---------------------------------------------------------------------------

def test_run_summary_fields(cfg_fast: DictConfig, tmp_path, monkeypatch) -> None:
    """run_summary.json must contain all 9 required observability fields."""
    from src.train import run_train

    monkeypatch.chdir(tmp_path)
    run_train(cfg_fast)

    data = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))

    required_fields = {
        "trial_id", "seed", "params", "final_metric",
        "best_metric", "convergence_step", "status",
        "wall_time", "traceback",
    }
    missing = required_fields - set(data.keys())
    assert not missing, f"run_summary.json is missing fields: {missing}"

    # Type assertions for fields that must never be absent
    assert data["status"] == "completed", (
        f"Expected status='completed', got {data['status']!r}"
    )
    assert isinstance(data["seed"], int), (
        f"seed must be int, got {type(data['seed'])}"
    )
    assert isinstance(data["wall_time"], (int, float)), (
        f"wall_time must be numeric, got {type(data['wall_time'])}"
    )
    assert data["wall_time"] > 0, "wall_time must be positive"
    assert isinstance(data["params"], dict), (
        f"params must be a dict, got {type(data['params'])}"
    )
    # traceback must be null on success
    assert data["traceback"] is None, (
        f"traceback must be null on success, got {data['traceback']!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: failure path sets status="failed" and records traceback
# ---------------------------------------------------------------------------

def test_run_summary_status_failed(
    cfg_fast: DictConfig, tmp_path, monkeypatch
) -> None:
    """When trainer.fit raises, run_summary.json must record status='failed'
    and a non-null traceback — even though the exception is re-raised.
    """
    import pytorch_lightning as pl
    from src.train import run_train

    # Patch trainer.fit to simulate a mid-training failure
    def _bad_fit(self, *args, **kwargs):  # noqa: ANN001
        raise RuntimeError("synthetic training failure for test")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pl.Trainer, "fit", _bad_fit)

    with pytest.raises(RuntimeError, match="synthetic training failure"):
        run_train(cfg_fast)

    summary_path = tmp_path / "run_summary.json"
    assert summary_path.exists(), (
        "run_summary.json must be written even when training fails."
    )

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["status"] == "failed", (
        f"Expected status='failed', got {data['status']!r}"
    )
    assert data["traceback"] is not None, (
        "traceback must be a non-null string when status='failed'."
    )
    assert "synthetic training failure" in data["traceback"], (
        "The traceback string must contain the original exception message."
    )


# ---------------------------------------------------------------------------
# Test 4: run_optuna writes optuna_summary.json with per-trial records + CSV
# ---------------------------------------------------------------------------

def test_optuna_summary_fields(
    cfg_optuna_fast: DictConfig, tmp_path, monkeypatch
) -> None:
    """run_optuna must produce optuna_summary.json with a 'trials' list and
    optuna_summary.csv with one row per trial.
    """
    from src.train import run_optuna

    monkeypatch.chdir(tmp_path)
    run_optuna(cfg_optuna_fast)

    # --- JSON ---
    json_path = tmp_path / "optuna_summary.json"
    assert json_path.exists(), "optuna_summary.json was not written."

    data = json.loads(json_path.read_text(encoding="utf-8"))
    for field in ("best_value", "best_params", "n_trials", "trials"):
        assert field in data, f"optuna_summary.json is missing field '{field}'"

    assert isinstance(data["trials"], list), "'trials' must be a list"
    assert len(data["trials"]) == 2, (
        f"Expected 2 trial records (n_trials=2), got {len(data['trials'])}"
    )

    # Each trial record must have the 4 required keys
    for trial_rec in data["trials"]:
        for key in ("trial_id", "state", "value", "params"):
            assert key in trial_rec, (
                f"Trial record is missing key '{key}': {trial_rec}"
            )

    # --- CSV ---
    csv_path = tmp_path / "optuna_summary.csv"
    assert csv_path.exists(), "optuna_summary.csv was not written."

    import csv as _csv
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        csv_rows = list(reader)

    assert len(csv_rows) == 2, (
        f"CSV should have 2 data rows, got {len(csv_rows)}"
    )
    for row in csv_rows:
        assert "trial_id" in row
        assert "state" in row
        assert "value" in row


# ---------------------------------------------------------------------------
# Test: loss-as-evaluation guard refuses non-convergence experiments
# ---------------------------------------------------------------------------

def test_loss_guard_refuses_evaluation_kind(
    project_root: str, tmp_path, monkeypatch
) -> None:
    """run_train must refuse to start when experiment.kind != 'convergence'
    and experiment.monitor looks like a loss. This is the enforcement
    mechanism for the rule in .claude/global.md → 'Evaluation metrics'.
    """
    from src.train import run_train

    cfg = _make_cfg(
        project_root,
        tmp_path,
        overrides=[
            "mode=debug",
            "dataset=wine",
            "method=mlp",
            "logger=mlflow",
            "experiment=evaluation",
            "experiment.monitor=val_loss",
        ],
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="looks like a loss"):
        run_train(cfg)


def test_loss_guard_allows_convergence_kind(
    project_root: str, tmp_path, monkeypatch
) -> None:
    """experiment=convergence is the one kind in which val_loss is a legal
    monitor — run_train must accept it and produce a normal summary.
    """
    from src.train import run_train

    cfg = _make_cfg(
        project_root,
        tmp_path,
        overrides=[
            "mode=debug",
            "dataset=wine",
            "method=mlp",
            "logger=mlflow",
            "experiment=convergence",
        ],
    )
    monkeypatch.chdir(tmp_path)
    run_train(cfg)

    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"


# ---------------------------------------------------------------------------
# Test 5: RunTrackerCallback unit test (no Trainer, no MLflow)
# ---------------------------------------------------------------------------

def test_run_tracker_callback() -> None:
    """RunTrackerCallback correctly tracks best_value, best_step,
    convergence_step, and status using only MagicMock trainers.
    """
    from src.callbacks.run_tracker import RunTrackerCallback

    # --- Setup: 3-epoch simulation with convergence_threshold=0.5 ---
    cb = RunTrackerCallback(
        monitor="val_loss", mode="min", convergence_threshold=0.5
    )

    epoch_losses = [0.8, 0.6, 0.4]
    for epoch, loss in enumerate(epoch_losses):
        trainer_mock = MagicMock()
        trainer_mock.callback_metrics = {"val_loss": loss}
        trainer_mock.current_epoch = epoch
        cb.on_validation_epoch_end(trainer_mock, MagicMock())

    # Best metric should be the minimum (0.4 at epoch 2)
    assert cb.best_value == pytest.approx(0.4), (
        f"best_value should be 0.4, got {cb.best_value}"
    )
    assert cb.best_step == 2, (
        f"best_step should be 2 (epoch index), got {cb.best_step}"
    )

    # Convergence at epoch 2 (first epoch where val_loss <= 0.5)
    assert cb.convergence_step == 2, (
        f"convergence_step should be 2, got {cb.convergence_step}"
    )

    # Status transitions
    assert cb.status == "running", "Status should be 'running' before fit ends"
    cb.on_fit_end(MagicMock(), MagicMock())
    assert cb.status == "completed", "Status should be 'completed' after on_fit_end"

    # --- Failure path ---
    cb_fail = RunTrackerCallback(monitor="val_loss", mode="min")
    assert cb_fail.status == "running"
    cb_fail.on_exception(MagicMock(), MagicMock(), RuntimeError("boom"))
    assert cb_fail.status == "failed", (
        "Status should be 'failed' after on_exception"
    )

    # --- No-threshold case: convergence_step stays None ---
    cb_no_thresh = RunTrackerCallback(monitor="val_loss", mode="min")
    for epoch, loss in enumerate([0.9, 0.3]):
        trainer_mock = MagicMock()
        trainer_mock.callback_metrics = {"val_loss": loss}
        trainer_mock.current_epoch = epoch
        cb_no_thresh.on_validation_epoch_end(trainer_mock, MagicMock())

    assert cb_no_thresh.convergence_step is None, (
        "convergence_step must be None when no threshold is configured"
    )
    assert cb_no_thresh.best_value == pytest.approx(0.3)

    # --- mode="max" ---
    cb_max = RunTrackerCallback(monitor="val_acc", mode="max", convergence_threshold=0.9)
    for epoch, acc in enumerate([0.7, 0.85, 0.95]):
        trainer_mock = MagicMock()
        trainer_mock.callback_metrics = {"val_acc": acc}
        trainer_mock.current_epoch = epoch
        cb_max.on_validation_epoch_end(trainer_mock, MagicMock())

    assert cb_max.best_value == pytest.approx(0.95)
    assert cb_max.best_step == 2
    assert cb_max.convergence_step == 2  # first epoch where acc >= 0.9
