"""Tests for ``mode=bench`` (run_bench) and the reporting-standard guard.

Reporting standard (see .claude/global.md → 'Reporting standard'): the only
reportable number is ``cfg.experiment.report_metric`` (a held-out ``test_*``
task metric) averaged over ``cfg.experiment.seeds`` (>= 3 distinct seeds) as
``mean ± std``. These tests pin both the guard (_validate_report) and the
aggregation (run_bench).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from src.train import _validate_report, run_bench


# ---------------------------------------------------------------------------
# _validate_report — refuses non-reportable bases
# ---------------------------------------------------------------------------

def _report_cfg(report_metric, seeds):
    return OmegaConf.create(
        {"experiment": {"report_metric": report_metric, "seeds": seeds}}
    )


def test_validate_report_rejects_val_metric():
    with pytest.raises(ValueError, match="held-out test metric"):
        _validate_report(_report_cfg("val_acc", [42, 123, 2024]))


def test_validate_report_rejects_loss():
    with pytest.raises(ValueError, match="loss"):
        _validate_report(_report_cfg("test_loss", [42, 123, 2024]))


def test_validate_report_rejects_too_few_seeds():
    with pytest.raises(ValueError, match=">= 3"):
        _validate_report(_report_cfg("test_acc", [42, 123]))


def test_validate_report_rejects_duplicate_seeds():
    with pytest.raises(ValueError, match="distinct"):
        _validate_report(_report_cfg("test_acc", [42, 42, 42]))


def test_validate_report_accepts_three_test_seeds():
    # Must not raise.
    _validate_report(_report_cfg("test_acc", [42, 123, 2024]))


# ---------------------------------------------------------------------------
# run_bench — aggregates the test metric across seeds into mean ± std
# ---------------------------------------------------------------------------

def _make_cfg(project_root, tmp_path, overrides):
    """Compose a cfg with an isolated per-test MLflow tracking URI.

    Mirrors tests/test_eval_mode.py: points ``logger.tracking_uri`` at a
    ``file://`` URI under ``tmp_path`` so bench runs never pollute the
    project-level ``mlruns/``.
    """
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


def test_run_bench_aggregates_three_seeds(tmp_path, project_root, monkeypatch):
    monkeypatch.chdir(tmp_path)

    cfg = _make_cfg(
        project_root,
        tmp_path,
        overrides=[
            "mode=bench",
            "dataset=wine",
            "method=mlp",
            "experiment=evaluation",
            "trainer.max_epochs=1",
            "experiment.seeds=[42,123,7]",
        ],
    )
    summary = run_bench(cfg)

    # ---- Returned summary shape ----
    assert summary["report_metric"] == "test_acc"
    assert summary["seeds"] == [42, 123, 7]
    assert summary["n"] == 3
    assert set(summary["per_seed"].keys()) == {"42", "123", "7"}
    assert summary["mean"] is not None
    assert summary["std"] is not None
    assert summary["headline"].startswith("test_acc = ")
    assert "(n=3)" in summary["headline"]
    for v in summary["per_seed"].values():
        assert 0.0 <= float(v) <= 1.0

    # ---- Persisted artifacts ----
    written = json.loads(Path("bench_summary.json").read_text())
    assert written == summary
    assert Path("bench_summary.csv").exists()


def test_run_bench_refuses_bad_report_metric(tmp_path, project_root, monkeypatch):
    """The guard fires before any training when report_metric is val-like."""
    monkeypatch.chdir(tmp_path)

    cfg = _make_cfg(
        project_root,
        tmp_path,
        overrides=[
            "mode=bench",
            "dataset=wine",
            "method=mlp",
            "experiment=evaluation",
            "experiment.report_metric=val_acc",
            "experiment.seeds=[42,123,7]",
        ],
    )
    with pytest.raises(ValueError, match="held-out test metric"):
        run_bench(cfg)
