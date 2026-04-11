"""Tests for post-hoc ``mode=eval`` and the consistency machinery backing it.

Coverage map from the plan:
  1. test_split_seed_reproducible  (C1)
  2. test_split_seed_differs       (C1)
  3. test_metric_analyzers_registry (C2)
  4. test_snapshot_roundtrip       (C3)
  5. test_eval_matches_auto_test   (C4 — the load-bearing round-trip)
  6. test_snapshot_mismatch_*      (C3 + C4 enforcement)
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from src.data.wine_datamodule import WineDataModule
from src.inference.metric_analyzers import (
    available_metric_analyzers,
    register_metric_analyzer,
    run_metric_analyzers,
)
from src.methods._eval_snapshot_mixin import (
    EVAL_SNAPSHOT_KEYS,
    compare_eval_snapshot,
    read_eval_snapshot,
)
from src.train import run_eval, run_train


# ---------------------------------------------------------------------------
# C1: split_seed determinism
# ---------------------------------------------------------------------------

def test_split_seed_reproducible():
    a = WineDataModule(split_seed=42)
    a.setup()
    b = WineDataModule(split_seed=42)
    b.setup()
    assert torch.equal(a.train_ds.tensors[0], b.train_ds.tensors[0])
    assert torch.equal(a.val_ds.tensors[0], b.val_ds.tensors[0])
    assert torch.equal(a.test_ds.tensors[0], b.test_ds.tensors[0])


def test_split_seed_differs():
    a = WineDataModule(split_seed=42)
    a.setup()
    c = WineDataModule(split_seed=7)
    c.setup()
    assert not torch.equal(a.train_ds.tensors[0], c.train_ds.tensors[0])


# ---------------------------------------------------------------------------
# C2: metric_analyzers registry
# ---------------------------------------------------------------------------

def _make_synth_results(n: int = 30, n_classes: int = 3, seed: int = 0):
    rng = torch.Generator().manual_seed(seed)
    y_true = torch.randint(0, n_classes, (n,), generator=rng)
    # Give the "right" class a higher logit most of the time
    logits = torch.randn(n, n_classes, generator=rng)
    for i in range(int(n * 0.8)):
        logits[i] = 0.0
        logits[i, y_true[i]] = 5.0
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    return {"pred": preds, "prob": probs, "true_label": y_true}


def test_metric_analyzers_builtin_registry():
    # Ensure the canonical ones are registered
    avail = available_metric_analyzers()
    for required in ("accuracy", "macro_f1", "ece", "brier", "classification_report_flat"):
        assert required in avail

    results = _make_synth_results()
    out = run_metric_analyzers(results, enabled=["accuracy", "macro_f1", "ece", "brier"])
    assert set(out.keys()) == {"accuracy", "macro_f1", "ece", "brier"}
    for k, v in out.items():
        assert isinstance(v, float), f"{k} is not float"
        assert 0.0 <= v <= 5.0, f"{k}={v} out of sane range"
    # Synthetic set is ~80% accurate
    assert out["accuracy"] >= 0.7


def test_metric_analyzers_custom_registration():
    def _dummy(results, cfg=None):
        return {"always_one": 1.0}

    register_metric_analyzer("dummy_always_one", _dummy)
    results = _make_synth_results()
    out = run_metric_analyzers(results, enabled=["dummy_always_one"])
    assert out == {"always_one": 1.0}


def test_metric_analyzers_unknown_is_skipped():
    results = _make_synth_results()
    out = run_metric_analyzers(results, enabled=["does_not_exist", "macro_f1"])
    # Unknown is silently skipped, macro_f1 still runs
    assert "macro_f1" in out
    assert "does_not_exist" not in out


# ---------------------------------------------------------------------------
# C3: snapshot mixin round-trip + mismatch detection
# ---------------------------------------------------------------------------

def test_snapshot_roundtrip(tmp_path, project_root, monkeypatch):
    """Train 1 real epoch; verify the saved ckpt carries eval_snapshot."""
    monkeypatch.chdir(tmp_path)

    cfg = _make_cfg(
        project_root,
        tmp_path,
        overrides=[
            "mode=train",
            "dataset=wine",
            "method=mlp",
            "trainer.max_epochs=1",
            "mode.run_test=false",
        ],
    )
    run_train(cfg)

    record = json.loads(Path("latest_checkpoint.json").read_text())
    ckpt_path = record["latest_checkpoint"]
    assert Path(ckpt_path).exists()

    snap = read_eval_snapshot(ckpt_path)
    assert snap is not None, "snapshot missing from checkpoint"
    for key in EVAL_SNAPSHOT_KEYS:
        assert key in snap, f"snapshot missing required key: {key}"
    assert snap["split_seed"] == 42
    assert snap["dataset_target"].endswith("WineDataModule")
    assert snap["method_target"].endswith("MLPLitModule")
    assert snap["snapshot_version"] == 1


def test_compare_eval_snapshot_detects_mismatch():
    snap = {
        "dataset_target": "src.data.wine_datamodule.WineDataModule",
        "split_seed": 42,
        "val_split": 0.2,
        "test_split": 0.1,
        "method_target": "src.methods.mlp.lit_module.MLPLitModule",
    }
    current = dict(snap)
    assert compare_eval_snapshot(snap, current) == {}

    current["split_seed"] = 7
    diffs = compare_eval_snapshot(snap, current)
    assert "split_seed" in diffs
    assert diffs["split_seed"] == (42, 7)


def test_compare_eval_snapshot_none_returns_empty():
    # Legacy checkpoints (no snapshot) → empty diffs, caller treats as unknown
    assert compare_eval_snapshot(None, {"split_seed": 42}) == {}


# ---------------------------------------------------------------------------
# C4: the load-bearing round-trip — eval metrics == train-time test metrics
# ---------------------------------------------------------------------------

def _make_cfg(project_root, tmp_path, overrides):
    """Compose a cfg with an isolated per-test MLflow tracking URI.

    Mirrors the pattern used in ``tests/test_tracking.py``: points
    ``logger.tracking_uri`` at a ``file://`` URI under ``tmp_path`` so runs
    never pollute the project-level ``mlruns/``. Also clears Hydra global
    state so multiple composes can happen within one test.

    Checkpoint paths are set post-compose via ``OmegaConf.update`` because
    the default ModelCheckpoint filename template contains ``=`` characters
    that confuse Hydra's CLI override parser.
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


def test_eval_matches_auto_test(tmp_path, project_root, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # ---- Train ----
    train_cfg = _make_cfg(
        project_root,
        tmp_path,
        overrides=["mode=train", "dataset=wine", "method=mlp", "trainer.max_epochs=2"],
    )
    train_metrics = run_train(train_cfg)
    assert "test_acc" in train_metrics
    assert "test_loss" in train_metrics

    ckpt = json.loads(Path("latest_checkpoint.json").read_text())["latest_checkpoint"]
    assert Path(ckpt).exists()

    # ---- Eval on that checkpoint ----
    eval_cfg = _make_cfg(
        project_root,
        tmp_path,
        overrides=["mode=eval", "dataset=wine", "method=mlp"],
    )
    OmegaConf.update(eval_cfg, "inference.checkpoint_path", ckpt)
    eval_metrics = run_eval(eval_cfg)

    # ---- Correctness: test metrics must match bit-for-bit (floating point) ----
    assert abs(train_metrics["test_acc"] - eval_metrics["test_acc"]) < 1e-6
    assert abs(train_metrics["test_loss"] - eval_metrics["test_loss"]) < 1e-5

    # ---- And the scalar metric analyzers produced sensible values ----
    assert "eval/macro_f1" in eval_metrics
    assert "eval/ece" in eval_metrics
    assert "eval/brier" in eval_metrics
    for k in ("eval/macro_f1", "eval/ece", "eval/brier"):
        assert 0.0 <= eval_metrics[k] <= 5.0


def test_eval_snapshot_strict_raises_on_mismatch(tmp_path, project_root, monkeypatch):
    """strict_snapshot=true turns a snapshot/cfg disagreement into a hard error."""
    monkeypatch.chdir(tmp_path)

    train_cfg = _make_cfg(
        project_root,
        tmp_path,
        overrides=["mode=train", "dataset=wine", "method=mlp", "trainer.max_epochs=1"],
    )
    run_train(train_cfg)
    ckpt = json.loads(Path("latest_checkpoint.json").read_text())["latest_checkpoint"]

    eval_cfg = _make_cfg(
        project_root,
        tmp_path,
        overrides=["mode=eval", "dataset=wine", "method=mlp"],
    )
    OmegaConf.update(eval_cfg, "inference.checkpoint_path", ckpt)
    # Force a mismatch — checkpoint has split_seed=42, override to 7
    OmegaConf.update(eval_cfg, "dataset.split_seed", 7)
    OmegaConf.update(eval_cfg, "eval.strict_snapshot", True)

    with pytest.raises(RuntimeError, match="snapshot mismatch"):
        run_eval(eval_cfg)


def test_eval_snapshot_nonstrict_warns_on_mismatch(tmp_path, project_root, monkeypatch):
    """strict_snapshot=false (default) downgrades a mismatch to a WARNING only.

    The eval still runs; metrics are returned. We assert both (a) that the run
    completes end-to-end and (b) that a "snapshot mismatch" warning was logged.

    Note: ``src.utils.logging.get_logger`` sets ``propagate=False`` so pytest's
    built-in ``caplog`` does not receive records. We attach a private handler
    to the ``src.train`` logger for the duration of the test instead.
    """
    import logging

    monkeypatch.chdir(tmp_path)

    train_cfg = _make_cfg(
        project_root,
        tmp_path,
        overrides=["mode=train", "dataset=wine", "method=mlp", "trainer.max_epochs=1"],
    )
    run_train(train_cfg)
    ckpt = json.loads(Path("latest_checkpoint.json").read_text())["latest_checkpoint"]

    eval_cfg = _make_cfg(
        project_root,
        tmp_path,
        overrides=["mode=eval", "dataset=wine", "method=mlp"],
    )
    OmegaConf.update(eval_cfg, "inference.checkpoint_path", ckpt)
    OmegaConf.update(eval_cfg, "dataset.split_seed", 7)
    # strict_snapshot defaults to false from configs/eval/default.yaml

    # Capture warnings from the non-propagating src.train logger.
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _ListHandler(level=logging.WARNING)
    train_logger = logging.getLogger("src.train")
    train_logger.addHandler(handler)
    try:
        eval_metrics = run_eval(eval_cfg)
    finally:
        train_logger.removeHandler(handler)

    assert any("snapshot mismatch" in r.getMessage().lower() for r in records), \
        f"expected a snapshot-mismatch warning, got: {[r.getMessage() for r in records]}"
    assert "test_acc" in eval_metrics
