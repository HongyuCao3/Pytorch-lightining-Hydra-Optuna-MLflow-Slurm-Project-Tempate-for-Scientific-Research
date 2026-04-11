"""Scalar metric analyzers: aggregated predictions -> Dict[str, float].

Strict companion to ``src/inference/analyzers.py``:

  * ``analyzers.py``         -> artifact files (PNG / JSON on disk)
  * ``metric_analyzers.py``  -> scalar metrics (logged to MLflow as ``eval/*``)

This split lets post-hoc evaluation add new metrics without touching
``lit_module.test_step`` (training-layer code). ``run_eval`` in ``src/train.py``
is the primary consumer; it calls ``run_metric_analyzers`` on the result of
``predict_and_aggregate`` and pipes the output through
``log_eval_metrics(prefix="eval")``.

Each analyzer has signature::

    fn(results: Dict[str, Any], cfg: Optional[DictConfig] = None) -> Dict[str, float]

``results`` is the aggregated dict produced by
``src/inference/pipeline.py::aggregate_predictions``: ``{"pred", "prob",
"true_label"}``. ``cfg`` is the full merged Hydra config; analyzers may read
their own sub-tree (e.g. ``cfg.inference.metric_analyzer_opts.ece.n_bins``)
but MUST tolerate ``cfg=None`` for unit testing.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(results: Dict[str, Any]):
    y_true = results["true_label"]
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    y_pred = results["pred"]
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    probs = results["prob"]
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    return y_true, y_pred, probs


def _opt(cfg: Optional[DictConfig], dotted: str, default):
    """Safely read a nested key from cfg.inference.metric_analyzer_opts.<dotted>."""
    if cfg is None:
        return default
    val = OmegaConf.select(cfg, f"inference.metric_analyzer_opts.{dotted}", default=default)
    return default if val is None else val


# ---------------------------------------------------------------------------
# Individual metric analyzers
# ---------------------------------------------------------------------------

def metric_macro_f1(
    results: Dict[str, Any],
    cfg: Optional[DictConfig] = None,
) -> Dict[str, float]:
    """Macro-averaged F1. Returns {'macro_f1': float}."""
    from sklearn.metrics import f1_score

    y_true, y_pred, _ = _to_numpy(results)
    return {"macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}


def metric_weighted_f1(
    results: Dict[str, Any],
    cfg: Optional[DictConfig] = None,
) -> Dict[str, float]:
    """Weighted F1 (class-frequency weighted)."""
    from sklearn.metrics import f1_score

    y_true, y_pred, _ = _to_numpy(results)
    return {"weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0))}


def metric_accuracy(
    results: Dict[str, Any],
    cfg: Optional[DictConfig] = None,
) -> Dict[str, float]:
    """Top-1 accuracy. Redundant with test_acc but useful as a sanity scalar."""
    y_true, y_pred, _ = _to_numpy(results)
    if len(y_true) == 0:
        return {"accuracy": 0.0}
    return {"accuracy": float((y_true == y_pred).mean())}


def metric_ece(
    results: Dict[str, Any],
    cfg: Optional[DictConfig] = None,
) -> Dict[str, float]:
    """Expected Calibration Error over the top predicted class.

    Partitions the test set into ``n_bins`` equal-width confidence bins and
    computes the absolute gap between average confidence and accuracy in each
    bin, weighted by bin size. Lower is better.

    Reads ``cfg.inference.metric_analyzer_opts.ece.n_bins`` (default 10).
    """
    y_true, y_pred, probs = _to_numpy(results)
    if len(y_true) == 0:
        return {"ece": 0.0}

    n_bins = int(_opt(cfg, "ece.n_bins", 10))
    confidences = probs.max(axis=1)
    correct = (y_pred == y_true).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        avg_conf = float(confidences[in_bin].mean())
        avg_acc = float(correct[in_bin].mean())
        ece += (count / n) * abs(avg_conf - avg_acc)

    return {"ece": float(ece)}


def metric_brier(
    results: Dict[str, Any],
    cfg: Optional[DictConfig] = None,
) -> Dict[str, float]:
    """Multi-class Brier score (mean squared error between one-hot y and probs).

    ``brier = mean_over_samples( sum_over_classes( (prob_c - onehot_c)^2 ) )``.
    Lower is better.
    """
    y_true, _, probs = _to_numpy(results)
    if len(y_true) == 0:
        return {"brier": 0.0}
    n_classes = probs.shape[1]
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    brier = float(((probs - onehot) ** 2).sum(axis=1).mean())
    return {"brier": brier}


def metric_classification_report_flat(
    results: Dict[str, Any],
    cfg: Optional[DictConfig] = None,
) -> Dict[str, float]:
    """Flatten sklearn ``classification_report`` into scalar MLflow-safe keys.

    Emits ``precision_class_0``, ``recall_class_1``, ``f1_class_2``, ``support_class_0``,
    plus ``macro_precision`` / ``weighted_recall`` / etc. from the averaged rows.
    """
    from sklearn.metrics import classification_report

    y_true, y_pred, _ = _to_numpy(results)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    out: Dict[str, float] = {}
    for key, val in report.items():
        if not isinstance(val, dict):
            # ``accuracy`` row is already a scalar
            out[f"report_{key}"] = float(val)
            continue
        # Replace spaces so the MLflow metric key stays clean
        norm_key = key.replace(" ", "_")
        for metric_name, metric_val in val.items():
            out[f"{metric_name}_{norm_key}"] = float(metric_val)
    return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

MetricAnalyzerFn = Callable[[Dict[str, Any], Optional[DictConfig]], Dict[str, float]]

_METRIC_ANALYZER_MAP: Dict[str, MetricAnalyzerFn] = {
    "accuracy": metric_accuracy,
    "macro_f1": metric_macro_f1,
    "weighted_f1": metric_weighted_f1,
    "ece": metric_ece,
    "brier": metric_brier,
    "classification_report_flat": metric_classification_report_flat,
}


def register_metric_analyzer(name: str, fn: MetricAnalyzerFn) -> None:
    """Register a new metric analyzer (for downstream extension)."""
    if name in _METRIC_ANALYZER_MAP:
        log.warning(f"Overwriting existing metric analyzer: {name!r}")
    _METRIC_ANALYZER_MAP[name] = fn


def available_metric_analyzers() -> List[str]:
    return sorted(_METRIC_ANALYZER_MAP.keys())


def run_metric_analyzers(
    results: Dict[str, Any],
    enabled: List[str],
    cfg: Optional[DictConfig] = None,
) -> Dict[str, float]:
    """Run every enabled metric analyzer and merge their outputs.

    Individual analyzer failures are logged and skipped; never raise.
    Returns a flat ``{metric_name: value}`` dict suitable for
    ``log_eval_metrics``.
    """
    merged: Dict[str, float] = {}
    for name in enabled or []:
        fn = _METRIC_ANALYZER_MAP.get(name)
        if fn is None:
            log.warning(
                f"Unknown metric analyzer {name!r}; skipping. "
                f"Available: {available_metric_analyzers()}"
            )
            continue
        log.info(f"Running metric analyzer: {name}")
        try:
            out = fn(results, cfg)
            if not isinstance(out, dict):
                log.warning(f"Metric analyzer {name!r} returned non-dict; skipping")
                continue
            for k, v in out.items():
                if k in merged:
                    log.warning(f"Metric key collision on {k!r} (overwriting)")
                merged[k] = float(v)
        except Exception as e:
            log.error(f"Metric analyzer {name!r} failed: {e}")
    return merged
