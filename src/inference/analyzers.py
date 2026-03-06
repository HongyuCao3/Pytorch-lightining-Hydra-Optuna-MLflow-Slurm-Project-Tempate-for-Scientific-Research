"""Analyzers: checkpoint -> granular analysis artifacts.

Each analyzer takes aggregated results dict and writes artifacts to disk.

Enabled analyzers are listed in cfg.inference.analyzers:
  - confusion_matrix
  - per_class_metrics
  - tsne_embedding
  - calibration       (optional)
  - saliency          (optional, requires grad)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Individual analyzers
# ---------------------------------------------------------------------------

def analyze_confusion_matrix(
    results: Dict[str, Any],
    output_dir: Path,
) -> str:
    """Generate and save a confusion matrix plot."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    except ImportError:
        log.warning("matplotlib / sklearn not available; skipping confusion_matrix")
        return ""

    y_true = results["true_label"].cpu().numpy()
    y_pred = results["pred"].cpu().numpy()

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=True)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Confusion matrix saved to {path}")
    return str(path)


def analyze_per_class_metrics(
    results: Dict[str, Any],
    output_dir: Path,
) -> str:
    """Compute per-class precision / recall / F1 and save as JSON."""
    try:
        from sklearn.metrics import classification_report
    except ImportError:
        log.warning("sklearn not available; skipping per_class_metrics")
        return ""

    y_true = results["true_label"].cpu().numpy()
    y_pred = results["pred"].cpu().numpy()

    report = classification_report(y_true, y_pred, output_dict=True)

    path = output_dir / "per_class_metrics.json"
    path.write_text(json.dumps(report, indent=2))
    log.info(f"Per-class metrics saved to {path}")
    return str(path)


def analyze_tsne_embedding(
    results: Dict[str, Any],
    output_dir: Path,
) -> str:
    """Compute 2-D t-SNE on predicted probabilities and save scatter plot."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        log.warning("matplotlib / sklearn not available; skipping tsne_embedding")
        return ""

    probs = results["prob"].cpu().numpy()
    y_true = results["true_label"].cpu().numpy()

    n_samples = probs.shape[0]
    if n_samples < 5:
        log.warning("Too few samples for t-SNE; skipping")
        return ""

    perplexity = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=500)
    embedding = tsne.fit_transform(probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=y_true, cmap="tab10", alpha=0.7, s=20)
    plt.colorbar(scatter, ax=ax, label="True Class")
    ax.set_title("t-SNE of Predicted Probabilities")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    plt.tight_layout()

    path = output_dir / "tsne_embedding.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"t-SNE embedding saved to {path}")
    return str(path)


def analyze_calibration(
    results: Dict[str, Any],
    output_dir: Path,
) -> str:
    """Reliability / calibration diagram for the top predicted class."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.calibration import CalibrationDisplay
    except ImportError:
        log.warning("matplotlib / sklearn not available; skipping calibration")
        return ""

    probs = results["prob"].cpu().numpy()
    y_true = results["true_label"].cpu().numpy()
    n_classes = probs.shape[1]

    fig, ax = plt.subplots(figsize=(6, 5))
    for cls in range(n_classes):
        y_bin = (y_true == cls).astype(int)
        prob_cls = probs[:, cls]
        if y_bin.sum() == 0:
            continue
        CalibrationDisplay.from_predictions(y_bin, prob_cls, n_bins=5, ax=ax, name=f"Class {cls}")
    ax.set_title("Calibration Curves")
    plt.tight_layout()

    path = output_dir / "calibration.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Calibration plot saved to {path}")
    return str(path)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_ANALYZER_MAP = {
    "confusion_matrix": analyze_confusion_matrix,
    "per_class_metrics": analyze_per_class_metrics,
    "tsne_embedding": analyze_tsne_embedding,
    "calibration": analyze_calibration,
}


def run_analyzers(
    results: Dict[str, Any],
    output_dir: Path,
    enabled: List[str],
) -> Dict[str, str]:
    """Run all enabled analyzers and return {name: artifact_path}."""
    artifact_paths: Dict[str, str] = {}
    for name in enabled:
        if name not in _ANALYZER_MAP:
            log.warning(f"Unknown analyzer '{name}'; skipping. Available: {list(_ANALYZER_MAP)}")
            continue
        log.info(f"Running analyzer: {name}")
        try:
            path = _ANALYZER_MAP[name](results, output_dir)
            artifact_paths[name] = path
        except Exception as e:
            log.error(f"Analyzer '{name}' failed: {e}")
            artifact_paths[name] = ""
    return artifact_paths
