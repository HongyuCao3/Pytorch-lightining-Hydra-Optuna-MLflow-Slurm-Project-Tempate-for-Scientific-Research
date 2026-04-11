"""Inference pipeline: checkpoint -> predict -> postprocess -> artifacts.

Entry point: run_inference_pipeline(cfg)

Pipeline steps
--------------
1. Load LightningModule from checkpoint
2. Run trainer.predict() on the datamodule's predict_dataloader
3. Aggregate per-batch predict_step outputs into flat arrays
4. Postprocess (threshold / argmax / etc.)
5. Run each enabled analyzer and collect artifact paths
6. Upload artifacts to MLflow (or write to output_dir)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

import hydra

from src.utils.logging import get_logger
from src.utils.mlflow_utils import log_artifact_to_run, log_config_to_run

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_predictions(
    batch_outputs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Concatenate list of predict_step dicts into flat arrays.

    Each element must satisfy: {"pred": Tensor, "prob": Tensor, "meta": {...}}
    """
    preds = torch.cat([o["pred"] for o in batch_outputs], dim=0)
    probs = torch.cat([o["prob"] for o in batch_outputs], dim=0)
    true_labels = torch.cat([o["meta"]["true_label"] for o in batch_outputs], dim=0)
    return {
        "pred": preds,
        "prob": probs,
        "true_label": true_labels,
    }


# ---------------------------------------------------------------------------
# Postprocess
# ---------------------------------------------------------------------------

def postprocess(aggregated: Dict[str, Any], cfg_post: DictConfig) -> Dict[str, Any]:
    """Apply postprocessing to aggregated predictions.

    Currently a pass-through; extend for threshold tuning, ensembling, etc.
    """
    return aggregated


# ---------------------------------------------------------------------------
# Predict + aggregate (reusable by run_eval — scalars-only path)
# ---------------------------------------------------------------------------

def predict_and_aggregate(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    cfg: DictConfig,
) -> Dict[str, Any]:
    """Run ``trainer.predict`` on an already-loaded model and aggregate outputs.

    Extracted from ``run_inference_pipeline`` so ``run_eval`` can reuse the
    predict + aggregate step without pulling in the visual-analyzer loop.
    The caller is responsible for loading the checkpoint and calling
    ``datamodule.setup(...)`` if needed.

    Returns the dict from :func:`aggregate_predictions` plus an (optional)
    postprocess pass reading ``cfg.inference.postprocess``.
    """
    model.eval()

    trainer = pl.Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=True,
    )
    batch_outputs: List[Dict[str, Any]] = trainer.predict(
        model, dataloaders=datamodule.predict_dataloader()
    )

    aggregated = aggregate_predictions(batch_outputs)
    log.info(f"Predictions aggregated: {aggregated['pred'].shape[0]} samples")

    post_cfg = OmegaConf.select(cfg, "inference.postprocess") or OmegaConf.create({})
    return postprocess(aggregated, post_cfg)


# ---------------------------------------------------------------------------
# Main pipeline (mode=infer: predict + aggregate + visual artifacts)
# ---------------------------------------------------------------------------

def run_inference_pipeline(cfg: DictConfig) -> Dict[str, Any]:
    """Full inference pipeline.

    Parameters
    ----------
    cfg : merged Hydra config with cfg.inference populated.

    Returns
    -------
    artifact_paths : dict mapping analyzer name -> local file path
    """
    from src.inference.analyzers import run_analyzers

    infer_cfg = cfg.inference
    output_dir = Path(infer_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = infer_cfg.checkpoint_path
    log.info(f"Loading checkpoint: {checkpoint_path}")

    # --- Load model ---------------------------------------------------------
    # Determine LightningModule class from config
    model_cls_target: str = cfg.method._target_
    # Import and get the class
    from hydra._internal.utils import _locate
    model_cls = _locate(model_cls_target)
    model: pl.LightningModule = model_cls.load_from_checkpoint(checkpoint_path)

    # --- Instantiate datamodule ---------------------------------------------
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.dataset)
    datamodule.setup("predict")

    # --- Predict + aggregate (+ postprocess) via the shared helper ----------
    results = predict_and_aggregate(model, datamodule, cfg)

    # --- Save raw predictions -----------------------------------------------
    preds_path = output_dir / "predictions.pt"
    torch.save(results, str(preds_path))
    log.info(f"Predictions saved to {preds_path}")

    # --- Run analyzers -------------------------------------------------------
    enabled_analyzers = list(infer_cfg.analyzers) if infer_cfg.analyzers else []
    artifact_paths = run_analyzers(
        results=results,
        output_dir=output_dir,
        enabled=enabled_analyzers,
    )
    artifact_paths["predictions"] = str(preds_path)

    # --- Log artifacts to MLflow --------------------------------------------
    try:
        import mlflow
        if mlflow.active_run():
            for name, path in artifact_paths.items():
                if path and Path(path).exists():
                    mlflow.log_artifact(path, artifact_path="inference")
    except Exception as e:
        log.warning(f"MLflow artifact logging skipped: {e}")

    log.info(f"Inference complete. Artifacts: {list(artifact_paths.keys())}")
    return artifact_paths
