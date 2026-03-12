# Inference Rules (applies to src/inference/**)

## pipeline.py
- Flow: load ckpt → `trainer.predict()` → `postprocess` → call analyzers → upload artifacts.
- MUST NOT modify cfg or any training-side state.
- `run_infer(cfg)` reads `cfg.inference.checkpoint_path`.

## predict_step contract (enforced here)
- Input: batch from DataModule.
- Output: `{"pred": Tensor, "prob": Tensor, "meta": {...}}`.
- Pipeline aggregates outputs across batches before postprocess.

## analyzers.py — available analyzers
| Analyzer | Output |
|---|---|
| `confusion_matrix` | plot + csv |
| `per_class_metrics` | json |
| `tsne_embedding` | plot |
| `calibration` | plot |

- Analyzers enabled via `cfg.inference.analyzers` list.
- Each analyzer writes to a local dir, then `mlflow_utils.log_artifact_to_run` uploads.

## Constraints
- No training logic, no optimizer, no loss computation in this module.
- Do not re-instantiate DataModule; receive predictions from pipeline input.
