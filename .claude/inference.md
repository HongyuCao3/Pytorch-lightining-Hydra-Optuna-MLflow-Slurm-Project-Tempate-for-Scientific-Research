# Inference Rules (applies to src/inference/**)

## Two post-hoc entrypoints
| Mode | Function | Purpose |
|---|---|---|
| `mode=infer` | `run_infer` → `run_inference_pipeline` | Visual artifacts: confusion matrix, tSNE, calibration PNGs, predictions.pt. |
| `mode=eval`  | `run_eval`  (in `src/train.py`)         | Scalar metrics only: `test_*` via `trainer.test` + `eval/*` via metric_analyzers. Logged to MLflow. |

`run_eval` is the only path for "re-evaluate a checkpoint with new metrics". It never writes visual artifacts. Users run `mode=infer` separately for plots.

## pipeline.py
- `predict_and_aggregate(model, datamodule, cfg)` is the shared helper used by both `run_inference_pipeline` and `run_eval`. It runs `trainer.predict()` and aggregates outputs; callers decide what to do with the result.
- `run_inference_pipeline` then runs visual analyzers on the result.
- `run_eval` runs scalar metric analyzers on the result.
- MUST NOT modify cfg or any training-side state.

## predict_step contract (enforced here)
- Input: batch from DataModule.
- Output: `{"pred": Tensor, "prob": Tensor, "meta": {...}}`.

## analyzers.py — visual artifacts (mode=infer)
| Analyzer | Output |
|---|---|
| `confusion_matrix` | plot + csv |
| `per_class_metrics` | json |
| `tsne_embedding` | plot |
| `calibration` | plot |

- Enabled via `cfg.inference.analyzers` list.

## metric_analyzers.py — scalar metrics (mode=eval)
| Analyzer | Output keys |
|---|---|
| `accuracy` | `accuracy` |
| `macro_f1` | `macro_f1` |
| `weighted_f1` | `weighted_f1` |
| `ece` | `ece` (reads `cfg.inference.metric_analyzer_opts.ece.n_bins`) |
| `brier` | `brier` |
| `classification_report_flat` | `precision_<cls>`, `recall_<cls>`, `f1-score_<cls>`, `support_<cls>`, plus macro/weighted rollups |

- Enabled via `cfg.inference.metric_analyzers` list.
- Each returns `Dict[str, float]`. Logged to MLflow as `eval/<key>` by `log_eval_metrics`.
- Extension: use `register_metric_analyzer("name", fn)` to add new metrics **without** touching `lit_module.test_step`. This is the canonical way to add a post-hoc metric to an already-trained checkpoint.

## Consistency contract (C3)
Every checkpoint saved by `run_train` embeds an `eval_snapshot` dict at the top level of the Lightning checkpoint via `src/methods/_eval_snapshot_mixin.py`. The snapshot records:
`dataset_target`, `split_seed`, `val_split`, `test_split`, `method_target`, `snapshot_version`.

On `run_eval`:
- `read_eval_snapshot(ckpt_path)` pulls the dict from the file (returns `None` for legacy checkpoints).
- `compare_eval_snapshot(snap, current)` reports any disagreement.
- `cfg.eval.strict_snapshot=true` → raise `RuntimeError` on mismatch.
- `cfg.eval.strict_snapshot=false` (default) → log a WARNING, still run.

This is the mechanism that enforces "every eval run, and every baseline comparison, shares identical data-split conditions" as a hard constraint rather than a convention. `run_train` also writes the same canonical keys as MLflow **tags** so cross-run filtering via `mlflow.search_runs(filter_string="tags.split_seed = '42'")` is possible.

## Constraints
- No training logic, no optimizer, no loss computation in this module.
- New metrics MUST be added via `metric_analyzers.py`, never by editing `lit_module.test_step`.
