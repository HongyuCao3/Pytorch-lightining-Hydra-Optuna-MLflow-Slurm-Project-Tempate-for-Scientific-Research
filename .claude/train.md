# Train Rules (applies to src/train.py and Trainer usage)

## Experiment objective schema (cfg.experiment)
Selected via the Hydra group `experiment` (preset files in `configs/experiment/`).
Every `run_train` / `run_optuna` reads the same five fields:

| field | meaning |
|---|---|
| `kind` | `convergence` \| `evaluation` \| `robust` |
| `monitor` | logged metric key (drives `ModelCheckpoint`, `EarlyStopping`, `RunTrackerCallback`, Optuna, `run_summary.json::final_metric`, Hydra sweeper return value) |
| `mode` | `min` or `max` |
| `patience` | EarlyStopping patience in epochs |
| `convergence_threshold` | first epoch crossing this is recorded as `convergence_step` (null = disabled) |

`_validate_experiment(cfg)` is called at the top of `run_train` and
`run_optuna`. It refuses any non-`convergence` kind whose `monitor` looks
loss-like (`loss` / `nll` substrings). See `.claude/global.md` →
*Evaluation metrics*.

## run_train contract
```python
def run_train(cfg: DictConfig) -> Dict[str, Any]:
    # seed → datamodule → model → attach_eval_snapshot → logger → log_tags
    # → Trainer → fit → (auto) test → return metrics
```
- Must return a dict with final val/test metrics.
- Trainer instantiated from `cfg.trainer` via `instantiate`.
- MUST call `model.attach_eval_snapshot({...})` before `fit()` when the model
  subclasses `EvalSnapshotMixin`, so every saved checkpoint carries an
  `eval_snapshot` key. `run_eval` reads this back to enforce identical
  re-eval conditions. See `.claude/inference.md` → *Consistency contract*.
- MUST emit lineage MLflow tags (`split_seed`, `val_split`, `test_split`,
  `dataset_target`, `method_target`, `run_kind=train`) via `log_tags`
  immediately after the logger is instantiated. `run_eval` records the
  same keys plus `parent_run_id` + `checkpoint_sha256` so cross-run
  comparisons can be filtered by `mlflow.search_runs(filter_string=...)`.

## run_eval contract
```python
def run_eval(cfg: DictConfig) -> Dict[str, float]:
    # validate ckpt → read_eval_snapshot → compare → warn/raise per strict flag
    # → load model → datamodule → trainer.test → predict_and_aggregate
    # → run_metric_analyzers → log_eval_metrics(prefix="eval")
```
- Scalars-only. Visual artifacts stay in `run_infer`.
- MUST pass `ckpt_path=None` to `trainer.test` — weights are already loaded
  via `load_from_checkpoint`; passing the path would trigger a redundant reload.
- MUST tag the new MLflow run with `run_kind=eval`, `parent_run_id`,
  `checkpoint_sha256`, `snapshot_status` (`match` / `mismatch` / `unknown`),
  and the full lineage keys.
- MUST NOT modify cfg or retrain anything.

## run_optuna contract
- `objective(trial)` returns a single scalar metric. The metric **defaults to
  `cfg.experiment.monitor`** with direction derived from `cfg.experiment.mode`
  (`min` → `minimize`, `max` → `maximize`). Override only via `cfg.optuna.metric`
  / `cfg.optuna.direction` and only when the search must intentionally
  optimise something different from the experiment's primary metric (rare).
- Only modify a **copy** of cfg (never in-place).
- Integrate with MLflow via `MLflowCallback`; each trial = MLflow child run.
- Storage: RDB, configurable via `cfg.optuna.storage`.

## OOM handling
- `callbacks/oom_handler.py` injected via `cfg.callbacks`.
- On OOM: `empty_cache()` → save temp checkpoint → log to MLflow → re-raise.

## Checkpoint
- `ModelCheckpoint` uses naming: `{method}-{dataset}-{timestamp}-epoch={E:02d}-val={val:.4f}.ckpt`.
- Write `latest_checkpoint.json` after each save (Windows-safe, no symlink).

## Constraints
- No network structure in this file.
- No direct DataLoader creation; always use DataModule.

## Tracking contract (hard constraint)
- `run_train` MUST instantiate `RunTrackerCallback` and include it in the callbacks list.
- `run_train` MUST wrap `trainer.fit()` and `trainer.test()` in a try/except that:
    1. Calls `log_exception_to_run(exc)` before re-raising.
    2. Builds and logs `run_summary.json` (via `log_run_summary`) unconditionally,
       even on failure.
    3. Re-raises the original exception **after** the summary is written.
- `run_optuna` MUST inject `optuna._trial_id = trial.number` into each per-trial cfg copy
  so that `run_train` can embed it in the summary.
- `run_optuna` MUST write both `optuna_summary.json` (with a `trials` list) and
  `optuna_summary.csv` after all trials complete.
- Never call `mlflow` directly from `train.py`; use `src/utils/mlflow_utils.py` helpers.
