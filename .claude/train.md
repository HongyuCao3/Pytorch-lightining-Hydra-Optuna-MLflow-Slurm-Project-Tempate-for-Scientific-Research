# Train Rules (applies to src/train.py and Trainer usage)

## run_train contract
```python
def run_train(cfg: DictConfig) -> Dict[str, Any]:
    # seed → datamodule → model → logger → Trainer → fit → test → return metrics
```
- Must return a dict with final val/test metrics.
- Trainer instantiated from `cfg.trainer` via `instantiate`.

## run_optuna contract
- `objective(trial)` returns a single scalar metric (e.g., `val_loss`).
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
