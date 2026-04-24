# Global Rules (highest priority)

## Environment
- This project's conda environment is **`research-template`**, defined by
  `environment.yml` at the repo root. The same identifier is used as
  `project_name` in `configs/config.yaml` and as the MLflow `experiment_name`
  in `configs/logger/mlflow.yaml`.
- Always run acceptance commands inside this env: `conda activate research-template`.
- Never create the env with `conda create -n <other-name>` — always
  `conda env create -f environment.yml` so the name stays canonical.

## Anti-patterns (hard constraints)
- `model.py` MUST NOT read cfg, use logger, or contain training logic.
- Network structure lives ONLY in `methods/<name>/model.py`.
- `inference/` pipeline MUST NOT modify cfg or training flow.
- `run_optuna` MUST only modify a copy of cfg, never in-place.

## Code style
- Keep functions focused; no helper abstraction for one-off use.
- No docstrings / comments on unchanged code.
- No backwards-compat shims; delete unused code outright.
- Validate only at system boundaries (user input, external APIs).

## Naming & paths
- Checkpoint: `{method}-{dataset}-{timestamp}-epoch={E:02d}-{monitor}={val:.4f}.ckpt`
  where `{monitor}` is `cfg.experiment.monitor` (e.g. `val_acc`, `val_f1`).
  Earlier template versions hard-coded `val=` regardless of which metric was
  monitored — that is no longer permitted; the filename must name the
  metric so a glance at the path tells you what the value means.
- Use `latest_checkpoint.json` (not symlink) for Windows compat.
- MLflow: log merged `config.yaml` + key artifacts per run.

## Evaluation metrics (hard constraint)
- `loss` is a training signal, not an evaluation metric. It may appear as
  `cfg.experiment.monitor` (and thus drive `ModelCheckpoint`, `EarlyStopping`,
  `RunTrackerCallback`, Optuna, and `run_summary.json::final_metric`) **only**
  when `cfg.experiment.kind == "convergence"`.
- For every other experiment kind (`evaluation`, `robust`, ablations,
  cross-method comparisons, hyperparameter search, paper numbers), the
  monitor MUST be a real task metric — accuracy, F1, AUROC, ECE, BLEU,
  perplexity, etc. — not a loss / NLL.
- `run_train` and `run_optuna` validate this via `_validate_experiment(cfg)`
  at start-up and refuse to run if the monitor looks loss-like for a
  non-convergence kind. Do not work around the guard; pick a task metric or
  switch the experiment kind.
- LR-scheduler `monitor` (e.g. `ReduceLROnPlateau`) is exempt — it is a
  local optimisation detail and may use loss freely.

## Tracking (hard constraints)
- Every `run_train` call MUST write `run_summary.json` to the working directory
  AND upload it as a MLflow artifact under `summary/`, even on exception.
- `run_summary.json` MUST contain exactly these 9 fields:
  `trial_id, seed, params, final_metric, best_metric, convergence_step,
  status, wall_time, traceback`
- `status` MUST be `"completed"` on success and `"failed"` on any exception.
- `traceback` MUST be the full formatted traceback string on failure, `null` on success.
- `tests/test_tracking.py` is a regression guard for these contracts — it MUST NOT be deleted.
