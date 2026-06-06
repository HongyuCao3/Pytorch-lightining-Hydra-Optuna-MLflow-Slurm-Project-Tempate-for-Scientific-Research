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

## Reporting standard (hard constraint)
- The ONLY number that may be reported as a result — in a paper, slide, weekly
  update, or any cross-method/cross-config comparison — is
  `cfg.experiment.report_metric` (a held-out **test** task metric, `test_*`
  prefix) averaged over `cfg.experiment.seeds` (>= 3 distinct seeds), quoted as
  **`mean ± std (n)`**. This is exactly what `mode=bench` (`run_bench`) produces
  in `bench_summary.json::headline`.
- FORBIDDEN as a reported/compared headline number: a single-seed point
  estimate; any `val_*` metric; any loss / NLL; a "mean rank" or other
  cross-table ordinal; an Optuna `best_value` (that is a val number chosen on
  the search split, for selection only). None of these are comparable across
  methods or runs.
- `run_bench` enforces this via `_validate_report(cfg)`: it refuses a
  `report_metric` that is not `test_*` or that looks loss-like, and refuses
  fewer than 3 distinct seeds. Do not work around the guard — fix the config.
- A single `run_train` / `run_eval` produces a model and diagnostics, NOT a
  reportable result. Tuning (`run_optuna`) selects a config on validation; the
  selected config's reportable number still comes from a subsequent `mode=bench`
  run. Selection (val) and reporting (test) stay decoupled — never tune on test.
- `src/scripts/aggregate_runs.py --group-by ... --metric test_*` is the post-hoc
  equivalent for sweeping a param across seeds; it aggregates a `test_*` metric
  only, never `final_metric`.

## Tracking (hard constraints)
- Every `run_train` call MUST write `run_summary.json` to the working directory
  AND upload it as a MLflow artifact under `summary/`, even on exception.
- `run_summary.json` MUST contain exactly these 9 fields:
  `trial_id, seed, params, final_metric, best_metric, convergence_step,
  status, wall_time, traceback`
- `status` MUST be `"completed"` on success and `"failed"` on any exception.
- `traceback` MUST be the full formatted traceback string on failure, `null` on success.
- `tests/test_tracking.py` is a regression guard for these contracts — it MUST NOT be deleted.
