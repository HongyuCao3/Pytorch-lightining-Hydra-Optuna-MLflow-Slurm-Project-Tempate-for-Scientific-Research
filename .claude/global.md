# Global Rules (highest priority)

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
- Checkpoint: `{method}-{dataset}-{timestamp}-epoch={E:02d}-val={val:.4f}.ckpt`
- Use `latest_checkpoint.json` (not symlink) for Windows compat.
- MLflow: log merged `config.yaml` + key artifacts per run.

## Tracking (hard constraints)
- Every `run_train` call MUST write `run_summary.json` to the working directory
  AND upload it as a MLflow artifact under `summary/`, even on exception.
- `run_summary.json` MUST contain exactly these 9 fields:
  `trial_id, seed, params, final_metric, best_metric, convergence_step,
  status, wall_time, traceback`
- `status` MUST be `"completed"` on success and `"failed"` on any exception.
- `traceback` MUST be the full formatted traceback string on failure, `null` on success.
- `tests/test_tracking.py` is a regression guard for these contracts — it MUST NOT be deleted.
