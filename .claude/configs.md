# Configs Rules (applies to configs/**)

## Structure
```
configs/
  config.yaml          # root: defaults list → dataset/method/trainer/logger/mode/optuna/experiment
  dataset/<name>.yaml
  method/<name>.yaml   # must include _target_ → LitModule class path
  trainer/default.yaml
  logger/mlflow.yaml
  mode/debug.yaml      # fast_dev_run: true
  mode/train.yaml
  mode/optuna.yaml
  mode/infer.yaml
  optuna/default.yaml
  inference/default.yaml  # checkpoint_path, analyzers list
  experiment/evaluation.yaml   # task-metric monitor (default)
  experiment/convergence.yaml  # loss monitor — only legal kind for loss
  experiment/robust.yaml       # task metric, aggregated across seeds
  hydra/launcher/slurm.yaml
```

## experiment group (cfg.experiment)
- Carries `kind`, `monitor`, `mode`, `patience`, `convergence_threshold`.
- The `monitor` here is the single source of truth for ModelCheckpoint,
  EarlyStopping, RunTrackerCallback, Optuna, and `run_summary.json`.
- See `.claude/train.md` → *Experiment objective schema* and
  `.claude/global.md` → *Evaluation metrics* for the rules.

## method config requirements
- `_target_`: full Python path to LitModule class.
- All hyperparams with defaults (lr, hidden_dim, dropout, etc.).
- `instantiate(cfg.method)` must return a LightningModule directly.

## inference config
- `checkpoint_path`: path to `.ckpt` file.
- `analyzers`: list of enabled analyzer names.
- `postprocess`: options dict passed to pipeline.

## SLURM launcher
- `configs/hydra/launcher/slurm.yaml` uses `hydra-submitit-launcher`.
- Set `timeout_min`, `mem_gb`, `gpus_per_node`, `partition` here.

## Constraints
- Do not put training logic or Python code in yaml files.
- Trainer strategy (deepspeed/fsdp) configured in `configs/trainer/` only.
- Never hardcode paths; use Hydra interpolation (`${hydra:runtime.cwd}`).
