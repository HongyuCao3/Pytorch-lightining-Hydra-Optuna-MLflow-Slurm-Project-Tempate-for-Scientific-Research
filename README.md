# PyTorch Lightning + Hydra + Optuna + MLflow + SLURM Research Template

A production-quality scientific research template integrating:

- **PyTorch Lightning** — structured training loop
- **Hydra** — composable config management
- **Optuna** — hyperparameter optimization
- **MLflow** — experiment tracking & artifact management
- **SLURM / submitit** — HPC cluster submission

---

## Directory Structure

```
project_root/
├── src/
│   ├── main.py                    # Hydra entry point
│   ├── train.py                   # run_train / run_optuna / run_infer
│   ├── entrypoints.py             # argparse CLI adapters
│   ├── data/
│   │   ├── preprocess.py          # raw -> tensor transforms (shared contract)
│   │   └── wine_datamodule.py     # example: sklearn wine dataset
│   ├── methods/
│   │   ├── registry.py            # @register decorator + get_method()
│   │   ├── mlp/                   # primary method
│   │   │   ├── model.py           # pure nn.Module (MLP)
│   │   │   ├── lit_module.py      # LightningModule
│   │   │   └── config.yaml        # method-local defaults
│   │   └── linear/                # baseline method
│   │       ├── model.py
│   │       ├── lit_module.py
│   │       └── config.yaml
│   ├── inference/
│   │   ├── pipeline.py            # predict -> postprocess -> artifacts
│   │   └── analyzers.py           # confusion_matrix, per_class, tsne, calibration
│   ├── callbacks/
│   │   └── oom_handler.py         # CUDA OOM recovery callback
│   ├── utils/
│   │   ├── mlflow_utils.py        # unified MLflow API
│   │   ├── logging.py             # get_logger()
│   │   └── metrics.py             # accuracy helpers
│   └── scripts/
│       └── export_ckpt.py         # .ckpt -> state_dict .pt
│
├── configs/
│   ├── config.yaml                # top-level defaults
│   ├── dataset/wine.yaml
│   ├── method/mlp.yaml, linear.yaml
│   ├── trainer/default.yaml
│   ├── logger/mlflow.yaml
│   ├── mode/train.yaml, debug.yaml, infer.yaml, optuna.yaml
│   ├── optuna/default.yaml
│   └── hydra/launcher/slurm.yaml
│
├── tests/
│   ├── conftest.py                # shared Hydra fixtures
│   ├── test_instantiation.py      # acceptance 1: instantiation checks
│   └── test_forward.py            # acceptance 2-4: forward + fast_dev_run
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Train (debug — fast_dev_run)

```bash
python -m src.main mode=debug dataset=wine method=mlp
```

### Train (full run)

```bash
python -m src.main dataset=wine method=mlp
```

### Train with linear baseline

```bash
python -m src.main dataset=wine method=linear
```

### Hyperparameter Search (Optuna)

```bash
python -m src.main mode=optuna dataset=wine method=mlp optuna.n_trials=20
```

### Inference + Analysis

```bash
python -m src.main mode=infer inference.checkpoint_path=checkpoints/mlp/<timestamp>/best.ckpt
```

### SLURM Multirun

```bash
python -m src.main -m hydra/launcher=slurm method=mlp,linear method.learning_rate=1e-3,1e-4
```

### Export checkpoint

```bash
python -m src.scripts.export_ckpt checkpoints/mlp/<ts>/best.ckpt
```

### Run tests

```bash
pytest tests/ -v
```

---

## Acceptance Criteria (from design spec)

| # | Check | Command |
|---|-------|---------|
| 1 | Instantiation | `pytest tests/test_instantiation.py` |
| 2 | Single-batch forward | `pytest tests/test_forward.py::test_mlp_single_batch_forward` |
| 3 | Fast dev run | `python -m src.main mode=debug dataset=wine method=mlp` |
| 4 | Full run (val_loss decreases) | `python -m src.main dataset=wine method=mlp` |
| 5 | Optuna 3 trials | `python -m src.main mode=optuna optuna.n_trials=3` |
| 6 | Inference + analyzers | `python -m src.main mode=infer inference.checkpoint_path=<ckpt>` |
| 7 | SLURM submit | `python -m src.main -m hydra/launcher=slurm ...` |

---

## Adding a New Method

1. Create `src/methods/<name>/model.py` — pure `nn.Module`, no training logic
2. Create `src/methods/<name>/lit_module.py` — inherit `pl.LightningModule`, decorate with `@register("<name>")`
3. Create `configs/method/<name>.yaml` — set `_target_` and all hyperparameters
4. Run: `python -m src.main method=<name>`

### predict_step contract (required)

```python
def predict_step(self, batch, batch_idx, dataloader_idx=0):
    return {
        "pred": Tensor,       # (N,) predicted class indices
        "prob": Tensor,       # (N, C) softmax probabilities
        "meta": {"true_label": Tensor, ...}
    }
```

---

## Key Design Constraints

- Models **must not** read `cfg` directly; receive only explicit constructor args
- Network structure lives **only** in `methods/<name>/model.py`
- `inference/pipeline.py` **must not** modify training flow or cfg
- Checkpoint naming: `{method}-{dataset}-{timestamp}-epoch={E:02d}-val={val:.4f}.ckpt`
- LLM/Diffusion extension: replace `methods/<name>/` contents only; Trainer/DataModule/Pipeline reuse as-is
