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
├── docs/                              # Quarto research documentation site
│   ├── _quarto.yml                    # site config (sidebar, theme, nav)
│   ├── index.qmd                      # dashboard / navigation hub
│   ├── task_definition.qmd            # research goal and scope
│   ├── literature_review.qmd          # background and citations
│   ├── claims.qmd                     # testable hypotheses & claim tracker
│   ├── method/
│   │   ├── overview.qmd               # architecture & component summary
│   │   ├── component_a.qmd            # per-component spec
│   │   └── component_b.qmd
│   ├── experiments/
│   │   ├── overview.qmd               # experiment navigation
│   │   ├── main.qmd                   # main experiment results
│   │   ├── ablation.qmd               # ablation study
│   │   └── hyperparam.qmd             # hyperparameter sensitivity
│   ├── references.bib                 # citation database
│   └── styles/                        # custom CSS & CSL
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

### Inference + Visual Analysis (`mode=infer`)

Produces confusion matrix / tSNE / calibration PNGs and `predictions.pt`. Does
not compute or log any scalar metrics beyond what training already logged.

```bash
python -m src.main mode=infer inference.checkpoint_path=checkpoints/mlp/<timestamp>/best.ckpt
```

### Post-hoc Re-evaluation with New Metrics (`mode=eval`)

Re-runs `trainer.test()` on a saved checkpoint **and** runs the scalar
metric_analyzers registry (`macro_f1`, `ece`, `brier`, …), logging everything
to a new MLflow run tagged with `parent_run_id`, `checkpoint_sha256`,
`split_seed`, etc. Use this when you want to evaluate a checkpoint with a
metric that did not exist at training time — no retraining required.

```bash
# Add new metrics without touching any training code
python -m src.main mode=eval inference.checkpoint_path=checkpoints/mlp/<ts>/best.ckpt \
    'inference.metric_analyzers=[macro_f1,ece,brier,classification_report_flat]'

# Strict mode: raise if the checkpoint's embedded eval_snapshot disagrees with
# the current cfg.dataset (split_seed, val_split, test_split, dataset target).
python -m src.main mode=eval inference.checkpoint_path=<ckpt> eval.strict_snapshot=true
```

Consistency contract: every checkpoint saved by `run_train` embeds an
`eval_snapshot` dict recording the exact data-split conditions it was trained
under. `run_eval` compares this against the current cfg and either warns
(default) or raises (`eval.strict_snapshot=true`). This guarantees that
post-hoc metrics computed on a checkpoint share identical evaluation
conditions with both the original training run and any other baseline run
sharing the same `split_seed` + dataset config.

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
| 6 | Inference + visual analyzers | `python -m src.main mode=infer inference.checkpoint_path=<ckpt>` |
| 7 | Post-hoc re-eval with new metrics | `python -m src.main mode=eval inference.checkpoint_path=<ckpt>` |
| 8 | SLURM submit | `python -m src.main -m hydra/launcher=slurm ...` |

---

## Research Documentation (Quarto)

The `docs/` directory contains a **Quarto website** for structured research documentation — task definition, literature review, claims, method specs, and experiment results.

### Render & Preview

```bash
cd docs && quarto preview   # live preview with hot reload
cd docs && quarto render     # build static site to docs/_site/
```

### Site Structure

| Page | Purpose |
|------|---------|
| `index.qmd` | Dashboard — key results & navigation |
| `task_definition.qmd` | Research goal, scope, constraints |
| `literature_review.qmd` | Background, related work, citations |
| `claims.qmd` | Testable hypotheses & claim status tracker |
| `method/overview.qmd` | Architecture overview & component table |
| `method/component_*.qmd` | Per-component detailed specs |
| `experiments/overview.qmd` | Experiment navigation hub |
| `experiments/main.qmd` | Main experiment results |
| `experiments/ablation.qmd` | Ablation study |
| `experiments/hyperparam.qmd` | Hyperparameter sensitivity |

### After Running Experiments

1. Fill results into the corresponding `experiments/*.qmd` (mean ± std)
2. Update the Claim Status Tracker in `claims.qmd`
3. Update Key Results metrics in `index.qmd`

### Adding a New Experiment

1. Create `docs/experiments/<name>.qmd` following the structure of `main.qmd`
2. Add the new page to the sidebar in `docs/_quarto.yml`
3. Add a nav-card to `experiments/overview.qmd`
4. Link the experiment to a claim in `claims.qmd`

### Adding a New Method Component

1. Copy `docs/method/component_a.qmd` → `docs/method/component_<name>.qmd`
2. Add the page to `docs/_quarto.yml` under the Method section
3. Add a row to the Components table in `method/overview.qmd`

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
